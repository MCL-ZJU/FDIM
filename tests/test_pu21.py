"""PU21 arithmetic, decoder ordering, CUDA buffer ownership and cache regressions."""
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "fdim"), str(ROOT / "fdim/dist")]
from dist import test_video_gen as video
from dist.deep_model import Model
from dist.pu21 import PU21Transform, PU21ReferenceCache, configure_source, skip_frame, reference_key
from pycvvdp.display_model import vvdp_display_photometry

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("display", ["standard_hdr_pq_tv", "standard_hdr_hlg_tv"])
@pytest.mark.parametrize("graph", [False, True])
def test_transform_is_bitwise_equal_and_outputs_are_owned(device, display, graph):
    torch.set_num_threads(2)
    dm = vvdp_display_photometry.load(display, config_paths=[])
    transform = PU21Transform(dm, graph)
    previous = []
    for h, w in [(32, 64), (32, 64), (33, 47), (33, 47)]:
        # Noncontiguous BCFHW, as emitted by the real HWC decoder.
        x = torch.rand(h, w, 3, device=device).permute(2, 0, 1)[None, :, None]
        x[..., 0, 0] = 0
        x[..., 0, 1] = 1
        expected = dm.source_2_target_colourspace(x, "RGB2020pu21")
        actual = transform(x)
        assert torch.equal(expected, actual)
        previous.append((actual, actual.clone()))
    assert all(torch.equal(a, b) for a, b in previous)
    assert not hasattr(dm, "check_input_range")


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    value = Model().eval()
    state = torch.load(ROOT / "fdim/dist/checkpoints/dist_5.0.0.ckpt", map_location="cpu")
    value.load_state_dict(state.get("state_dict", state), strict=True)
    return value


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pu21")
    rng = np.random.default_rng(21)
    for side in ("ref", "dis"):
        raw = directory / (side + "_64x64_10b_2020.yuv")
        rng.integers(64, 941, (7, 64 * 64 * 3 // 2), dtype=np.uint16).tofile(raw)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "yuv420p10le",
                        "-video_size", "64x64", "-framerate", "30", "-i", str(raw),
                        "-c:v", "ffv1", "-threads", "1", "-colorspace", "bt2020nc",
                        "-color_trc", "smpte2084", "-color_primaries", "bt2020",
                        str(directory / (side + ".mkv"))], check=True)
    return directory


@pytest.fixture
def small_source(monkeypatch):
    original = video._build_pu21_video_source
    def build(*args, **kwargs):
        source = original(*args, **kwargs)
        implementation = getattr(source, "vs", source)
        implementation.resize_resolution = (64, 64)
        for name in ("reference_vidr", "test_vidr"):
            reader = getattr(implementation, name)
            reader.resize_height = reader.resize_width = 64
        return source
    monkeypatch.setattr(video, "_build_pu21_video_source", build)
    return build


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("display", ["standard_hdr_pq_tv", "standard_hdr_hlg_tv"])
@pytest.mark.parametrize("suffix", ["_64x64_10b_2020.yuv", ".mkv"])
@pytest.mark.parametrize("interval", [1, 3])
def test_pixels_and_scores_match_baseline(media, model, small_source, device, display, suffix, interval):
    paths = [media / (side + suffix) for side in ("ref", "dis")]
    kwargs = dict(display_model=display, frame_limit=7, verbose=False, width=64, height=64, bit_depth=10)
    baseline = small_source(paths[1], paths[0], **kwargs)
    optimized = small_source(paths[1], paths[0], **kwargs)
    configure_source(optimized, use_cuda_graph=device == "cuda")
    try:
        for index in range(7):
            for side in ("reference", "test"):
                expected = getattr(baseline, "get_" + side + "_frame")(index, device, "RGB2020pu21")
                if index % interval:
                    skip_frame(optimized, side, index, device)
                else:
                    actual = getattr(optimized, "get_" + side + "_frame")(index, device, "RGB2020pu21")
                    assert torch.equal(expected, actual)
    finally:
        for source in (baseline, optimized):
            if hasattr(source, "close"):
                source.close()
    options = dict(preprocess="pu21", display_model=display, ref_width=64, ref_height=64,
                   ref_bit_depth=10, calc_interval=interval, frame_limit=7, return_frame_scores=True)
    model = model.to(device)
    old = video.evaluate_video_gen(*paths, model, device, pu21_mode="legacy", **options)
    cache = PU21ReferenceCache(1024**2)
    for mode in ("eager", "cuda_graph"):
        new = video.evaluate_video_gen(*paths, model, device, pu21_mode=mode,
                                       reference_frame_cache=cache, **options)
        assert old == new
    assert cache.hits == len(range(0, 7, interval))
    assert cache.bytes <= cache.max_bytes


def test_reference_cache_is_bounded_and_owns_data():
    cache = PU21ReferenceCache(16)
    frame = torch.arange(4, dtype=torch.float32)
    cache.put("a", frame)
    frame.zero_()
    assert torch.equal(cache.get("a", "cpu"), torch.arange(4, dtype=torch.float32))
    cache.put("b", frame)
    assert cache.get("a", "cpu") is None
    assert cache.bytes == 16
    cache.put("large", torch.zeros(5))
    assert cache.get("large", "cpu") is None
    cache.clear()
    assert cache.bytes == 0 and not cache.frames


def test_out_of_range_general_display_path_retains_clipping():
    dm = vvdp_display_photometry.load("standard_hdr_pq_tv", config_paths=[])
    x = torch.tensor([-1., 0., 1., 2.]).view(1, 1, 1, 1, 4).expand(1, 3, 1, 1, 4)
    assert torch.equal(dm.forward(x), dm.forward(x.clamp(0, 1)))


def test_cache_key_tracks_file_display_and_resize(media, small_source, tmp_path):
    import shutil
    path = tmp_path / "ref_64x64_10b_2020.yuv"
    shutil.copyfile(media / "ref_64x64_10b_2020.yuv", path)
    source = small_source(str(path), str(path), display_model="standard_hdr_pq_tv",
                          frame_limit=7, verbose=False, width=64, height=64, bit_depth=10)
    def key():
        return reference_key(path, source, "cpu", 64, 64, 10, "ori", 1)
    before = key()
    source.dm_photometry.Y_peak += 1
    assert key() != before
    source.dm_photometry.Y_peak -= 1
    assert key() == before
    source.resize_resolution = (128, 128)
    assert key() != before
    source.resize_resolution = (64, 64)
    # Atomic replacement changes file identity even on filesystems whose
    # timestamps cannot distinguish two writes in the same clock tick.
    replacement = tmp_path / "replacement.yuv"
    data = bytearray(path.read_bytes())
    data[:2] = b"\x00\x00"
    replacement.write_bytes(data)
    replacement.replace(path)
    assert key() != before


def test_skip_preserves_decoder_errors_and_does_not_convert(media, small_source):
    source = small_source(media / "dis.mkv", media / "ref.mkv",
                          display_model="standard_hdr_pq_tv", frame_limit=7, verbose=False)
    def unexpected(*args, **kwargs):
        raise AssertionError("A skipped frame must not be converted")
    source.vs._prepare_frame = unexpected
    try:
        with pytest.raises(RuntimeError, match="sequential"):
            skip_frame(source, "test", 1, "cpu")
        for i in range(7):
            skip_frame(source, "test", i, "cpu")
        with pytest.raises(RuntimeError, match="Could not read"):
            skip_frame(source, "test", 7, "cpu")
    finally:
        source.close()
