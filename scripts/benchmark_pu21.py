"""Compare exact PU21 implementations, including setup, decode and model time.

Example: python scripts/benchmark_pu21.py --cases cases.json --output result.json
Cases: [{"name": "pq", "ref": "/path/ref.mp4", "dis": "/path/dis.mp4",
         "display_model": "standard_hdr_pq_tv"}]
Inputs are copied/decoded to local scratch before measurement. No public dataset
is bundled. Every measured inference must match the baseline scores exactly.
"""
import argparse
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "fdim"), str(ROOT / "fdim/dist")]
from dist.deep_model import Model
from dist.test_video_gen import evaluate_video_gen, _build_pu21_video_source
from dist.pu21 import PU21Transform, PU21ReferenceCache, configure_source, skip_frame
from pycvvdp.display_model import vvdp_display_photometry
from run_fdim import run_model


def synchronize():
    torch.cuda.synchronize()


def timed(function):
    synchronize()
    start = time.perf_counter()
    result = function()
    synchronize()
    return time.perf_counter() - start, result


def transform_benchmark(display, repeats):
    dm = vvdp_display_photometry.load(display, config_paths=[])
    x = torch.rand(2160, 3840, 3, device="cuda").permute(2, 0, 1)[None, :, None]
    functions = {
        "legacy": lambda: dm.source_2_target_colourspace(x, "RGB2020pu21"),
        "eager": lambda transform=PU21Transform(dm): transform(x),
        "cuda_graph": lambda transform=PU21Transform(dm, True): transform(x),
    }
    expected = functions["legacy"]()
    for function in functions.values():
        for _ in range(3):
            assert torch.equal(function(), expected)
    samples = {mode: [] for mode in functions}
    for repeat in range(repeats):
        for mode in (list(functions) if repeat % 2 == 0 else list(functions)[::-1]):
            duration, actual = timed(functions[mode])
            assert torch.equal(actual, expected)
            samples[mode].append(duration)
    return {mode: {"median_s": statistics.median(values), "seconds": values,
                   "max_abs_diff": 0.0} for mode, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for this benchmark")
    torch.set_num_threads(args.threads)
    torch.manual_seed(21)
    model = Model().eval().cuda()
    checkpoint = ROOT / "fdim/dist/checkpoints/dist_5.0.0.ckpt"
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=True)
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "python": platform.python_version(),
              "threads": args.threads, "frames": args.frames, "repeats": args.repeats,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "transforms": {}, "rows": [], "full_metric": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = args.output.with_suffix(".partial.json")
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(args.output)
    with torch.no_grad(), tempfile.TemporaryDirectory(prefix="fdim-pu21-") as scratch:
        for case in json.loads(args.cases.read_text()):
            name, display = case["name"], case["display_model"]
            report["transforms"][name] = transform_benchmark(display, max(9, args.repeats))
            save()
            directory = Path(scratch) / name
            directory.mkdir()
            compressed, raw = [], []
            for side in ("ref", "dis"):
                src = Path(case[side])
                local = directory / (side + src.suffix)
                shutil.copyfile(src, local)
                compressed.append(local)
                dest = directory / (side + "_3840x2160_10b_2020.yuv")
                subprocess.run(["ffmpeg", "-v", "error", "-threads", str(args.threads),
                                "-i", str(local), "-frames:v", str(args.frames),
                                "-pix_fmt", "yuv420p10le", "-f", "rawvideo", str(dest)], check=True)
                raw.append(dest)
            for fmt, paths in [("raw", raw), ("compressed", compressed)]:
                for interval in (1, 30):
                    options = dict(preprocess="pu21", display_model=display, ref_width=3840,
                                   ref_height=2160, dis_width=3840, dis_height=2160,
                                   ref_bit_depth=10, dis_bit_depth=10, input_resolution="ori",
                                   calc_interval=interval, frame_limit=args.frames, return_frame_scores=True)
                    def infer(mode, cache=None):
                        return evaluate_video_gen(*paths, model, "cuda", pu21_mode=mode,
                                                  reference_frame_cache=cache, **options)
                    # One untimed full pass per path/mode; each timed call still
                    # constructs its own source/graph, so setup costs are included.
                    expected = infer("legacy")
                    cache = PU21ReferenceCache(4 * 1024**3)
                    for mode in ("eager", "cuda_graph"):
                        assert infer(mode) == expected
                    assert infer("eager", cache) == expected
                    samples = {mode: [] for mode in ("legacy", "eager", "cuda_graph", "reference_cache")}
                    for repeat in range(args.repeats):
                        modes = list(samples)
                        if repeat % 2:
                            modes.reverse()
                        for mode in modes:
                            duration, actual = timed(lambda: infer(
                                "eager" if mode == "reference_cache" else mode,
                                cache if mode == "reference_cache" else None))
                            assert actual == expected, (name, fmt, interval, mode, "score mismatch")
                            samples[mode].append(duration)
                            print(name, fmt, interval, repeat, mode, round(duration, 4), flush=True)
                    for mode, seconds in samples.items():
                        row = {"case": name, "format": fmt, "interval": interval, "mode": mode,
                               "seconds": seconds, "median_s": statistics.median(seconds),
                               "speedup": statistics.median(samples["legacy"]) / statistics.median(seconds),
                               "raw_scores": expected[1]["raw_scores"], "deep_score": expected[0],
                               "max_abs_diff": 0.0}
                        report["rows"].append(row)
                    save()
            # Full FDIM includes unchanged VMAF. Raw clips have a bounded frame
            # count, and the native 10-bit HDR preprocessing protocol is retained.
            full_options = dict(ref_fmt="yuv420p10le", dis_fmt="yuv420p10le",
                                ref_width=3840, ref_height=2160, dis_width=3840, dis_height=2160,
                                ref_bit_depth=10, dis_bit_depth=10, preprocess="pu21",
                                display_model=display, calc_interval=30, frame_limit=args.frames,
                                n_threads=args.threads, video_temp_path=str(directory / "vmaf"))
            results = {}
            for mode in ("legacy", "eager", "cuda_graph"):
                seconds, scores = timed(lambda: run_model(*map(str, raw), model, "cuda",
                                                          pu21_mode=mode, **full_options))
                results[mode] = {"seconds": seconds, "scores": scores}
            assert results["legacy"]["scores"] == results["eager"]["scores"] == results["cuda_graph"]["scores"]
            report["full_metric"].append({"case": name, "results": results, "max_abs_diff": 0.0})
            save()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
