"""Exact FP32 PU21 execution for FDIM's clipped integer video readers.

No LUT, mixed precision, operator fusion, or change in arithmetic order is used.
The general pycvvdp entrypoints retain their input validation and diagnostics.
"""
from collections import OrderedDict
import copy
import json
from pathlib import Path

import torch


class PU21Transform:
    """Display transform with device constants and optional CUDA graph replay.

    Inputs must be finite RGB in [0, 1], in BCFHW layout. The FDIM YUV
    readers establish this contract before calling the display transform.
    Instances belong to one video source and must not be used concurrently.
    """

    def __init__(self, photometry, use_cuda_graph=False):
        self.dm = copy.copy(photometry)
        self.dm.check_input_range = False
        self.use_cuda_graph = use_cuda_graph
        self.signature = None
        self.graph = None

    def _initialize(self, frame):
        from pycvvdp.display_model import XYZ_to_RGB2020, pu21
        self.signature = (frame.device, frame.dtype, tuple(frame.shape), tuple(frame.stride()))
        self.matrix = (torch.as_tensor(XYZ_to_RGB2020, device=frame.device, dtype=frame.dtype)
                       @ torch.tensor(self.dm.rgb2xyz_list, device=frame.device, dtype=frame.dtype))
        self.low = torch.tensor(pu21.L_min, device=frame.device, dtype=frame.dtype)
        self.high = torch.tensor(pu21.L_max, device=frame.device, dtype=frame.dtype)
        self.p = torch.tensor(pu21.par, device=frame.device, dtype=frame.dtype)
        self.scale = torch.tensor(595.39 + 1e-3, device=frame.device, dtype=frame.dtype)
        self.graph = None
        self.static_input = self.static_output = None

    def _evaluate(self, frame):
        linear = self.dm.forward(frame)
        rgb = torch.empty_like(linear)
        for channel in range(3):
            rgb[..., channel, :, :, :] = torch.sum(
                linear * self.matrix[channel, :].view(1, 3, 1, 1, 1),
                dim=-4, keepdim=True)
        x = torch.clamp(rgb, min=self.low, max=self.high)
        p = self.p
        pow_x = torch.pow(x, p[3])
        encoded = p[6] * (torch.pow((p[0] + p[1] * pow_x) /
                                   (1 + p[2] * pow_x), p[4]) - p[5])
        return encoded / self.scale

    def __call__(self, frame, colorspace="RGB2020pu21"):
        if colorspace != "RGB2020pu21":
            raise ValueError("FDIM PU21 transform only supports RGB2020pu21")
        signature = (frame.device, frame.dtype, tuple(frame.shape), tuple(frame.stride()))
        if signature != self.signature:
            self._initialize(frame)
        if not self.use_cuda_graph or frame.device.type != "cuda":
            return self._evaluate(frame)
        if self.graph is None:
            with torch.cuda.device(frame.device):
                self.static_input = torch.empty_strided(
                    frame.shape, frame.stride(), dtype=frame.dtype, device=frame.device)
                self.static_input.copy_(frame)
                stream = torch.cuda.Stream(device=frame.device)
                stream.wait_stream(torch.cuda.current_stream(frame.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._evaluate(self.static_input)
                torch.cuda.current_stream(frame.device).wait_stream(stream)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.static_output = self._evaluate(self.static_input)
        self.static_input.copy_(frame)
        self.graph.replay()
        # Reference and distorted reads share this transform. Never expose the
        # reusable graph buffer, which the next read will overwrite.
        return self.static_output.clone()


def configure_source(source, use_cuda_graph=False):
    """Optimize only readers with a known finite/clipped RGB output contract."""
    if not hasattr(getattr(source, "vs", source), "dm_photometry"):
        return
    from pycvvdp.video_source_file import video_source_video_file, video_reader_yuv_pytorch
    from pycvvdp.video_source_yuv import video_source_yuv_file
    implementation = getattr(source, "vs", source)
    supported = (type(implementation) is video_source_yuv_file or
                 (type(implementation) is video_source_video_file and
                  type(implementation.test_vidr) is video_reader_yuv_pytorch and
                  type(implementation.reference_vidr) is video_reader_yuv_pytorch))
    if supported:
        implementation.apply_dm_and_colour_transform = PU21Transform(
            implementation.dm_photometry, use_cuda_graph)


def skip_frame(source, side, index, device):
    """Advance compressed streams without unpacking, transfer or PU21 math."""
    from pycvvdp.video_source_file import video_source_video_file
    from pycvvdp.video_source_yuv import video_source_yuv_file
    implementation = getattr(source, "vs", source)
    if type(implementation) is video_source_yuv_file:
        # Raw YUV is memory mapped and indexed, so no decoder needs advancing.
        return
    if type(implementation) is video_source_video_file:
        reader = getattr(implementation, side + "_vidr")
        if index != reader.curr_frame + 1:
            raise RuntimeError("PU21 frame skipping must preserve sequential reads")
        if reader.get_frame() is None:
            raise RuntimeError("Could not read frame {}".format(index))
        return
    # Keep image sequences and third-party/custom sources compatible.
    getattr(source, "get_" + side + "_frame")(
        index, device=device, colorspace="RGB2020pu21")


def reference_key(path, source, device, width, height, bits, resolution, interval):
    implementation = getattr(source, "vs", source)
    stat = Path(path).stat()
    dm = implementation.dm_photometry
    # Include actual display values, not just its user-visible name. Ignore
    # lazily allocated device tensors; their values derive from rgb2xyz_list.
    display = {key: value for key, value in vars(dm).items()
               if not isinstance(value, torch.Tensor)}
    reader = getattr(implementation, "reference_vidr", None)
    reader_config = tuple(getattr(reader, key, None) for key in (
        "width", "height", "bit_depth", "color_space", "chroma_ss",
        "resize_width", "resize_height", "resize_fn"))
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return (str(Path(path).resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns, torch.backends.cuda.matmul.allow_tf32,
            json.dumps(display, sort_keys=True), str(device), torch.__version__,
            width, height, bits, resolution, interval,
            reader_config,
            str(getattr(implementation, "resize_resolution", None)))


class PU21ReferenceCache:
    """Optional bounded, lossless host-memory cache for repeated references.

    Reuse this object across evaluate_video_gen/run_model calls. A 4K FP32
    RGB frame occupies about 95 MiB. Caching is disabled unless supplied.
    This object is not thread safe; each inference worker needs its own cache.
    """

    def __init__(self, max_bytes=512 * 1024**2):
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes = int(max_bytes)
        self.bytes = 0
        self.frames = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key, device):
        value = self.frames.get(key)
        if value is None:
            self.misses += 1
            return None
        self.frames.move_to_end(key)
        self.hits += 1
        return value.to(device=device)

    def put(self, key, frame):
        size = frame.numel() * frame.element_size()
        if size > self.max_bytes:
            return
        old = self.frames.pop(key, None)
        if old is not None:
            self.bytes -= old.numel() * old.element_size()
        while self.bytes + size > self.max_bytes:
            _, old = self.frames.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
        self.frames[key] = frame.detach().to(device="cpu", copy=True)
        self.bytes += size

    def clear(self):
        self.frames.clear()
        self.bytes = 0
