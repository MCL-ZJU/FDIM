# Exact HDR preprocessing

FDIM retains the original display model, PQ/HLG transfer functions, color matrix,
PU21 parameters, normalization, resize order and FP32 arithmetic. Model weights,
frame batch size, selected frame indices, temporal averaging and VMAF are unchanged.

## Modes

Use `--preprocess pu21 --display_model standard_hdr_pq_tv` (or the corresponding
HLG/display model) with one of these options on the repository's single/batch
entrypoints:

- `--pu21_mode eager` (default): reuse device constants and omit redundant value
  diagnostics for the integer YUV readers whose output is already finite and
  clipped to [0, 1]. Non-scored compressed frames are drained from the decoder
  without unpacking, GPU transfer or display/PU21 conversion.
- `--pu21_mode cuda_graph`: additionally capture and replay the original display
  and PU21 operators. This does not use `torch.compile`, fusion, a LUT, reduced
  precision or a different model batch size. Graph setup is per video source;
  very short clips can be slower. On CPU this uses the eager implementation.
- `--pu21_mode legacy`: retain the vendored arithmetic and diagnostics as a
  comparison/rollback path. Compressed videos advance sequentially and convert
  the skipped frames. This fixes the previous compressed-HDR random-seek error
  when `calc_interval > 1`. Raw YUV remains indexed and skips unused frames in
  all modes, so it does not claim an artificial sampling speedup.

These options are separate from `--inference_mode`, which controls the existing
SDR streaming/VMAF pipeline. Unsupported source implementations retain their
original transform and diagnostic behavior.

The graph owns its static input/output buffers and returns a separate output on
every call to prevent a later distorted-frame read overwriting a reference frame.
Changing device, dtype, dimensions or strides rebuilds its constants/graph.
Each video source must stay in one inference thread.

## Optional reference cache (Python API)

```python
from dist.pu21 import PU21ReferenceCache

cache = PU21ReferenceCache(max_bytes=4 * 1024**3)
# Reuse the same object for multiple distorted versions of a reference:
score = evaluate_video_gen(
    ref_path, dis_path, model, device,
    preprocess="pu21", display_model="standard_hdr_pq_tv",
    reference_frame_cache=cache,
)
```

`run_model` also accepts `reference_frame_cache`. Cached values retain their FP32
representation in host memory; there is no quantization or lossy serialization.
The cache is optional, bounded by `max_bytes`, and uses LRU eviction. Each 4K RGB
frame requires about 95 MiB, so size the cache for the sampled reference frames.
A cache too small to hold a repeatedly scanned reference can provide no benefit.
Cold population and host/device copies may outweigh savings for a single pair.

Cache identity includes reference file path/stat identity, actual display values,
reader dimensions/bit depth, resize/sampling settings, device, PyTorch version and
matrix precision setting. File identity/stat changes invalidate its entries.
Inputs must remain immutable while a cache is in use: an in-place edit that
preserves filesystem timestamps is not detectable without hashing the full file.
Call `clear()` after editing an input or changing other numerical backend
settings. One cache belongs to one inference worker.
Cached references still drain compressed decoder frames to preserve alignment;
this optimization saves reference preprocessing, not compressed decoding.

## Validation and timing

Run `python -m pytest tests/test_pu21.py tests/test_official_inference.py -q`.
CUDA-specific cases run automatically when CUDA is available. Tests compare
PU21 tensors, each raw frame score and the mapped video score with zero tolerance,
and cover PQ/HLG, raw/encoded 10-bit video, full/interval sampling, graph buffer
ownership/shape changes, cache eviction/invalidation and decoder errors.

For GPU measurements, supply two or more real HDR pairs in a JSON file:

```json
[
  {"name": "pq", "ref": "/data/ref.mp4", "dis": "/data/dis.mp4",
   "display_model": "standard_hdr_pq_tv"}
]
```

```sh
python scripts/benchmark_pu21.py --cases cases.json --output benchmark.json \
  --frames 32 --repeats 3 --threads 8
```

The benchmark requires 4K 10-bit input, stages inputs on local storage outside the
measured interval, and reports synchronized CUDA timings. Deep measurements
include source creation, decoder startup, transfer, preprocessing, graph setup
and model inference. Each mode is warmed up, measurement order alternates, and
individual repetitions plus medians are retained. The optional cache row measures
reuse after population; it is not a cold-cache speed claim. Pure transform timing
measures steady-state device execution separately. Full FDIM spot checks also
compare the unchanged VMAF/fusion results; their single-run timings are not used
as statistically stable throughput estimates.

Every reported timed inference must exactly match the baseline frame scores.
A final `complete: true` distinguishes a finished report from partial results.
Hardware/runtime-specific evidence is recorded alongside the benchmark report;
no cross-device or cross-version bitwise equivalence is implied.
