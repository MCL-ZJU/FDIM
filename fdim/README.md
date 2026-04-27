# FDIM Package Notes

This directory contains the core FDIM implementation.

For installation, usage, HDR preprocessing options, and example commands, see the repository-level [README](../README.md).

Useful locations in this package:

- `dist/`: deep branch, HDR preprocessing path, and integrated third-party `pycvvdp` utilities.
- `utils/`: video I/O, color conversion, and shared helpers.
- `run_fdim.py`: main FDIM inference pipeline.
- `run_fdim_dataset_infer.py`: dataset-level inference entry.
- `run_fdim_single_infer.py`: single-video inference entry.

Reference notes:

- `pycvvdp` under `dist/pycvvdp/` is a locally integrated third-party module adapted from the ColorVideoVDP project.
- `PU21` refers to the perceptually uniform HDR encoding proposed in "PU21: A novel perceptually uniform encoding for adapting existing quality metrics for HDR" (Mantiuk and Azimi, PCS 2021).
