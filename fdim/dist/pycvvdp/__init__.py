"""Vendored pycvvdp utilities used by FDIM HDR preprocessing.

This package is adapted from the ColorVideoVDP project and is kept locally so
FDIM can reuse its display model and video source pipeline for HDR content.
Reference project: https://github.com/gfxdisp/ColorVideoVDP
"""

from .display_model import vvdp_display_geometry, vvdp_display_photometry, vvdp_display_photo_eotf
from .video_source_yuv import video_source_yuv_file
from .video_source_file import video_source_file
from . import utils
from .utils import config_files
