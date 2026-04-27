#!/bin/bash
# install FDIM package

set -e
set -x


# virtutalenv
conda create -n fdim python=3.9.20 -y
eval "$(conda shell.bash hook)"
conda activate fdim

# FDIM
pip install -e .

python -c "import shutil; import sys; sys.exit(0 if shutil.which('ffmpeg') else 1)" || (echo "ffmpeg not found in PATH. Please install ffmpeg first." && exit 1)

