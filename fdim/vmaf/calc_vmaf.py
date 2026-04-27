
import os
import subprocess
import platform
import urllib.request
from urllib.error import URLError

from fdim.utils.utils import get_unique_name

def _bytes_per_frame(width, height, pix_fmt, bit_depth):
    if pix_fmt == "420":
        samples = width * height * 3 / 2
    elif pix_fmt == "422":
        samples = width * height * 2
    elif pix_fmt == "444":
        samples = width * height * 3
    else:
        raise ValueError(f"Unsupported pixel_format {pix_fmt}")

    bytes_per_sample = 1 if bit_depth <= 8 else 2
    return int(samples * bytes_per_sample)

def _validate_yuv_file(path, width, height, pix_fmt, bit_depth, label):
    file_size = os.path.getsize(path)
    bpf = _bytes_per_frame(width, height, pix_fmt, bit_depth)
    if bpf <= 0:
        raise ValueError(f"Invalid bytes_per_frame={bpf}")
    if file_size % bpf != 0:
        frames = file_size / bpf
        raise RuntimeError(
            f"{label} YUV file size is not aligned to full frames. "
            f"path={path}, size={file_size}, bytes_per_frame={bpf}, frames={frames:.6f}, "
            f"width={width}, height={height}, pixel_format={pix_fmt}, bitdepth={bit_depth}"
        )

def download_model(url, model_path):
    if not os.path.exists(model_path):
        try:
            print("Downloading VMAF...")
            urllib.request.urlretrieve(url, model_path)
            print(f"File downloaded successfully: {model_path}")
        except:
            raise URLError(f"Please download model file from {url}, and then put it in fdim/vmaf folder")
          

def calc_vmaf(ref_video_path, dis_video_path, width, height, pix_fmt="420", bit_depth=8, model_version=None, subsample=1,
             n_threads=1, video_temp_path='./vmaf/temp'):

    os_system = platform.system().lower()
    print("os_system: ", os_system)

    if os_system == "linux":
        url = "https://github.com/Netflix/vmaf/releases/download/v3.0.0/vmaf"
        model_path = "fdim/vmaf/vmaf"
    elif os_system == "windows":
        url = "https://github.com/Netflix/vmaf/releases/download/v3.0.0/vmaf.exe"
        model_path = r"fdim\vmaf\vmaf.exe"
    else:
        raise Exception(f"not support system {os_system}")

    download_model(url, model_path)

    if os_system == "linux":
        subprocess.run(['chmod', "+x", model_path])

    dis_video_name = os.path.split(dis_video_path)[1]

    assert os.path.exists(ref_video_path)
    assert os.path.exists(dis_video_path)

    _validate_yuv_file(ref_video_path, width, height, pix_fmt, bit_depth, label="Reference")
    _validate_yuv_file(dis_video_path, width, height, pix_fmt, bit_depth, label="Distorted")

    vmaf_res_dir_tmp = video_temp_path + '_vmaf_result'
    os.makedirs(vmaf_res_dir_tmp, exist_ok=True)
    unique_name = get_unique_name()
    vmaf_result_txt = "{}/{}_{}_vmaf.txt".format(vmaf_res_dir_tmp, dis_video_name, unique_name)

    if not model_version:
        if width * height >= 3840 * 2160:
            model_version = "vmaf_4k_v0.6.1"
        else:
            model_version = "vmaf_v0.6.1"

    input_para = "--reference {} --distorted {} --width {} --height {} --model version={} --pixel_format {} --bitdepth {} --subsample {} --threads {} --output {}".format(
        ref_video_path,
        dis_video_path,
        width,
        height,
        model_version,
        pix_fmt,
        bit_depth,
        subsample,
        n_threads,
        vmaf_result_txt,
    )


    cmd_exe = model_path + " "
    cmd_vmaf = cmd_exe + input_para
    print("cmd vmaf: ", cmd_vmaf)
    proc = subprocess.run(cmd_vmaf, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"VMAF failed (exit_code={proc.returncode}).\n"
            f"cmd={cmd_vmaf}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}"
        )
    if not os.path.exists(vmaf_result_txt):
        raise RuntimeError(
            f"VMAF did not produce output file.\n"
            f"cmd={cmd_vmaf}\n"
            f"expected_output={vmaf_result_txt}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}"
        )
    score = 0
    with open(vmaf_result_txt, "r") as f:
        for line in f.readlines():
            if 'metric name="vmaf"' in line:
                lst = line.split()
                score = float(lst[4].split('"')[1])
    os.remove(vmaf_result_txt)
    return score
