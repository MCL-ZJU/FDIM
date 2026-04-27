import os
import time

from utils.utils import map_score, get_unique_name
from utils.convert_color import rgb2yuv_resize, yuv2yuv_resize, get_ffmpeg_pix_fmt, get_vmaf_pix_fmt
from vmaf.calc_vmaf import calc_vmaf
from dist.test_video_gen import evaluate_video_gen


def run_model(
        ref_path,
        dis_path,
        model,
        device,
        ref_fmt="rgb",
        dis_fmt="rgb",
        ref_width=0,
        ref_height=0,
        dis_width=0,
        dis_height=0,
        ref_bit_depth=0,
        dis_bit_depth=0,
        resize_flag=False,
        resize_method='default',
        input_resolution='ori',
        calc_interval=1,
        n_threads=32,
        vmaf_version=None,
        video_temp_path='./video_temp/',
        preprocess="none",
        display_model=None):

    temp_root = video_temp_path
    os.makedirs(temp_root, exist_ok=True)

    ref_name = os.path.split(ref_path)[-1]
    dis_name = os.path.split(dis_path)[-1]

    t20 = time.time()

    max_bit_depth = max(ref_bit_depth, dis_bit_depth)
    if ref_fmt != 'rgb' and ref_bit_depth == max_bit_depth:
        ref_path_yuv = ref_path
        ref_yuv_fmt = ref_fmt
    else:
        unique_suffix_ref = get_unique_name()
        ref_path_yuv = os.path.join(temp_root, f'{ref_name}_{unique_suffix_ref}_ref.yuv')
        ref_yuv_fmt = get_ffmpeg_pix_fmt(bit_depth=max_bit_depth, video_fmt="yuv420p")
        if ref_fmt != 'rgb':
            yuv2yuv_resize(
                ref_path,
                ref_path_yuv,
                input_width=ref_width,
                input_height=ref_height,
                input_format=ref_fmt,
                output_width=ref_width,
                output_height=ref_height,
                output_fmt=ref_yuv_fmt,
                resize_flag=False,
                resize_method=resize_method,
                threads=n_threads,
                temp_root=temp_root)
        else:
            rgb2yuv_resize(
                ref_path,
                ref_path_yuv,
                output_fmt=ref_yuv_fmt,
                resize_flag=False,
                threads=n_threads,
                temp_root=temp_root)

    if dis_fmt != 'rgb':
        if resize_flag or dis_bit_depth != max_bit_depth or dis_fmt != ref_yuv_fmt:
            unique_suffix_dis = get_unique_name()
            dis_path_yuv = os.path.join(temp_root, f'{dis_name}_{unique_suffix_dis}_dis.yuv')
            yuv2yuv_resize(
                dis_path,
                dis_path_yuv,
                input_width=dis_width,
                input_height=dis_height,
                input_format=dis_fmt,
                output_width=ref_width,
                output_height=ref_height,
                output_fmt=ref_yuv_fmt,
                resize_flag=resize_flag,
                resize_method=resize_method,
                threads=n_threads,
                temp_root=temp_root)
            dis_yuv_fmt = ref_yuv_fmt
        else:
            dis_path_yuv = dis_path
            dis_yuv_fmt = dis_fmt
    else:
        unique_suffix_dis = get_unique_name()
        dis_path_yuv = os.path.join(temp_root, f'{dis_name}_{unique_suffix_dis}_dis.yuv')
        rgb2yuv_resize(
            dis_path,
            dis_path_yuv,
            output_width=ref_width,
            output_height=ref_height,
            output_fmt=ref_yuv_fmt,
            resize_flag=resize_flag,
            resize_method=resize_method,
            threads=n_threads,
            temp_root=temp_root)
        dis_yuv_fmt = ref_yuv_fmt

    assert dis_yuv_fmt == ref_yuv_fmt

    t21 = time.time()
    print("===========%s used time rgb2yuv %ss" % (dis_name, t21 - t20))
    if not vmaf_version:
        if ref_width * ref_height >= 3840 * 2160:
            vmaf_version = "vmaf_4k_v0.6.1"
        else:
            vmaf_version = "vmaf_v0.6.1"

    vmaf_pix_fmt = get_vmaf_pix_fmt(ref_yuv_fmt)

    vmaf_score = calc_vmaf(
        ref_video_path=ref_path_yuv,
        dis_video_path=dis_path_yuv,
        width=ref_width,
        height=ref_height,
        pix_fmt=vmaf_pix_fmt,
        bit_depth=max_bit_depth,
        model_version=vmaf_version,
        subsample=1,
        n_threads=n_threads,
        video_temp_path=video_temp_path)

    map_vmaf_score = map_score(vmaf_score, score_flag="VMAF")

    t22 = time.time()
    print("===========%s used time for calc vmaf: %ss" % (dis_name, t22 - t21))
    deep_score = evaluate_video_gen(
        data_ref_path=ref_path,
        data_dis_path=dis_path,
        model=model,
        device=device,
        ref_fmt=ref_fmt,
        dis_fmt=dis_fmt,
        ref_width=ref_width,
        ref_height=ref_height,
        ref_bit_depth=ref_bit_depth,
        dis_bit_depth=dis_bit_depth,
        dis_width=dis_width,
        dis_height=dis_height,
        resize_flag=resize_flag,
        resize_method=resize_method,
        input_resolution=input_resolution,
        calc_interval=calc_interval,
        preprocess=preprocess,
        display_model=display_model)

    t23 = time.time()
    print("===========%s used time for calc dist: %ss" % (dis_name, t23 - t22))


    fdim_score = (deep_score + map_vmaf_score) / 2
    print("deep_score: ", deep_score)
    print("map_vmaf_score: ", map_vmaf_score)
    print("fdim_score: ", fdim_score)


    if temp_root in dis_path_yuv and dis_path_yuv.endswith("_dis.yuv"):
        os.remove(dis_path_yuv)

    if temp_root in ref_path_yuv and ref_path_yuv.endswith("_ref.yuv"):
        os.remove(ref_path_yuv)

    return fdim_score, vmaf_score, deep_score
