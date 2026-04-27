import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.reader import Reader
from utils.utils import map_score
from .img_process import AdaptiveResize, resize_img


def _is_yuv_file(path):
    return Path(path).suffix.lower() == ".yuv"


def _to_bchw(frame):
    t = frame[0] if isinstance(frame, (tuple, list)) else frame
    if t.dim() == 5 and t.shape[2] == 1:
        t = t.squeeze(2)
    if t.dim() == 4:
        if t.shape[0] == 3:
            t = t.permute(1, 0, 2, 3)
        return t
    if t.dim() == 3:
        return t.unsqueeze(0)
    raise ValueError(f"Unexpected frame shape: {tuple(t.shape)}")


def _adaptive_resize_bchw(x, size):
    if not size or size <= 0:
        return x
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if min(h, w) < size:
        return x
    scale = float(size) / float(min(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _build_pu21_video_source(test_path, ref_path, *, display_model, frame_limit, verbose, width=None, height=None, bit_depth=None):
    import sys
    dist_dir = Path(__file__).resolve().parent
    if str(dist_dir) not in sys.path:
        sys.path.insert(0, str(dist_dir))
    # The vendored pycvvdp package comes from ColorVideoVDP and provides the
    # display-aware HDR video source used by FDIM's PU21 preprocessing path.
    import pycvvdp

    display_photometry = pycvvdp.vvdp_display_photometry.load(display_model, config_paths=[])
    display_geometry = pycvvdp.vvdp_display_geometry.load(display_model, config_paths=[])

    if _is_yuv_file(test_path):
        return pycvvdp.video_source_yuv_file(
            str(test_path),
            str(ref_path),
            display_photometry=display_photometry,
            full_screen_resize="nearest",
            resize_resolution=display_geometry.resolution,
            frames=frame_limit,
            verbose=verbose,
            width=width,
            height=height,
            bit_depth=bit_depth,
        )

    return pycvvdp.video_source_file(
        str(test_path),
        str(ref_path),
        display_photometry=display_photometry,
        config_paths=[],
        full_screen_resize="nearest",
        resize_resolution=display_geometry.resolution,
        frames=frame_limit,
        fps=None,
        frame_range=None,
        preload=False,
        ffmpeg_cc=False,
        verbose=verbose,
    )


def evaluate_video_gen(
        data_ref_path,
        data_dis_path,
        model,
        device,
        ref_fmt="rgb",
        dis_fmt="rgb",
        ref_width=0,
        ref_height=0,
        dis_width=0,
        dis_height=0,
        ref_bit_depth=8,
        dis_bit_depth=8,
        resize_flag=False,
        resize_method='default',
        input_resolution="ori",
        calc_interval=1,
        transform=None,
        preprocess="none",
        display_model=None):

    # if model is original resolution input, change parameter input_resolution="ori".

    model.eval()
    if preprocess == "pu21":
        # PU21 is the perceptually uniform HDR encoding from:
        # "PU21: A novel perceptually uniform encoding for adapting existing
        # quality metrics for HDR" (Mantiuk and Azimi, PCS 2021).
        if not display_model or str(display_model).lower() == "none":
            display_model = "standard_hdr_pq_tv"
        frame_limit = -1
        vs = None
        try:
            vs = _build_pu21_video_source(
                data_dis_path,
                data_ref_path,
                display_model=display_model,
                frame_limit=frame_limit,
                verbose=False,
                width=ref_width if _is_yuv_file(data_ref_path) else None,
                height=ref_height if _is_yuv_file(data_ref_path) else None,
                bit_depth=ref_bit_depth if _is_yuv_file(data_ref_path) else None,
            )

            _, _, total_frames = vs.get_video_size()
            effective_frames = total_frames

            deep_score = []
            with torch.no_grad():
                for frm_idx in range(0, effective_frames, int(calc_interval)):
                    ref = vs.get_reference_frame(frm_idx, device=device, colorspace="RGB2020pu21")
                    dis = vs.get_test_frame(frm_idx, device=device, colorspace="RGB2020pu21")
                    ref_img_data = _to_bchw(ref).to(dtype=torch.float32).clamp(0.0, 1.0)
                    dis_img_data = _to_bchw(dis).to(dtype=torch.float32).clamp(0.0, 1.0)

                    if resize_flag and (dis_img_data.shape[-2:] != ref_img_data.shape[-2:]):
                        dis_img_data = F.interpolate(
                            dis_img_data,
                            size=ref_img_data.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    if input_resolution.isdigit():
                        ref_img_data = _adaptive_resize_bchw(ref_img_data, int(input_resolution))
                        dis_img_data = _adaptive_resize_bchw(dis_img_data, int(input_resolution))
                    elif input_resolution == "resize1_2":
                        size = int(min(ref_img_data.shape[-2], ref_img_data.shape[-1]) / 2)
                        ref_img_data = _adaptive_resize_bchw(ref_img_data, size)
                        dis_img_data = _adaptive_resize_bchw(dis_img_data, size)
                    elif input_resolution != "ori":
                        raise ValueError("not support input_resolution:%s" % input_resolution)

                    frame_deep_score = model(ref_img_data, dis_img_data)
                    deep_score.append(frame_deep_score.item())

            if not deep_score:
                raise ValueError(f"video path {data_dis_path} or {data_ref_path} not exist:")

            video_deep_score_ori = np.mean(deep_score)
            video_deep_score = map_score(video_deep_score_ori, score_flag="reg")
            return video_deep_score
        finally:
            if vs is not None and hasattr(vs, "close"):
                vs.close()

    if not transform:
        if input_resolution.isdigit():
            input_size = int(input_resolution)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                AdaptiveResize(input_size),
                transforms.ToTensor(),
            ])
        elif input_resolution == "ori":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif input_resolution == "resize1_2":
            input_size = int(min(ref_width, ref_height) / 2)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                AdaptiveResize(input_size),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("not support input_resolution:%s"%input_resolution)

    if ref_fmt != "rgb":
        ref_reader = Reader(
            data_ref_path,
            ref_width,
            ref_height,
            ref_fmt)
        print("ref_fmt: ", ref_fmt)

    else:
        ref_reader = Reader(data_ref_path, ref_width, ref_height)

    if dis_fmt != "rgb":
        dis_reader = Reader(
            data_dis_path,
            dis_width,
            dis_height,
            dis_fmt)
        print("dis_fmt: ", dis_fmt)

    else:
        dis_reader = Reader(data_dis_path, dis_width, dis_height)

    deep_score = []
    frm_idx = 0

    with torch.no_grad():
        while True:
            ref_flag, ref_rgb = ref_reader.next()
            dis_flag, dis_rgb = dis_reader.next()
            if not ref_flag or not dis_flag:
                break

            if frm_idx % calc_interval == 0:
                if resize_flag:
                    dis_rgb = resize_img(
                        dis_rgb,
                        ref_width,
                        ref_height,
                        resize_method)

                ref_img_data = transform(ref_rgb).unsqueeze(0).to(device)
                dis_img_data = transform(dis_rgb).unsqueeze(0).to(device)
                frame_deep_score = model(ref_img_data, dis_img_data)
                deep_score.append(frame_deep_score.item())
            frm_idx += 1
    print("total calculate frames: ", frm_idx)


    if frm_idx == 0:
        print(f"{data_dis_path} exists {os.path.exists(data_dis_path)}")
        print(f"{data_ref_path} exists {os.path.exists(data_ref_path)}")
        raise ValueError(f"video path {data_dis_path} or {data_ref_path} not exist:")

    video_deep_score_ori = np.mean(deep_score)
    video_deep_score = map_score(video_deep_score_ori, score_flag="reg")

    ref_reader.close()
    dis_reader.close()
    return video_deep_score
