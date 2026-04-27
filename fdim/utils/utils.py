
import os
import time
import uuid
import json
import numpy as np
import pandas as pd
import torch
from .reader import RgbReader


def get_unique_name():
    timestamp = time.time()
    unique_id = uuid.uuid4().hex[:8]
    unique_name = f'{unique_id}_{timestamp}'

    return unique_name

def get_dict_val(dict, key):
    if key in dict:
        return dict[key]
    else:
        return None

def get_vmaf_lst(lst_res):
    for res in lst_res:
        if len(res.split(':')) == 2:
            k, v = res.split(':')
            if k == 'VMAF_score':
                return float(v[:-1])
    return None


def get_vmaf_score(json_path, format="str", string="VMAF_score"):
    with open(json_path, 'r') as f:
        if format == "str":
            for line in f.readlines():
                if 'Aggregate ' in line:
                    res_split = line.split()
                    return get_vmaf_lst(res_split)
        elif format == "json":
            data = json.load(f)
            vmaf_score = data["aggregate"][string]
            return vmaf_score
        elif format == "xml":
            for line in f.readlines():
                if 'metric name="vmaf" ' in line:
                    lst_res = line.split()
                    for res in lst_res:
                        if "mean" in res:
                            vmaf_score = float(res.split("=")[1].replace("\"", "", 2))
                            return vmaf_score
        else:
            print("not support format %s" % format)

def func(x, b0, b1, b2, b3):
    return b1 + np.divide(b0 - b1, 1 + np.exp(np.divide(b2 - x, np.abs(b3))))

def map_score(input_val, score_flag="reg"):
    if score_flag == "reg":
        b0, b1, b2, b3 = 5.1354, -0.7089, -1.4327, 0.6885
        output_val = func(input_val, b0, b1, b2, b3)
    elif score_flag == "VMAF":
        b0, b1, b2, b3 = 4.97, 1.76, 72.27, 11.05
        output_val = func(input_val, b0, b1, b2, b3)
    else:
        raise ValueError("not support %s" % score_flag)
    return output_val

def readinfo(
        csv_path,
        dis_column='dis_name',
        ref_column='ref_name',
        mos_column='mos',
        ref_root="",
        dis_root="",
        ref_width_column='ref_width',
        ref_height_column='ref_height',
        dis_width_column='dis_width',
        dis_height_column='dis_height',
        ref_fmt='rgb',
        dis_fmt='rgb',
        frame_rate_col="frame_rate",
        ref_bit_depth_column="ref_bit_depth",
        dis_bit_depth_column="dis_bit_depth",
        calc_interval=1):
    df = pd.read_csv(csv_path)
    dis_names = df[dis_column].values.tolist()
    ref_names = df[ref_column].values.tolist()
    if mos_column in df.columns:
        mos_lst = df[mos_column].values.tolist()
    else:
        mos_lst = [0 for i in range(len(df))]
    if ref_width_column in df.columns and ref_height_column in df.columns:
        ref_width_lst = df[ref_width_column].values.tolist()
        ref_height_lst = df[ref_height_column].values.tolist()
    else:
        ref_width_lst, ref_height_lst, _ = get_batch_prop(ref_names, ref_root)
    if dis_width_column in df.columns and dis_height_column in df.columns:
        dis_width_lst = df[dis_width_column].values.tolist()
        dis_height_lst = df[dis_height_column].values.tolist()
    else:
        dis_width_lst, dis_height_lst, _ = get_batch_prop(dis_names, dis_root)
    if frame_rate_col in df.columns:
        frame_rate_lst = df[frame_rate_col].values.tolist()
    else:
        if calc_interval < 0 and ref_fmt == "rgb":
            _, _, frame_rate_lst = get_batch_prop(ref_names, ref_root)
        else:
            frame_rate_lst = [0 for i in range(len(df))]
    if ref_bit_depth_column in df.columns:
        ref_bit_depth_lst = df[ref_bit_depth_column].values.tolist()
    else:
        ref_bit_depth_lst = [8 for i in range(len(df))]

    if dis_bit_depth_column in df.columns:
        dis_bit_depth_lst = df[dis_bit_depth_column].values.tolist()
    else:
        dis_bit_depth_lst = [8 for i in range(len(df))]

    if ref_fmt in df.columns:
        ref_fmt_lst = df[ref_fmt].values.tolist()
    else:
        ref_fmt_lst = ["rgb" for i in range(len(df))]
    if dis_fmt in df.columns:
        dis_fmt_lst = df[dis_fmt].values.tolist()
    else:
        dis_fmt_lst = ["rgb" for i in range(len(df))]

    return df, dis_names, ref_names, mos_lst, ref_width_lst, ref_height_lst, dis_width_lst, dis_height_lst, frame_rate_lst, ref_bit_depth_lst, dis_bit_depth_lst, ref_fmt_lst, dis_fmt_lst

def load_checkpoint(model, ckpt, device):
    if os.path.isfile(ckpt):
        print("[*] loading checkpoint '{}'".format(ckpt))
        checkpoint = torch.load(ckpt, map_location=device)
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("[!] no checkpoint found at '{}'".format(ckpt))
    model = model.to(device)
    return model

def get_rgb_resolution_fps(data_path):
    video_reader = RgbReader(data_path)
    width = video_reader.width
    height = video_reader.height
    fps = video_reader.fps
    video_reader.close()
    return width, height, fps

def get_batch_prop(data_lst, data_root):
    width_lst = []
    height_lst = []
    fps_lst = []
    for i in range(len(data_lst)):
        data_name = data_lst[i]
        data_path = os.path.join(data_root, data_name)
        width, height, fps = get_rgb_resolution_fps(data_path)
        width_lst.append(width)
        height_lst.append(height)
        fps_lst.append(fps)
    return width_lst, height_lst, fps_lst
