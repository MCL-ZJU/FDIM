import os
import argparse
import time
import torch
from tqdm import tqdm
from utils.utils import readinfo, load_checkpoint
from dist.deep_model import Model
from run_fdim import run_model

def main(args):
    check_arguments(args)

    csv_path = args.csv_path
    ref_root = args.ref_dir
    dis_root = args.dis_dir
    model_weight_path = args.model_path
    output_folder = args.save_dir
    save_name = args.save_name

    dis_column = args.dis_column
    ref_column = args.ref_column
    mos_column = args.mos_column
    ref_fmt = args.ref_fmt
    dis_fmt = args.dis_fmt
    ref_bit_depth_column = args.ref_bit_depth_column
    dis_bit_depth_column = args.dis_bit_depth_column

    ref_width_column = args.ref_width_column
    ref_height_column = args.ref_height_column
    dis_width_column = args.dis_width_column
    dis_height_column = args.dis_height_column
    frame_rate_col = args.frame_rate_col

    input_resolution = args.input_resolution
    calc_interval = args.calc_interval
    resize = args.resize
    resize_method = args.resize_method

    n_threads = args.n_threads
    vmaf_version = args.vmaf_version
    video_temp_path = args.video_temp_path
    gpu_idx = args.gpu_idx
    preprocess = getattr(args, "preprocess", "none")
    display_model = getattr(args, "display_model", None)

    os.makedirs(output_folder, exist_ok=True)

    device = torch.device("cuda:{}".format(
        gpu_idx) if torch.cuda.is_available() else "cpu")

    model = Model()

    model = load_checkpoint(model, model_weight_path, device)


    data_info, dis_names, ref_names, mos_lst, ref_width_lst, ref_height_lst, dis_width_lst, dis_height_lst, frame_rate_lst, ref_bit_depth_lst, dis_bit_depth_lst, ref_fmt_lst, dis_fmt_lst = readinfo(
        csv_path, dis_column=dis_column, ref_column=ref_column, mos_column=mos_column, ref_root=ref_root, dis_root=dis_root, ref_width_column=ref_width_column, ref_height_column=ref_height_column, dis_width_column=dis_width_column, dis_height_column=dis_height_column, ref_fmt=ref_fmt, dis_fmt=dis_fmt, frame_rate_col=frame_rate_col,
        ref_bit_depth_column=ref_bit_depth_column,
        dis_bit_depth_column=dis_bit_depth_column,
        calc_interval=calc_interval)

    fdim_scores = []
    deep_scores = []
    vmaf_scores = []

    for i in tqdm(range(len(ref_names))):
        t10 = time.time()
        ref_name = ref_names[i]
        dis_name = dis_names[i]
        ref_path = os.path.join(ref_root, ref_name)
        dis_path = os.path.join(dis_root, dis_name)
        ref_width = ref_width_lst[i]
        ref_height = ref_height_lst[i]
        dis_width = dis_width_lst[i]
        dis_height = dis_height_lst[i]

        frame_rate = frame_rate_lst[i]
        ref_bit_depth = ref_bit_depth_lst[i]
        dis_bit_depth = dis_bit_depth_lst[i]

        ref_fmt = ref_fmt_lst[i].lower()
        dis_fmt = dis_fmt_lst[i].lower()

        if calc_interval < 0:
            calc_interval_update = int(frame_rate / abs(calc_interval))
        else:
            calc_interval_update = calc_interval


        fdim_score, vmaf_score, deep_score = run_model(
            ref_path,
            dis_path,
            model=model,
            device=device,
            ref_fmt=ref_fmt,
            dis_fmt=dis_fmt,
            ref_width=ref_width,
            ref_height=ref_height,
            dis_width=dis_width,
            dis_height=dis_height,
            ref_bit_depth=ref_bit_depth,
            dis_bit_depth=dis_bit_depth,
            resize_flag=resize,
            resize_method=resize_method,
            input_resolution=input_resolution,
            calc_interval=calc_interval_update,
            n_threads=n_threads,
            vmaf_version=vmaf_version,
            video_temp_path=video_temp_path,
            preprocess=preprocess,
            display_model=display_model)
        fdim_scores.append(fdim_score)
        vmaf_scores.append(vmaf_score)
        deep_scores.append(deep_score)

        t11 = time.time()
        print('%s fdim score is : %s' % (dis_name, fdim_score))
        print('%s video used time is : %ss' % (dis_name, t11 - t10))

    data_info['fdim_score'] = fdim_scores
    data_info['vmaf_score'] = vmaf_scores
    data_info['deep_score'] = deep_scores

    csv_name = os.path.split(args.csv_path)[-1]
    output_csv = os.path.join(
        output_folder,
        '%s_%s_%s_calc_intervel_%s.csv' %
        (csv_name, save_name, args.input_resolution, calc_interval))
    print("output_csv: ", output_csv)
    data_info.to_csv(output_csv, mode='w', header=True, index=False)

    return data_info

def check_arguments(args):
    if args.ref_fmt != "rgb":
        assert args.ref_width_column is not None and args.ref_height_column is not None
        if args.calc_interval < 0:
            assert args.frame_rate_col is not None
    if args.dis_fmt != "rgb":
        assert args.dis_width_column is not None and args.dis_height_column is not None


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_path',
        type=str,
        default='./data/dataset0/datatest_info.csv')
    parser.add_argument('--ref_dir', type=str, default='./data/test/')
    parser.add_argument('--dis_dir', type=str, default='./data/test/')
    parser.add_argument(
        "--model_path",
        type=str,
        default="./dist/checkpoints/dist_5.0.0.ckpt")
    parser.add_argument('--save_dir', type=str, default='./data/result')
    parser.add_argument('--save_name', type=str, default='fdim_results')
    parser.add_argument('--dis_column', type=str, default='dis_name')
    parser.add_argument('--ref_column', type=str, default='ref_name')
    parser.add_argument('--mos_column', type=str, default='mos')
    parser.add_argument(
        '--ref_fmt',
        type=str,
        default="rgb",
        help="If the reference video is in non YUV format, this parameter is provided as 'rgb', else this parameter must be provided by column name in csv file")
    parser.add_argument(
        '--dis_fmt',
        type=str,
        default="rgb",
        help="If the distorted video is in non YUV format, this parameter is provided as 'rgb', else this parameter must be provided by column name in csv file")
    parser.add_argument(
        '--ref_bit_depth_column',
        type=str,
        default="ref_bit_depth",
        help="bit depth column for reference video")
    parser.add_argument(
        '--dis_bit_depth_column',
        type=str,
        default="dis_bit_depth",
        help="bit depth column for distorted video")
    parser.add_argument(
        '--ref_width_column',
        type=str,
        default=None,
        help='If the reference video is in YUV format, this parameter must be provided')
    parser.add_argument(
        '--ref_height_column',
        type=str,
        default=None,
        help='If the reference video is in YUV format, this parameter must be provided')
    parser.add_argument(
        '--dis_width_column',
        type=str,
        default=None,
        help='If the distorted video is in YUV format, this parameter must be provided')
    parser.add_argument(
        '--dis_height_column',
        type=str,
        default=None,
        help='If the distorted video is in YUV format, this parameter must be provided')
    parser.add_argument(
        '--frame_rate_col',
        type=str,
        default=None,
        help="If the video is in YUV format and only calculate a certain amount frames per second from video, this parameter must be provided")
    parser.add_argument(
        '--input_resolution',
        type=str,
        default="512",
        help="if 'ori' is not assigned ,image will be scaled to assigned value; otherwise image will be sent be network by originial resolution")
    parser.add_argument(
        '--calc_interval',
        type=int,
        default=1,
        help="if parameter is positive, the video will be calculated by every 'calc_ineterval' frames, else the video will be calculated by the abs('calc_ineterval') frames per second")
    parser.add_argument(
        '--preprocess',
        type=str,
        default="none",
        choices=["none", "pu21"])
    parser.add_argument(
        '--display_model',
        type=str,
        default=None)
    parser.add_argument(
        '--resize',
        type=str2bool,
        default=False,
        help="If resolution of the distorted video is not same as the reference video, this parameter must be provided as 'True'")
    parser.add_argument(
        '--resize_method',
        type=str,
        default="default",
        help="If parameter of 'resize' is True, this parameter must be provided, default parameter is 'default'")
    parser.add_argument('--n_threads', type=str, default=32, help="thread count for VMAF and ffmepg")
    parser.add_argument('--vmaf_version', type=str, default=None, help="vmaf version")
    parser.add_argument('--video_temp_path', type=str, default='./data/video_temp/', help="storage path for temporal file")
    parser.add_argument('--gpu_idx', type=int, default=0, help="gpu index for calculation of deep metric, use cpu if gpu_idx=-1")
    parser.add_argument('--correlation', default=False, action='store_true', help='Calculate the correlation between predicted scores and labels')

    args = parser.parse_args()

    main(args)
