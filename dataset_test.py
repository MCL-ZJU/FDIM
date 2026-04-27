import argparse
import sys
import os
from common_utils import corr

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def fdim():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_path',
        type=str,
        default='./data/dataset/datatest_info.csv')
    parser.add_argument('--ref_dir', type=str, default='./data/test/')
    parser.add_argument('--dis_dir', type=str, default='./data/test/')
    parser.add_argument(
        "--model_path",
        type=str,
        default="./fdim/dist/checkpoints/dist_5.0.0.ckpt")
    parser.add_argument('--save_dir', type=str, default='data/result')
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
        default="ori",
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
    args = parser.parse_args(sys.argv[3:])

    return run_fdim_dataset_infer.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, choices=['fdim'], default='fdim', help='chose a metric')
    parser.add_argument('--save_dir', type=str, default='data/result')
    parser.add_argument('--save_name', type=str, default='fdim_results')
    parser.add_argument('--correlation', default=False, action='store_true', help='Calculate the correlation between predicted scores and labels')
    args, unknown = parser.parse_known_args()

    if args.metric == 'fdim':
        sys.path.append(os.path.join(os.path.dirname(__file__), 'fdim'))
        import run_fdim_dataset_infer
        results_info = fdim()
    else:
        parser.print_help()

    if (results_info['mos'] == 0).all() == False and args.correlation:
        corr.calc_correlation(results_info, args.save_dir, args.save_name, args.metric)
    
    print('Finished!')
