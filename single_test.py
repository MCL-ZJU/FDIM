import argparse
import sys
import os

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
    parser.add_argument('--ref_video_root', type=str, default='./data/test/SRC1001_1920x1080_25_yuv420p.mp4')
    parser.add_argument('--dis_video_root', type=str, default='./data/test/SRC1001_1920x1080_25_yuv420p.mp4.x265.r0.265.mp4')
    parser.add_argument('--model_path', type=str, default='./fdim/dist/checkpoints/dist_5.0.0.ckpt')
    parser.add_argument(
        '--ref_fmt',
        type=str,
        default="rgb",
        help="If the reference video is in non YUV format, this parameter is provided as 'rgb', else this parameter must be provided, such as yuv420p, yuv422p, etc.")
    parser.add_argument(
        '--dis_fmt',
        type=str,
        default="rgb",
        help="If the distorted video is in non YUV format, this parameter is provided as 'rgb', else this parameter must be provided, such as yuv420p, yuv422p, etc.")
    parser.add_argument(
        '--ref_bit_depth',
        type=int,
        default=8,
        help="bit depth for reference video")
    parser.add_argument(
        '--dis_bit_depth',
        type=int,
        default=8,
        help="bit depth for distorted video")
    parser.add_argument(
        '--ref_width',
        type=int,
        default=None,
        help='If the reference video is in YUV format, this parameter must be provided')
    parser.add_argument(
        '--ref_height',
        type=int,
        default=None,
        help='If the reference video is in YUV format, this parameter must be provided')
    parser.add_argument(
        '--dis_width',
        type=int,
        default=None,
        help='If the distorted video is in YUV format, this parameter must be provided')
    parser.add_argument(
        '--dis_height',
        type=int,
        default=None,
        help='If the distorted video is in YUV format, this parameter must be provided')
    parser.add_argument(
        '--frame_rate',
        type=int,
        default=None,
        help="If the video is in YUV format and only calculate a certain amount frames per second from video, this parameter must be provided")

    parser.add_argument(
        '--input_resolution',
        type=str,
        default="ori",
        help="if 'ori' is not assigned, image will be scaled to assigned value; otherwise image will be sent be network by originial resolution")
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
    args = parser.parse_args(sys.argv[3:])

    return run_fdim_single_infer.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, choices=['fdim'], default='fdim', help='choose a metric')
    args, unknown = parser.parse_known_args()

    if args.metric == 'fdim':
        sys.path.append(os.path.join(os.path.dirname(__file__), 'fdim'))
        import run_fdim_single_infer
        fdim()
    else:
        parser.print_help()
    
