import argparse
import time
import torch

from utils.utils import get_rgb_resolution_fps, load_checkpoint
from dist.deep_model import Model
from run_fdim import run_model

def main(args):
    check_arguments(args)

    ref_path = args.ref_video_root
    dis_path = args.dis_video_root
    model_weight_path = args.model_path

    ref_fmt = args.ref_fmt
    dis_fmt = args.dis_fmt

    ref_width = args.ref_width
    ref_height = args.ref_height
    dis_width = args.dis_width
    dis_height = args.dis_height
    ref_bit_depth = args.ref_bit_depth
    dis_bit_depth = args.dis_bit_depth
    frame_rate = args.frame_rate

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

    device = torch.device("cuda:{}".format(
        gpu_idx) if torch.cuda.is_available() else "cpu")
    model = Model()
    model = load_checkpoint(model, model_weight_path, device)

    if not (ref_width and ref_height) and ref_fmt == "rgb":
        ref_width, ref_height, _ = get_rgb_resolution_fps(ref_path)

    if not (dis_width and dis_height) and dis_fmt == "rgb":
        dis_width, dis_height, _ = get_rgb_resolution_fps(dis_path)


    if not frame_rate:
        if calc_interval < 0 and ref_fmt == "rgb":
            _, _, frame_rate = get_rgb_resolution_fps(ref_path)
        else:
            frame_rate = 0

    if calc_interval < 0:
        calc_interval_update = int(frame_rate / abs(calc_interval))
    else:
        calc_interval_update = calc_interval

    t10 = time.time()

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

    t11 = time.time()
    print('%s video score is : %s' % (dis_path, fdim_score))
    print('%s video used time is : %ss' % (dis_path, t11 - t10))


def check_arguments(args):
    if args.ref_fmt != "rgb":
        assert args.ref_width is not None and args.ref_height is not None
        if args.calc_interval < 0:
            assert args.frame_rate is not None
    if args.dis_fmt != "rgb":
        assert args.dis_width is not None and args.dis_height is not None


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
    parser.add_argument('--ref_video_root', type=str, default='./data/test/SRC1001_1920x1080_25_yuv420p.mp4')
    parser.add_argument('--dis_video_root', type=str,
                        default='./data/test/SRC1001_1920x1080_25_yuv420p.mp4.x265.r0.265.mp4')
    parser.add_argument('--model_path', type=str, default='./dist/checkpoints/dist_5.0.0.ckpt')
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
    args = parser.parse_args()

    main(args)
