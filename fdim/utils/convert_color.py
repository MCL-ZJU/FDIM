import os
import subprocess
import cv2
import numpy as np


def get_ffmpeg_pix_fmt(bit_depth, video_fmt):
    pix_fmt_map = {
        'rgb': {
            8: 'rgb24',
            10: 'rgb48le',
            12: 'rgb48le',
        },
        'yuv420p': {
            8: 'yuv420p',
            10: 'yuv420p10le',
            12: 'yuv420p12le',
        },
        'yuv422p': {
            8: 'yuv422p',
            10: 'yuv422p10le',
            12: 'yuv422p12le',
        },
        'yuv444p': {
            8: 'yuv444p',
            10: 'yuv444p10le',
            12: 'yuv444p12le',
        },
    }

    try:
        return pix_fmt_map[video_fmt][bit_depth]
    except KeyError:
        raise ValueError(f"Unsupported combination of bit depth {bit_depth} and video format {video_fmt}")


def get_vmaf_pix_fmt(yuv_fmt):
    pix_fmt_420 = ["nv12", "nv21", "i420", "p010le", "yu12", "yv12"]
    pix_fmt_422 = ["yuyv422", "uyvy422"]
    pix_fmt_444 = ["i444"]

    if "444" in yuv_fmt:
        fmt = "444"
    elif "422" in yuv_fmt:
        fmt = "422"
    elif "420" in yuv_fmt:
        fmt = "420"
    elif yuv_fmt.lower() in pix_fmt_420:
        fmt = "420"
    elif yuv_fmt.lower() in pix_fmt_422:
        fmt = "422"
    elif yuv_fmt.lower() in pix_fmt_444:
        fmt = "444"
    else:
        raise ValueError(f"Unsupported yuv format {yuv_fmt}")

    return fmt

def rgb2yuv_resize(
        filepath,
        filepath_new,
        output_width=0,
        output_height=0,
        output_fmt="yuv444p",
        resize_flag=False,
        resize_method="default",
        threads=1,
        temp_root='./video_temp/'):

    os.makedirs(temp_root, exist_ok=True)
    if not resize_flag:
        ff = 'ffmpeg -y -i {} -c:v rawvideo -pix_fmt {} -an -vsync 0 {} -threads {} -loglevel error'.format(
            filepath, output_fmt, filepath_new, threads)
    else:
        if resize_method == "default":
            ff = 'ffmpeg -y -i {} -c:v rawvideo -s {}x{} -pix_fmt {} -an -vsync 0 {} -threads {} -loglevel error'.format(
                filepath, output_width, output_height, output_fmt, filepath_new, threads)
        elif resize_method == "lanczos":
            # ffmpeg.exe -s hd1080 -r 30 -i docoded.yuv -vcodec rawvideo
            # -sws_flags lanczos -s 3840x2160 -r 30 upscaled.yuv
            ff = 'ffmpeg -y -i {} -c:v rawvideo -sws_flags lanczos -s {}x{} -pix_fmt {} -an -vsync 0 {} -threads {} -loglevel error'.format(
                filepath, output_width, output_height, output_fmt, filepath_new, threads)
        else:
            raise ValueError

    print('ff: ', ff)
    proc = subprocess.run(ff, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit_code={proc.returncode}).\n"
            f"cmd={ff}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}"
        )
    return 0


def yuv2yuv_resize(
        filepath,
        filepath_new,
        input_width,
        input_height,
        input_format,
        output_width,
        output_height,
        output_fmt="yuv444p",
        resize_flag=False,
        resize_method="default",
        threads=1,
        temp_root='./video_temp/'):
    os.makedirs(temp_root, exist_ok=True)
    # ffmpeg -s [input_width]x[input_height] -pix_fmt [input_pixel_format] -i
    # [input_file] -s [output_width]x[output_height] -pix_fmt
    # [output_pixel_format] [output_file]
    if not resize_flag:
        ff = 'ffmpeg -s {}x{} -pix_fmt {} -i {} -s {}x{} -pix_fmt {} {} -threads {} -loglevel error'.format(
            input_width,
            input_height,
            input_format,
            filepath,
            output_width,
            output_height,
            output_fmt,
            filepath_new,
            threads)
    else:
        if resize_method == "default":
            ff = 'ffmpeg -s {}x{} -pix_fmt {} -i {} -s {}x{} -pix_fmt {} {} -threads {} -loglevel error'.format(
                input_width,
                input_height,
                input_format,
                filepath,
                output_width,
                output_height,
                output_fmt,
                filepath_new,
                threads)
        elif resize_method == "lanczos":
            # ffmpeg.exe -s hd1080 -r 30 -i docoded.yuv -vcodec rawvideo -sws_flags
            # lanczos -s 3840x2160 -r 30 upscaled.yuv
            ff = 'ffmpeg -s {}x{} -pix_fmt {} -i {} -sws_flags lanczos -s {}x{} -pix_fmt {} {} -threads {} -loglevel error'.format(
                input_width,
                input_height,
                input_format,
                filepath,
                output_width,
                output_height,
                output_fmt,
                filepath_new,
                threads)
        else:
            print("resize_method: ", resize_method)
            raise ValueError(f"not support {resize_method}")

    print("ff: ", ff)
    proc = subprocess.run(ff, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit_code={proc.returncode}).\n"
            f"cmd={ff}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}"
        )
    return 0


def conversion_command(filepath, output, input_fmt, width, height,
                       output_fmt, output_width, output_height, resize_flag=False,
                       resize_method="default", threads=1, frame_limit=-1):
    """The existing raw conversion/resizing protocol, expressed as argv."""
    command = ["ffmpeg", "-y", "-nostdin"]
    if input_fmt != "rgb":
        command += ["-s", f"{width}x{height}", "-pix_fmt", input_fmt]
    command += ["-i", str(filepath)]
    if input_fmt == "rgb":
        command += ["-c:v", "rawvideo"]
    if resize_flag and resize_method == "lanczos":
        command += ["-sws_flags", "lanczos"]
    elif resize_flag and resize_method != "default":
        raise ValueError(f"Unsupported resize method: {resize_method}")
    if resize_flag or input_fmt != "rgb":
        command += ["-s", f"{output_width}x{output_height}"]
    command += ["-pix_fmt", output_fmt]
    if input_fmt == "rgb":
        command += ["-an", "-vsync", "0"]
    if frame_limit > 0:
        command += ["-frames:v", str(frame_limit)]
    # Explicit rawvideo is needed for FIFO paths; it is the same .yuv muxer.
    command += ["-f", "rawvideo", str(output), "-threads", str(threads), "-loglevel", "error"]
    return command


def _stop_process(process):
    if process.poll() is None:
        process.kill()
    process.wait()


from contextlib import contextmanager


@contextmanager
def vmaf_inputs(ref_path, dis_path, ref_fmt, dis_fmt, ref_width, ref_height,
                dis_width, dis_height, ref_bit_depth, dis_bit_depth,
                resize_flag=False, resize_method="default", n_threads=1,
                temp_root="./video_temp", streaming=True, frame_limit=-1):
    """Yield VMAF inputs, using bounded OS pipes for conversions on POSIX."""
    import tempfile
    from pathlib import Path
    maximum = max(ref_bit_depth, dis_bit_depth)
    target_fmt = ref_fmt if ref_fmt != "rgb" and ref_bit_depth == maximum else get_ffmpeg_pix_fmt(maximum, "yuv420p")
    specs = [
        (ref_path, ref_fmt, ref_width, ref_height, ref_bit_depth, False),
        (dis_path, dis_fmt, dis_width, dis_height, dis_bit_depth, resize_flag),
    ]
    Path(temp_root).mkdir(parents=True, exist_ok=True)
    processes, logs, paths = [], [], []
    with tempfile.TemporaryDirectory(prefix="fdim-vmaf-", dir=temp_root) as directory:
        try:
            for index, (source, fmt, width, height, bits, resize) in enumerate(specs):
                no_truncation = frame_limit < 0
                if frame_limit > 0 and fmt != "rgb":
                    ratio = {"420": 1.5, "422": 2, "444": 3}[get_vmaf_pix_fmt(fmt)]
                    limit_bytes = int(width * height * ratio * (1 if bits <= 8 else 2) * frame_limit)
                    no_truncation = os.path.getsize(source) <= limit_bytes
                reuse = fmt != "rgb" and fmt == target_fmt and bits == maximum and not resize and no_truncation
                if reuse:
                    paths.append(str(source))
                    continue
                target = str(Path(directory) / f"{index}.yuv")
                command = conversion_command(source, target, fmt, width, height, target_fmt,
                    ref_width, ref_height, resize, resize_method, n_threads, frame_limit)
                if streaming and hasattr(os, "mkfifo"):
                    os.mkfifo(target)
                    log = tempfile.TemporaryFile()
                    logs.append(log)
                    processes.append(subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                                     stderr=log, stdin=subprocess.DEVNULL))
                else:
                    subprocess.run(command, check=True, capture_output=True)
                paths.append(target)
            yield paths, get_vmaf_pix_fmt(target_fmt), maximum, processes
            for process, log in zip(processes, logs):
                if process.wait(timeout=30):
                    log.seek(0)
                    raise RuntimeError("VMAF input conversion failed: " + log.read().decode(errors="replace")[-2000:])
        finally:
            for process in processes:
                _stop_process(process)
            for log in logs:
                log.close()
