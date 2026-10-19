"""Measure the official FDIM API; this script implements no decoding or scoring."""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"fdim"))
from dist.deep_model import Model
from dist.test_video_gen import evaluate_video_gen
from run_fdim import run_model


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class RSS:
    """10ms process-tree RSS sampling; shared pages can be counted more than once."""
    def sample(self):
        process = psutil.Process()
        total = 0
        for child in [process] + process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        self.peak = max(self.peak, total)

    def loop(self):
        while not self.stop.wait(.01):
            self.sample()

    def __enter__(self):
        self.peak, self.stop = 0, threading.Event()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)



def measure(args, model, device):
    options = dict(ref_fmt=args.ref_fmt, dis_fmt=args.dis_fmt,
        ref_width=args.width, ref_height=args.height, dis_width=args.width, dis_height=args.height,
        ref_bit_depth=8, dis_bit_depth=8, calc_interval=args.interval,
        inference_mode="legacy" if args.mode in ("legacy", "stock") else "stream",
        frame_batch_size=args.batch_size, prefetch=args.prefetch, pipeline=args.pipeline,
        uint8_transfer=args.uint8_transfer, frame_limit=args.frames, return_frame_scores=True)
    sync(device)
    start = time.perf_counter()
    if args.scope == "full":
        fdim, vmaf, deep, details = run_model(args.ref, args.dis, model, device,
            n_threads=args.threads, video_temp_path=args.scratch,
            parallel_branches=args.parallel_branches, **options)
    else:
        deep, details = evaluate_video_gen(args.ref, args.dis, model, device, **options)
    sync(device)
    row = {"total_s": time.perf_counter() - start, "deep_score": float(deep),
           "raw_scores": details["raw_scores"], "frame_indices": details["frame_indices"]}
    if args.scope == "full":
        row.update(fdim_score=fdim, vmaf_score=vmaf)
    if row["frame_indices"] != args.indices or not all(math.isfinite(v) for v in row["raw_scores"]):
        raise ValueError("Unexpected frame indices/count or non-finite scores")
    return row


def worker(args, device_name, output, barrier, gate):
    if hasattr(os, "setsid"):
        os.setsid()
    try:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            if not args.allow_other_gpu and "A100" not in torch.cuda.get_device_name(device):
                raise ValueError("A100 required")
        model = Model().eval()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("state_dict", state), strict=True)
        model.to(device)
        for _ in range(args.warmups):
            measure(args, model, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        barrier.wait(timeout=args.timeout)
        if not gate.wait(timeout=args.timeout):
            raise TimeoutError("Start gate timed out")
        with RSS() as rss:
            rows = [measure(args, model, device) for _ in range(args.repeats)]
        result = {"device": device_name, "rows": rows, "host_peak_tree_rss_bytes": rss.peak}
        if device.type == "cuda":
            result.update(gpu_name=torch.cuda.get_device_name(device),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        output.write_text(json.dumps(result, allow_nan=False))
    except BaseException:
        output.write_text(json.dumps({"error": traceback.format_exc()}))
        barrier.abort()
        raise


def group(args, directory, baseline=False):
    config = argparse.Namespace(**vars(args))
    devices = args.devices[:1] if baseline else args.devices
    if baseline:
        config.mode, config.batch_size, config.prefetch, config.pipeline = "legacy", 1, 0, False
        config.parallel_branches = False
    ctx = mp.get_context("spawn")
    barrier, gate = ctx.Barrier(len(devices) + 1), ctx.Event()
    processes, files = [], []
    try:
        for index, device in enumerate(devices):
            path = directory / f'{"baseline" if baseline else "run"}.{index}.json'
            config.scratch = str(directory / f'vmaf-{baseline}-{index}')
            process = ctx.Process(target=worker, args=(config, device, path, barrier, gate))
            process.start()
            processes.append(process)
            files.append(path)
        barrier.wait(timeout=args.timeout)
        begin = time.perf_counter()
        gate.set()
        deadline = begin + args.timeout
        with RSS() as rss:
            for process in processes:
                process.join(timeout=max(0, deadline - time.perf_counter()))
        elapsed = time.perf_counter() - begin
        if any(p.is_alive() or p.exitcode != 0 for p in processes):
            raise RuntimeError("A worker failed or timed out")
        return {"elapsed_s": elapsed, "aggregate_sampled_pairs_s": len(devices)*len(args.indices)*args.repeats/elapsed,
                "host_peak_tree_rss_bytes": rss.peak,
                "workers": [json.loads(path.read_text()) for path in files]}
    except BaseException:
        for path in files:
            if path.exists():
                value = json.loads(path.read_text())
                if "error" in value:
                    print(value["error"], file=sys.stderr)
        raise
    finally:
        for process in processes:
            if process.is_alive():
                if hasattr(os, "getpgid") and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            process.join()
def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be >= 1')
    return number



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--dis", required=True)
    parser.add_argument("--width", type=positive, required=True)
    parser.add_argument("--height", type=positive, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--frames", type=positive, default=150)
    parser.add_argument("--ref-fmt", default=None)
    parser.add_argument("--dis-fmt", default=None)
    parser.add_argument("--sampling", choices=["full", "1fps"], default="1fps")
    parser.add_argument("--mode", choices=["stream", "legacy", "stock"], default="stream")
    parser.add_argument("--scope", choices=["deep", "full"], default="deep")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--batch-size", type=positive, default=1)
    parser.add_argument("--prefetch", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--uint8-transfer", dest="uint8_transfer", action="store_true", default=True)
    parser.add_argument("--no-uint8-transfer", dest="uint8_transfer", action="store_false")
    parser.add_argument("--parallel-branches", action="store_true")
    parser.add_argument("--threads", type=positive, default=8)
    parser.add_argument("--warmups", type=positive, default=3)
    parser.add_argument("--repeats", type=positive, default=10)
    parser.add_argument("--timeout", type=positive, default=900)
    parser.add_argument("--checkpoint", default=str(ROOT/"fdim/dist/checkpoints/dist_5.0.0.ckpt"))
    parser.add_argument("--scratch")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--score-atol", type=float, default=0.)
    parser.add_argument("--allow-other-gpu", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps < 1 or not math.isfinite(args.score_atol) or args.score_atol < 0:
        parser.error("Invalid fps or score tolerance")
    if args.json and args.json.exists():
        parser.error("Use a fresh --json path")
    args.ref_fmt = args.ref_fmt or ("yuv420p" if Path(args.ref).suffix.lower()==".yuv" else "rgb")
    args.dis_fmt = args.dis_fmt or ("yuv420p" if Path(args.dis).suffix.lower()==".yuv" else "rgb")
    args.interval = max(1, round(args.fps)) if args.sampling=="1fps" else 1
    args.indices = list(range(0,args.frames,args.interval))
    result = {"success": False, "config": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    try:
        with tempfile.TemporaryDirectory(prefix="fdim-benchmark-",dir=args.scratch) as tmp:
            baseline = group(args,Path(tmp),baseline=True)
            measured = group(args,Path(tmp))
            reference = baseline["workers"][0]["rows"][0]
            result.update(baseline=baseline,measured=measured)
            for value in (baseline,measured):
                for item in value["workers"]:
                    for row in item["rows"]:
                        delta = max(abs(x-y) for x,y in zip(reference["raw_scores"],row["raw_scores"]))
                        if len(row["raw_scores"])!=len(reference["raw_scores"]) or delta>args.score_atol:
                            raise ValueError(f"Raw scores differ from official legacy inference: {delta}")
                        for key in ("deep_score","fdim_score","vmaf_score"):
                            if key in row and abs(row[key]-reference[key])>args.score_atol:
                                raise ValueError(f"{key} differs from official legacy inference")
                        row["max_raw_delta_vs_legacy"] = delta
            for index,item in enumerate(measured["workers"]):
                seconds=[r["total_s"] for r in item["rows"]]
                gpu = (f"{item['gpu_peak_allocated_bytes']/2**30:.3f}/{item['gpu_peak_reserved_bytes']/2**30:.3f} GiB"
                       if "gpu_peak_allocated_bytes" in item else "N/A")
                print(f"worker {index}: {statistics.mean(seconds):.6f} +/- {statistics.pstdev(seconds):.6f} s/video; "
                      f"RSS {item['host_peak_tree_rss_bytes']/2**30:.3f} GiB; GPU allocated/reserved {gpu}")
            print(f"Group throughput: {measured['aggregate_sampled_pairs_s']:.3f} selected pairs/s; parity PASS")
            result["environment"] = {"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
                "os":platform.platform(),"checkpoint_sha256":sha(args.checkpoint),
                "cudnn_tf32":torch.backends.cudnn.allow_tf32,"matmul_tf32":torch.backends.cuda.matmul.allow_tf32,
                "sources":{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"fdim").rglob("*.py")}}
            result["success"] = True
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if args.json:
            args.json.parent.mkdir(parents=True,exist_ok=True)
            with args.json.open("x") as stream:
                json.dump(result,stream,indent=2,allow_nan=False)
                stream.write("\n")


if __name__=="__main__":
    main()
