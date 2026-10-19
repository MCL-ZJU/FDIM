"""Shared bounded SDR inference for the API, command line and benchmarks."""
import contextlib
from collections import deque
import queue
import threading
from types import SimpleNamespace
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from utils.reader import SampledReader
from utils.utils import map_score


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stream_batches(args, paths, device):
    if device.type == "cuda":
        torch.cuda.set_device(device)
    readers, buffers = [], [[], []]
    try:
        for path, fmt, width, height in zip(paths, args.formats, args.widths, args.heights):
            readers.append(SampledReader(path, width, height, fmt, args.interval, args.frame_limit))
        while True:
            frames = [reader.next() for reader in readers]
            if not all(ok for ok, _ in frames):
                break
            for (_, frame), buffer in zip(frames, buffers):
                buffer.append(frame)
            if len(buffers[0]) == args.batch_size:
                yield pack_batch(buffers, device, args.pipeline, args.uint8_transfer)
                buffers = [[], []]
        if buffers[0]:
            yield pack_batch(buffers, device, args.pipeline, args.uint8_transfer)
    finally:
        for reader in readers:
            reader.close()


def pack_batch(buffers, device, pinned, uint8_transfer=False):
    # Keep the original ToTensor CPU normalization and contiguous NCHW layout.
    convert = (lambda frame: torch.from_numpy(frame).permute(2, 0, 1)) if uint8_transfer else to_tensor
    batch = [torch.stack([convert(frame) for frame in frames]) for frames in buffers]
    return [value.pin_memory() for value in batch] if pinned and device.type == 'cuda' else batch


def prefetched(source, depth):
    if not depth:
        yield from source
        return
    items, stop = queue.Queue(maxsize=depth), threading.Event()
    def put(value):
        while not stop.is_set():
            try:
                items.put(value, timeout=.1)
                return
            except queue.Full:
                pass
    def produce():
        try:
            for batch in source:
                if stop.is_set():
                    break
                put((batch, None))
        except BaseException as exc:
            put((None, exc))
        finally:
            source.close()
            put((None, None))
    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            batch, error = items.get()
            if error:
                raise error
            if batch is None:
                break
            yield batch
    finally:
        stop.set()
        thread.join(timeout=5)

def _score_stream(model, device, args, paths):
    sync(device)
    outputs, pending = [], deque()
    if args.uint8_transfer and (not hasattr(model, '_fdim_value_lut') or model._fdim_value_lut.device != device):
        # Copy the exact CPU ToTensor values; do not substitute GPU division/rounding.
        values = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
        model._fdim_value_lut = to_tensor(values).reshape(-1).to(device)
    copy_stream = torch.cuda.Stream(device=device) if args.pipeline and device.type == 'cuda' else None
    with contextlib.closing(prefetched(stream_batches(args, paths, device), args.prefetch)) as batches:
        with torch.inference_mode():
            for cpu_batch in batches:
                # At most two in-flight batches: asynchronous work cannot grow with video length.
                if len(pending) == 2:
                    pending.popleft()[0].synchronize()
                if copy_stream is not None:
                    with torch.cuda.stream(copy_stream):
                        gpu_batch = [value.to(device, non_blocking=True) for value in cpu_batch]
                        copied = torch.cuda.Event()
                        copied.record(copy_stream)
                    current = torch.cuda.current_stream(device)
                    current.wait_event(copied)
                    for value in gpu_batch:
                        value.record_stream(current)
                else:
                    gpu_batch = [value.to(device) for value in cpu_batch]
                if getattr(args, 'uint8_transfer', False):
                    gpu_batch = [model._fdim_value_lut[value.long()] for value in gpu_batch]
                values = model(*gpu_batch).reshape(-1)
                if values.numel() != cpu_batch[0].shape[0]:
                    raise ValueError('Model returned an invalid score count')
                if copy_stream is not None:
                    done = torch.cuda.Event()
                    done.record()
                    pending.append((done, cpu_batch))
                    outputs.append(values.detach())
                else:
                    # Same per-batch synchronization as the original path when pipeline is off.
                    outputs.extend(values.detach().cpu().tolist())
                del gpu_batch
    if copy_stream is not None and outputs:
        outputs = torch.cat(outputs).cpu().tolist()
    sync(device)
    if not outputs:
        raise ValueError('No aligned frame pairs could be decoded')
    score = float(map_score(np.mean(outputs), 'reg'))
    return score, {'raw_scores': outputs, 'frame_indices': list(range(0, len(outputs) * args.interval, args.interval)), 'raw_mean': float(np.mean(outputs)), 'mapped_score': score, 'inference_mode': 'stream'}


def evaluate_stream(model, device, paths, formats, widths, heights, interval=1,
                    frame_limit=-1, frame_batch_size=1, prefetch=0, pipeline=False,
                    uint8_transfer=True):
    model.eval()
    device = torch.device(device)
    args = SimpleNamespace(formats=formats, widths=widths, heights=heights,
        interval=interval, frame_limit=frame_limit, batch_size=frame_batch_size,
        prefetch=prefetch, pipeline=pipeline, uint8_transfer=uint8_transfer)
    return _score_stream(model, device, args, paths)
