import cv2
import numpy as np
import subprocess


class RgbReader(object):
    DATA_FORMAT = "rgb"

    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def next(self, format="uint"):
        ret, bgr = self.cap.read()
        if not ret:
            raise StopIteration
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if format == "uint":
            return rgb
        else:  # "float"
            return rgb.astype(np.double) / (2.0 ** 8 - 1.0)

    def close(self):
        self.cap.release()


class Reader(object):

    def __init__(self, filepath, width, height, data_type="rgb"):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.data_type = data_type

        if "rgb" not in self.data_type.lower():
            cmd = [
                'ffmpeg',
                '-s', f'{width}x{height}',
                '-pix_fmt', data_type,
                '-i', filepath,
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo',
                '-nostdin',
                '-loglevel', 'quiet',
                '-'
            ]
        else:
            cmd = [
                'ffmpeg',
                '-i', filepath,
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo',
                '-nostdin',
                '-loglevel', 'quiet',
                '-'
            ]
        self.frame_size = width * height * 3
        self.pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)

    def next(self):
        in_bytes = self.pipe.stdout.read(self.frame_size)
        if len(in_bytes) < self.frame_size:
            return False, None
        frame = np.frombuffer(in_bytes, np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def close(self):
        self.pipe.stdout.close()
        self.pipe.wait()


class SampledReader:
    """Bounded sampled RGB reader preserving the original Reader's color semantics.

    Raw 8-bit YUV420 seeks to selected frames. Encoded input stays in the original
    direct-to-RGB FFmpeg path, retaining range/matrix metadata and timestamp resampling.
    """
    def __init__(self, filepath, width, height, data_type="rgb", interval=1,
                 frame_limit=-1, timeout=120):
        from pathlib import Path
        import tempfile
        import threading
        self.width, self.height = int(width), int(height)
        self.interval, self.frame_limit = int(interval), int(frame_limit)
        if self.width <= 0 or self.height <= 0 or self.interval < 1:
            raise ValueError("Positive dimensions and sampling interval are required")
        if data_type not in ("rgb", "yuv420p"):
            raise ValueError("SampledReader supports encoded RGB or raw 8-bit YUV420")
        self.timeout, self.count, self.error = timeout, 0, None
        self.encoded, self.source_index = data_type == "rgb", 0
        self.frame_size = self.width * self.height * 3
        self.stop = threading.Event()
        self.logs = tempfile.TemporaryFile()
        self.feeder = None
        self.path = Path(filepath)
        if data_type == "yuv420p":
            if self.width % 2 or self.height % 2:
                raise ValueError("YUV420 dimensions must be even")
            self.raw_size = self.width * self.height * 3 // 2
            size = self.path.stat().st_size
            if size % self.raw_size:
                raise ValueError("Raw video ends with an incomplete frame")
            self.frames = size // self.raw_size
            if self.frame_limit > 0:
                self.frames = min(self.frames, self.frame_limit)
            command = ["ffmpeg", "-threads", "1", "-f", "rawvideo", "-s",
                       f"{self.width}x{self.height}", "-pix_fmt", "yuv420p", "-i", "pipe:0"]
        else:
            command = ["ffmpeg", "-i", str(self.path)]
            if self.frame_limit > 0:
                command += ["-frames:v", str(self.frame_limit)]
        command += ["-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo",
                    "-threads", "1", "-nostdin", "-loglevel", "error", "pipe:1"]
        self.pipe = subprocess.Popen(command, stdin=subprocess.PIPE if data_type == "yuv420p" else subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=self.logs, bufsize=0)
        if data_type == "yuv420p":
            self.feeder = threading.Thread(target=self._feed, daemon=True)
            self.feeder.start()

    def _feed(self):
        try:
            with self.path.open("rb", buffering=0) as source:
                for index in range(0, self.frames, self.interval):
                    if self.stop.is_set():
                        break
                    source.seek(index * self.raw_size)
                    frame = source.read(self.raw_size)
                    if len(frame) != self.raw_size:
                        raise ValueError("Raw input changed or ended during inference")
                    view = memoryview(frame)
                    while view and not self.stop.is_set():
                        written = self.pipe.stdin.write(view)
                        if not written:
                            raise RuntimeError("RGB conversion pipe closed")
                        view = view[written:]
        except BaseException as exc:
            if not self.stop.is_set():
                self.error = exc
        finally:
            self.pipe.stdin.close()

    def _next_rgb(self):
        data = bytearray(self.frame_size)
        offset = 0
        while offset < self.frame_size:
            size = self.pipe.stdout.readinto(memoryview(data)[offset:])
            if not size:
                if self.feeder:
                    self.feeder.join(timeout=self.timeout)
                    if self.feeder.is_alive():
                        raise TimeoutError("Raw-video feeder did not finish")
                if self.error:
                    raise self.error
                code = self.pipe.wait(timeout=self.timeout)
                if code or offset:
                    self.logs.seek(0)
                    raise RuntimeError("Video decoding failed or returned a partial frame: " +
                                       self.logs.read().decode(errors="replace")[-2000:])
                return False, None
            offset += size
        self.count += 1
        return True, np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)

    def next(self):
        if not self.encoded:
            return self._next_rgb()
        # The original RGB muxer can duplicate/drop frames by timestamp (VFR).
        # Filter its emitted sequence, not decoder indices, to preserve semantics.
        while True:
            ok, frame = self._next_rgb()
            if not ok:
                return False, None
            index = self.source_index
            self.source_index += 1
            if index % self.interval == 0:
                return True, frame

    def close(self):
        self.stop.set()
        if self.pipe.poll() is None:
            self.pipe.kill()
        if self.feeder:
            self.feeder.join(timeout=5)
        self.pipe.wait()
        self.pipe.stdout.close()
        self.logs.close()
