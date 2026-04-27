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
