import numpy as np
from PIL import Image
from torchvision import transforms

class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size
        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)

def resize_img(cv_img, resize_width, resize_height, resize_method):
    pil_img = Image.fromarray(cv_img)
    if resize_method == "default":
        resize_img = pil_img.resize((resize_width, resize_height),resample=Image.BICUBIC)
    elif resize_method == "lanczos":
        resize_img = pil_img.resize((resize_width, resize_height),resample=Image.LANCZOS)
    else:
        raise ValueError
    resize_img_arr = np.array(resize_img)
    return resize_img_arr