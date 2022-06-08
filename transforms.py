"""
普通的数据增强方法，可继续写新的方法，
"""
import numbers
import random
from typing import Optional, List, Tuple, Sequence

import torch
from torch import nn, Tensor
from torchvision.transforms import functional as F


class Compose(object):
    """
    组合多个transform函数
    """

    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """
    将PIL图像转为Tensor
    """

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """
    随机水平翻转图像以及bboxes
    """

    def __init__(self, prob=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = F.hflip(image)  # 水平翻转图片
            box = target['box']
            # bbox: xmin, ymin, xmax, ymax
            # box[0], box[2] = width - box[2], width - box[0]  # 垂直翻转对应bbox坐标信息
            box[:, [0, 2]] = width - box[:, [2, 0]]
            target['box'] = box
        return image, target


class RandomVerticalFlip(object):
    """
    随机垂直翻转图像以及boxes
    """

    def __init__(self, prob=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = F.vflip(image)
            box = target['box']
            # bbox: xmin, ymin, xmax, ymax
            # box[1], box[3] = height - box[3], height - box[1]  # 翻转对应bbox坐标信息
            box[:, [1, 3]] = width - box[:, [3, 1]]
            target['box'] = box
        return image, target


# 将此功能移入GeneralizedTransform类中
# class ResizeImageSize(object):
#     """
#     将图像调整大小，主要为其gt框大小的调整
#     """
#
#     def __init__(self, new_size):
#         super(ResizeImageSize, self).__init__()
#         self.new_size = new_size
#
#     def __call__(self, image, target):
#         height, width = image.shape[-2:]
#         scale_factor = float(self.new_size / height)
#         # interpolate利用插值的方法缩放图片
#         # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
#         # bilinear只支持4D Tensor
#         image = torch.nn.functional.interpolate(
#             image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True,
#             align_corners=False)[0]
#
#         ratio = torch.tensor(self.new_size, dtype=torch.float32, device=target['box'].device) / torch.tensor(height, dtype=torch.float32, device=target['box'].device)
#
#         box = target['box']
#         # 调整gt四个坐标
#         xmin, ymin, xmax, ymax = box.unbind(0)
#         xmin = torch.trunc(xmin * ratio)
#         xmax = torch.trunc(xmax * ratio)
#         ymin = torch.trunc(ymin * ratio)
#         ymax = torch.trunc(ymin * ratio)
#
#         box = torch.stack((xmin, ymin, xmax, ymax), dim=0)
#
#         target['box'] = box
#
#         return image, target


# class Normalize(object):
#     """
#     标准化图像
#     """
#
#     def __init__(self, mean, std, inplace=False):
#         super(Normalize, self).__init__()
#         self.mean = mean
#         self.std = std
#         self.inplace = inplace
#
#     def __call__(self, image, target):
#         image = F.normalize(image, self.mean, self.std, self.inplace)
#         return image, target


class Grayscale(object):
    """
    将图片灰度化
    """

    def __init__(self, num_output_channels=1):
        super(Grayscale, self).__init__()
        self.num_output_channels = num_output_channels

    def __call__(self, image, target):
        image = F.rgb_to_grayscale(image, num_output_channels=self.num_output_channels)
        return image, target


class ColorJitter(object):
    """
    随机改变图像亮度、对比度、饱和度和色调
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitter, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(
            brightness: Optional[List[float]],
            contrast: Optional[List[float]],
            saturation: Optional[List[float]],
            hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(
            torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(
            torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, image, target):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                image = F.adjust_brightness(image, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                image = F.adjust_contrast(image, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                image = F.adjust_saturation(image, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                image = F.adjust_hue(image, hue_factor)

        return image, target


class GaussianBlur(object):
    """
    对图像高斯滤波，模拟纸张震动
    """

    def __init__(self, kernel_size, sigma=(0.1, 0.2)):
        super(GaussianBlur, self).__init__()
        self.kernel_size = _setup_size(kernel_size,
                                       'Kernel size should be a tuple/list of two integers')
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError('Kernel size value should be an odd and positive number.')

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError('If sigma is a single number, it must be positive.')
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError('sigma values should be positive and of the form (min, max).')
        else:
            raise ValueError('sigma should be a single number or a list/tuple with length 2.')

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def __call__(self, image, target):
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        image = F.gaussian_blur(image, self.kernel_size, [sigma, sigma])
        return image, target


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size
