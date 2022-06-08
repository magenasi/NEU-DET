"""
单独实现GeneralizedTransform类，因为此项目对数据图像特殊处理，需要将图像中目标框进行同步缩放，
并将标准化处理同样整合进类中
"""

import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor

from image_list import ImageList


def resize_box(box, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """将boxes参数根据图像的缩放情况进行相应的缩放

    Args:
        box (Tensor): box坐标位置
        original_size (List[int]): 图像缩放前的尺寸
        new_size (List[int]): 图像缩放后的尺寸
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=box.device) / 
        torch.tensor(s_orig, dtype=torch.float32, device=box.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    
    ratio_height, ratio_width = ratios
    
    # 调整tensor维度, box [4]
    xmin, ymin, xmax, ymax = box.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def resize_image(image, new_size):
    # type: (Tensor, float) -> Tensor
    img_w, img_h = torch.tensor(image.shape[-2:])
    # 因为图像为正方形，所以只算出宽或者高的一个缩放比例即可
    img_w = float(img_w)
    scale_factor = new_size / img_w
        
    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def batch_images(images, size_divisible=32):
    # type: (List[Tensor], int) -> Tensor
    """将一批图像打包成一个batch返回
    Args:
        images: 输入的一批图像
        size_divisible: 将图像高和宽调整到该数的整数倍

    Returns:
        batched_imgs: 打包成一个batch后的tensor数据
    """

    img_size = list(images[0].shape)

    stride = float(size_divisible)
    # 将height向上调整到stride的整数倍
    img_size[1] = int(math.ceil(float(img_size[1]) / stride) * stride)
    # 将width向上调整到stride的整数倍
    img_size[2] = int(math.ceil(float(img_size[2]) / stride) * stride)

    # [batch, channel, height, width]
    batch_shape = [len(images)] + img_size

    # 创建shape为batch_shape且值全部为0的tensor
    batched_imgs = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batched_imgs):
        # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
        # 这样保证输入到网络中一个batch的每张图片的shape相同
        # copy_: Copies the elements from src into self tensor and returns self
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


class GeneralizedTransform(nn.Module):
    """
    输入模型前将input / target进行transformation
    
    包括： 标准化和调整image和boxes大小（标准化也可以在pytorch自带的torchvision.transforms.Compose中实现）
          同时将输入数据整合为适合输入网络计算的格式
    
    返回图像为ImageList，boxes为List[Tensor]
    """
    def __init__(self, new_size, image_mean, image_std):
        super().__init__()
        if not isinstance(new_size, (list, tuple)):
            new_size = (new_size,)
        
        self.new_size = new_size        # 指定图像调整后的尺寸
        self.image_mean = image_mean    # 指定图像在标准化处理中的均值
        self.image_std = image_std      # 指定图像在标准化处理中的方差
        
    def normalize(self, image):
        """标准化处理"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None]: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]
    
    def resize(self, image, target):
        """
        将图片缩放到指定的大小范围内，并对应缩放boxes信息
        Args:
            image: 输入的图片
            target: 输入图片的相关信息（包括bboxes信息）

        Returns:
            image: 缩放后的图片
            target: 缩放boxes后的图片相关信息
        """
        # 图像尺寸为 [channel, height, width]
        h, w = image.shape[-2:]
        
        image = resize_image(image, float(self.new_size[0]))
        
        if target is None:
            return image, target
        
        # 根据图像的缩放比例缩放box
        bbox = target['box']
        bbox = resize_box(bbox, [h, w], image.shape[-2:])
        target['box'] = bbox
        
        return image, target

    def forward(self,
                images,         # type: List[Tensor]
                targets=None,   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None
            
            if image.dim() != 3:
                raise ValueError('images is expected to be a list of 3d tensors '
                                 'of shape [C, H, W], got {}'.format(image.shape))
                
            image = self.normalize(image)                           # 对图像进行标准化处理
            image, target_index = self.resize(image, target_index)  # 对图像和对应的boxes缩放到指定大小 
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
                
        # 记录resize后的图像尺寸
        image_sizes = [img.shape[-2:] for img in images]
        images = batch_images(images)  # 将images打包成一个batch
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))
            
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets
