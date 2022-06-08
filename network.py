# coding=utf-8

"""
Author: xiezhenqing
date: 2022/5/26 13:41
desc: 构建模型架构
"""
import os
from typing import OrderedDict

import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.poolers import MultiScaleRoIAlign

from mmdet.models.backbones.resnet import ResNetV1d

from transform import GeneralizedTransform


class BackBone(nn.Module):
    """
    构建骨干特征提取网络 --> ResNetV1d(depth=50) + FPN
    """
    def __init__(self, return_layers=None, pretrained_path=''):
        super(BackBone, self).__init__()

        if return_layers is None:
            return_layers = [1, 2, 3, 4]

        # 返回的特征层个数肯定小于5大于0
        assert min(return_layers) > 0 and max(return_layers) < 5
        # 字典形式，为 ResNetV1d50 输出的特征层{'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        self.return_layers = {f'layer{k}': str(v) for v, k in enumerate(return_layers)}

        # 定义ResNetV1d50
        self.backbone = ResNetV1d(depth=50)

        if pretrained_path != '':
            assert os.path.exists(pretrained_path), '{} is not exists.'.format(pretrained_path)
            # 载入预训练权重
            print(self.backbone.load_state_dict(torch.load(pretrained_path), strict=False))

        # 从backbone返回指定返回特征层
        self.body = IntermediateLayerGetter(self.backbone, self.return_layers)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
            extra_blocks=None
        )

    def forward(self, inputs):
        x = self.body(inputs)
        x = self.fpn(x)

        return x


class MyModel(nn.Module):
    """
    整体模型结构，包括backbone(ResNetV1d50 + FPN), MultiScaleRoIAlign, classifier
    """
    def __init__(self, return_layers=None, pretrained_path='', box_roi_pool=None,
                 num_classes=6, image_mean=None, image_std=None, new_size=224):
        super(MyModel, self).__init__()

        self.backbone = BackBone(return_layers=return_layers, pretrained_path=pretrained_path)

        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        # 暂时为ImageNet数据集标准
        # if image_mean is None:
        #     image_mean = [0.485, 0.456, 0.406]
        # if image_std is None:
        #     image_std = [0.229, 0.224, 0.225]

        # 由于钢铁表面数据集为灰度图，所以改变image_mean和image_std
        if image_mean is None:
            image_mean = [0.5]
        if image_std is None:
            image_std = [0.5]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        self.transform = GeneralizedTransform(new_size, image_mean, image_std)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=[7, 7],
                sampling_ratio=0
            )
        self.box_roi_pool = box_roi_pool

        # 定义最后的classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, images, targets=None):
        """

        :param images: (list[Tensor]): images to be processed
        :param targets: (list[Dict[Tensor]]): ground-truth box present in the image (optional)
        :return:
        """
        if self.training and targets is None:
            raise ValueError('在训练时，图像中应有缺陷目标')

        if self.training:
            assert targets is not None
            for target in targets:
                # 判断传入的target的gt坐标参数是否符合规定
                box = target['box']
                if isinstance(box, torch.Tensor):
                    if len(box.shape) != 2 or box.shape[-1] != 4:
                        raise ValueError(
                            'target_box应是一个shape为[4]的tensor，但却得到的是{:}.'.format(box.shape))
                else:
                    raise ValueError('target_box应是tensor格式，但得到的是{:}类型.'.format(type(box)))

        images, targets = self.transform(images, targets)   # 对图像进行预处理

        # 将图像输入backbone得到特征图
        features = self.backbone(images.tensors)

        # 若只在一层特征图上预测，将feature放入有序字典中，并编号为“0”
        if isinstance(features, torch.Tensor):
            features = OrderedDict[('0', features)]

        # 定义输入roialign层的box列表
        boxes = []
        for target in targets:
            box = target['box']
            boxes.append(box)

        x = self.box_roi_pool(features, boxes, [(224, 224)])
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    net = MyModel()
    print(net)
