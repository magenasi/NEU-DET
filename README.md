# NEU-DET分类任务

## 文件结构：

```
|--network.py: 模型网络结构
|--my_dataset.py: 自定义dataset类
|--train_eval_utils.py: 一个epoch内训练和验证过程
|--draw_box_utils.py: 辅助函数，便于在图片上标注出缺陷框
|--image_list.py: 借鉴Faster-RCNN，对批量图片进行组合
|--NEU_classes.json: 标签文件
|--split_data.py: 划分验证集和训练集
|--train.py: 模型训练过程
|--transform.py: 定义GeneralizedTransform类，数据输入网络时调整图片和gt框的大小，
                 对图片标准化，借鉴pytorch官方Faster-RCNN源码
|--transforms.py: 一些普通的数据增强方法，适用于有gt框的图片，如图片翻转时gt框随之
                  翻转，在加载dataset时使用
```

## 预训练权重下载（下载后直接放入项目根目录即可）

- 使用mmclassification提供的ResNetV1D-50提供的预训练权重，下载链接：[https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth]()

- 注意，下载后的预训练权重记得重命名，比如train.py中读取的是`resnetv1d50.pth`文件，要改为这个名字

## 数据集，采用东北大学NEU-DET数据集

格式按VOC数据集放置：
![]()

## 训练方法

直接使用train.py训练脚本，若要修改参数，可在train.py文件中修改

## 训练结果

![]()

![]()
