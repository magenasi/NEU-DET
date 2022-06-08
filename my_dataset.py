# coding=utf-8

"""
Author: xiezhenqing
date: 2022/5/26 13:41
desc: 自定义dataset类，主要是要将每张图片的缺陷框同时返回
"""

import json
import os

import torch
from torch.utils.data import Dataset
from lxml import etree
from PIL import Image

IMAGE_FORMATS = ['JPEG', 'PNG']


def resize_box(box, original_size, new_size):
	"""
	将box参数根据图像的缩放情况进行相应的缩放
	:param box: box坐标位置
	:param original_size: 图像缩放前的尺寸
	:param new_size: 图像缩放后的尺寸
	:return: 调整后的box框坐标信息
	"""
	ratio = float(new_size / original_size)

	xmin, ymin, xmax, ymax = box
	xmin = int(xmin * ratio)
	xmax = int(xmax * ratio)
	ymin = int(ymin * ratio)
	ymax = int(ymax * ratio)

	return [xmin, ymin, xmax, ymax]


def resize(image, target, new_size):
	"""
	对图片和gt框进行相应缩放
	:param image: 待缩放的图片
	:param target: 待缩放图片对应的标注信息
	:param new_size: 缩放后的图片大小
	:return: 缩放后图片和对应的标注信息
	"""

	h, _ = image.size

	image = image.resize((new_size, new_size), Image.BICUBIC)

	if target is None:
		return image, target

	# 根据图像的缩放比例缩放gt框
	bbox = target['box']
	bbox = resize_box(bbox, h, new_size)
	target['box'] = bbox

	return image, target


class MyDataSet(Dataset):
	"""读取解析NEU-DET数据集"""
	def __init__(self, neu_root, transforms=None, txt_name: str = 'train.txt'):

		assert os.path.exists(neu_root), "path: '{}' does not exist.".format(neu_root)

		self.root = neu_root
		self.img_root = os.path.join(self.root, 'JPEGImages')
		self.annotations_root = os.path.join(self.root, 'Annotations')

		# 读取 train.txt 或者 val.txt 文件
		txt_path = os.path.join(self.root, 'ImageSets', txt_name)
		assert os.path.exists(txt_path), 'not found {} file.'.format(txt_name)

		with open(txt_path) as read:
			self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
							 for line in read.readlines() if len(line.strip()) > 0]

		# 检查文件
		assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(
			txt_path)
		for xml_path in self.xml_list:
			assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

		# 读取 class_indict
		json_file = './NEU_classes.json'
		assert os.path.exists(json_file), "{} file not exists.".format(json_file)
		json_file = open(json_file, 'r')
		self.class_dict = json.load(json_file)
		json_file.close()

		self.transforms = transforms

	def __len__(self):
		return len(self.xml_list)

	def __getitem__(self, idx):
		# 读取xml文件
		xml_path = self.xml_list[idx]
		with open(xml_path) as f:
			xml_str = f.read()
		xml = etree.fromstring(xml_str)
		data = self.parse_xml_to_dict(xml)['annotation']
		# 因为有的data['filename']没有'.jpg'后缀，所以更改一下
		img_path = os.path.join(self.img_root, data['filename'].split('.')[0] + '.jpg')
		image = Image.open(img_path)
		if image.format not in IMAGE_FORMATS:
			raise ValueError("Image '{}' format not in IMAGE_FORMATS".format(img_path))

		# 此处因为该数据集一张图片可能有多个缺陷目标，所以需要boxes为列表（嵌套）
		# 在钞票数据集中每张图片只有一个缺陷目标，所以box只是列表，里面有一个bbox的四个坐标值
		boxes = []  # 存储一张图片中所有gt框的坐标信息

		area_start = 0  # gt框面积大小
		assert 'object' in data, "{} lack of object information.".format(xml_path)
		box = []
		labels = []
		label = self.class_dict[data['object'][0]['name']]
		labels.append(label)
		for obj in data['object']:
			x_min = float(obj['bndbox']['xmin'])
			x_max = float(obj['bndbox']['xmax'])
			y_min = float(obj['bndbox']['ymin'])
			y_max = float(obj['bndbox']['ymax'])

			# 检查数据，看是否有标记错误
			if x_max <= x_min or y_max <= y_min:
				print("Warning: in '{}' xml, there are some bbox w/h <= 0".format(xml_path))
				continue

			area = (x_max - x_min) * (y_max - y_min)
			# 更新gt框的坐标位置，只使用面积最大的gt框
			if area > area_start:
				box = [x_min, y_min, x_max, y_max]
				area_start = area
		boxes.append(box)

		# 转换为torch.Tensor
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.as_tensor(labels, dtype=torch.int64)

		target = {'box': boxes, 'label': labels}

		if self.transforms is not None:
			image, target = self.transforms(image, target)

		return image, target

	def parse_xml_to_dict(self, xml):
		"""将xml文件解析成字典形式
		Args:
			xml: 由lxml.etree解析XML文件而返回的xml tree

		Returns:
			包含XML内容的python字典
		"""

		if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
			return {xml.tag: xml.text}

		result = {}
		for child in xml:
			child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
			if child.tag != 'object':
				result[child.tag] = child_result[child.tag]
			else:
				# 因为object可能有多个，所以需要放入列表中
				if child.tag not in result:
					result[child.tag] = []
				result[child.tag].append(child_result[child.tag])
		return {xml.tag: result}

	@staticmethod
	def collate_fn(batch):
		return tuple(zip(*batch))


if __name__ == '__main__':

	import transforms
	from draw_box_utils import draw_box
	from PIL import Image
	import json
	import matplotlib.pyplot as plt
	import torchvision.transforms as ts
	import random

	# 读取class_indict
	category_index = {}
	try:
		json_file = open('./NEU_classes.json', 'r')
		class_dict = json.load(json_file)
		category_index = {v: k for k, v in class_dict.items()}
	except Exception as e:
		print(e)
		exit(-1)

	data_transform = {
		'train': transforms.Compose([transforms.ToTensor(),
									 # transforms.Normalize(mean=(0.5), std=(0.5)),
									 # transforms.ResizeImageSize(224),
									 transforms.RandomHorizontalFlip(0.5),
									 transforms.RandomVerticalFlip(0.5),
									 # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
									 # transforms.GaussianBlur(3),
									 # transforms.Grayscale()
									 ]),
		"val": transforms.Compose([transforms.ToTensor()])
	}

	# load train data set
	train_data_set = MyDataSet('./NEU-DET', data_transform["train"], "train.txt")
	print(len(train_data_set))
	figure = plt.figure()
	for index in random.sample(range(0, len(train_data_set)), k=5):
		img, target = train_data_set[index]
		print(target)
		img = ts.ToPILImage()(img)

		draw_box(img,
				 target['box'].numpy(),
				 target['label'].numpy(),
				 [1 for i in range(len(target['box'].numpy()))],
				 category_index,
				 thresh=0.5,
				 line_thickness=5)
		plt.imshow(img)
		plt.show()
