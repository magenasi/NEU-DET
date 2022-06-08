# coding=utf-8

"""
Author: xiezhenqing
Email: 441798804@qq.com
date: 2022/5/26 14:36
desc: 一个epoch训练过程和验证过程
"""

import sys

from tqdm import tqdm
import torch
from torch import Tensor
from typing import List


def train_one_epoch(model, optimizer, data_loader, epoch, epochs, device):
	model.train()

	# 定义损失函数
	loss_function = torch.nn.CrossEntropyLoss()

	# 定义平均损失
	mean_loss = torch.zeros(1).to(device)

	optimizer.zero_grad()

	data_loader = tqdm(data_loader, file=sys.stdout)

	for step, [images, targets] in enumerate(data_loader):
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		optimizer.zero_grad()
		logits = model(images, targets)

		labels = torch.cat([target['label'] for target in targets], 0)

		loss = loss_function(logits, labels.to(device))
		loss.backward()
		mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # 更新平均损失

		if not torch.isfinite(loss):
			print('WARNING: non-finite loss, ending training ', loss)
			sys.exit(1)

		optimizer.step()

		data_loader.desc = 'train epoch[{}/{}]  loss: {:.3f}'.format(epoch+1, epochs, mean_loss.item())

	return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, epoch, epochs, device):
	model.eval()

	# 用于存储预测正确的样本个数
	sum_num = torch.zeros(1).to(device)
	# 统计验证集样本总数目
	num_samples = len(data_loader.dataset)

	# 打印验证进度
	data_loader = tqdm(data_loader, desc="validation...", file=sys.stdout)

	for step, data in enumerate(data_loader):
		images, targets = data
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
		preds = model(images, targets)

		# 将targets中每个图片target的'label'标签拼接为shape为[batch_size]的一维tensor
		labels = torch.cat([target['label'] for target in targets], 0)
		# loss = loss_function(outputs, val_labels.to(device))
		preds = torch.max(preds, dim=1)[1]
		sum_num += torch.eq(preds, labels.to(device)).sum()

	# 计算预测正确的比例
	acc = sum_num.item() / num_samples

	data_loader.desc = 'valid epoch[{}/{}]'.format(epoch+1, epochs)

	return acc
