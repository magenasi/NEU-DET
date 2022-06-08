# coding=utf-8

"""
Author: xiezhenqing
Email: 441798804@qq.com
date: 2022/5/26 14:36
desc: 训练模型
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import transforms
from my_dataset import MyDataSet
from network import MyModel
from train_eval_utils import train_one_epoch, evaluate

# 参数设置
# ----------------------------------------------------------------------------------------

# batch_size
BATCH_SIZE = 32

# num_classes
NUM_CLASSES = 6

# 预训练权重地址
WEIGHTS = './resnetv1d50.pth'

# 优化器选择
SGD = True  # 其默认参数为torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False)
Adam = False  # 其默认参数为torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, maximize=False)
AdamW = False  # 其默认参数为torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, *, maximize=False)

# 学习率   还可根据优化器的选择添加 momentum 等不同参数
LR = 0.005

# 学习率调整策略
COSINEANNEALINGLR = True  # cosine annealing schedule
STEPLR = False  # Decays the learning rate of each parameter group by gamma every step_size epochs
EXPONENTIALLR = False  # 指数衰减

# epochs
EPOCHS = 30

# 模型保存地址
SAVE_PATH = './save_weights'

# ----------------------------------------------------------------------------------------


def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('using {} device.'.format(device))
	print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

	# 实例化SummaryWriter对象
	tb_writer = SummaryWriter(log_dir="runs/train_experiment")

	if os.path.exists('./weights') is False:
		os.makedirs('./weights')

	# 定义训练以及预测时输入模型前对数据图像进行初步预处理方法
	data_transform = {
		'train': transforms.Compose([transforms.ToTensor(),
									 # # transforms.Normalize(mean=(0.5), std=(0.5)),
									 # transforms.ResizeImageSize(224),
									 transforms.RandomHorizontalFlip(0.5),
									 transforms.RandomVerticalFlip(0.5),
									 # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
									 # transforms.GaussianBlur(3),
									 # transforms.Grayscale()
									 ]),
		'val': transforms.Compose([transforms.ToTensor()])
	}

	data_root = os.getcwd()
	if os.path.exists(os.path.join(data_root, 'NEU-DET')) is False:
		raise FileNotFoundError(
			"NEU-DET dose not in path: '{}'.".format(data_root)
		)

	# 实例化训练数据集
	train_dataset = MyDataSet('./NEU-DET', data_transform["train"], "train.txt")
	train_num = len(train_dataset)

# {"crazing": 1, "inclusion": 2, "patches": 3, "pitted_surface": 4, "rolled-in_scale": 5, "scratches": 6}

	# 在windows系统下，nw设为0？
	nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])
	print('Using {} dataloader workers every process'.format(nw))

	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=BATCH_SIZE,
											   shuffle=True,
											   pin_memory=True,
											   num_workers=0,
											   collate_fn=train_dataset.collate_fn)

	# 实例化验证数据集
	val_dataset = MyDataSet('./NEU-DET', data_transform["val"], "val.txt")
	val_num = len(val_dataset)

	val_loader = torch.utils.data.DataLoader(val_dataset,
											 batch_size=BATCH_SIZE,
											 shuffle=False,
											 pin_memory=True,
											 num_workers=0,
											 collate_fn=val_dataset.collate_fn)

	print('using {} images for training, {} images for validation.'.format(train_num, val_num))

	# 实例化模型
	model = MyModel(pretrained_path=WEIGHTS, num_classes=NUM_CLASSES)
	model.to(device)

	# 将模型写入tensorboard
	# 此使模型前向传播时需要缺陷框的坐标信息，所以没有成功
	# init_img = torch.zeros((1, 3, 224, 224), device=device)
	# tb_writer.add_graph(model, init_img)

	# 定义优化器
	params = [p for p in model.parameters() if p.requires_grad]
	if SGD:
		optimizer = optim.SGD(params, lr=LR)
	elif Adam:
		optimizer = optim.Adam(params, lr=LR)
	elif AdamW:
		optimizer = optim.AdamW(params, lr=LR)
	else:
		raise ValueError('优化器未在上述范围')

	# 混合精度训练
	scaler = torch.cuda.amp.GradScaler()

	# 学习率调整策略
	if COSINEANNEALINGLR:
		# T_max: 最大迭代次数；eta_min: 最小学习率
		lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
	elif STEPLR:
		# step_size: 学习率衰减周期；gamma: 学习率衰减乘法因子.Default: 0.1
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
	elif EXPONENTIALLR:
		# 每个 epoch 衰减每个参数组的学习率。当 last_epoch=-1 时，设置初始 lr 为 lr。
		lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
	else:
		raise ValueError('学习率调整策略没在上述范围')

	epochs = EPOCHS

	best_acc = 0.0

	for epoch in range(epochs):
		# train
		mean_loss = train_one_epoch(
			model=model,
			optimizer=optimizer,
			data_loader=train_loader,
			epoch=epoch,
			epochs=epochs,
			device=device
		)

		# 更新学习率
		lr_schedular.step()

		# validate
		acc = evaluate(
			model=model,
			data_loader=val_loader,
			epoch=epoch,
			epochs=epochs,
			device=device
		)

		# 向tensorboard添加loss, acc, lr
		print('[epoch {}] accuracy: {}'.format(epoch, round(acc, 3)))
		tags = ['train_loss', 'accuracy', 'learning_rate']
		tb_writer.add_scalar(tags[0], mean_loss, epoch)
		tb_writer.add_scalar(tags[1], acc, epoch)
		tb_writer.add_scalar(tags[2], optimizer.param_groups[0]['lr'], epoch)

		if not os.path.exists(SAVE_PATH):
			os.makedirs(SAVE_PATH)

		# 保存模型权重
		if acc > best_acc:
			best_acc = acc
			torch.save(model.state_dict(), './save_weights/model-{:.3f}-{}.pth'.format(best_acc, epoch))

	print('Finished Training')


if __name__ == '__main__':
	main()
