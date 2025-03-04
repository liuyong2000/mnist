import torch
import torchvision
from torchvision import datasets, transforms

# 定义数据预处理
transform = transforms.Compose([transforms.ToTensor()])

# 加载MNIST训练集
train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)

# 查看数据集的基本信息
print(f"训练集样本数量: {len(train_ds)}")
print(f"第一个样本的类型: {type(train_ds[0])}")
print(f"第一个样本的长度: {len(train_ds[0])}")
#print(f"第一个样本: {train_ds[0]}")
print(f"第一个样本的图像数据类型: {type(train_ds[0][0])}")
print(f"第一个样本的图像数据形状: {train_ds[0][0].shape}")
print(f"第一个样本的标签数据类型: {type(train_ds[0][1])}")
print(f"第一个样本的标签值: {train_ds[0][1]}")