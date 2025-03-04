# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from PIL import Image
from tqdm import tqdm

torch.manual_seed(2)

# 定义模型
class NnNet(nn.Module):
   def __init__(self):
      super().__init__()
      self.linear1 = nn.Linear(784, 250)  # 输入 784，输出 250
      self.bn1 = nn.BatchNorm1d(250)
      self.linear2 = nn.Linear(250, 100)  # 输入 250，输出 100
      self.bn2 = nn.BatchNorm1d(100)
      self.linear3 = nn.Linear(100, 10)   # 输入 100，输出 10

   def forward(self, X):
      # X = F.relu(self.linear1(X))  # 第一层 + ReLU 激活
      X = F.relu(self.bn1(self.linear1(X)))
      #X = F.relu(self.linear2(X))  # 第二层 + ReLU 激活
      X = F.relu(self.bn2(self.linear2(X)))
      X = self.linear3(X)          # 第三层（无激活函数）
      return X


train_ds = datasets.MNIST('data',train=True,download=True, transform=transforms.Compose(
   [transforms.ToTensor(),                   # 将图像转换为 Tensor [1, 28, 28]
    transforms.Lambda(lambda x: torch.flatten(x))]))  # 展平为一维 Tensor [784]]))
batch_size = 100
validation_split = .1
shuffle_dataset = True
random_seed= 2

# Creating data indices for training and validation splits:
dataset_size = len(train_ds)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
   np.random.seed(random_seed)
   np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)


train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                                sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=False,download=True,
      transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])),
      batch_size=batch_size,shuffle=True)

model = NnNet()
optimizer = optim.SGD(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

train_errors = []
train_acc = []
val_errors = []
val_acc = []
n_train = len(train_loader)*batch_size
n_val = len(validation_loader)*batch_size

epoch = 10

for i in range(epoch):  # 假设进行10个epoch的训练
   total_loss = 0
   total_acc = 0  
   c = 0
    
   # 在训练数据加载器上加上tqdm包装器以显示进度
   with tqdm(total=len(train_loader), desc=f'Epoch {i+1}/{epoch} Training') as pbar:
      for images, labels in train_loader:
         # images = images.cuda()
         # labels = labels.cuda()
         
         optimizer.zero_grad()
         output = model(images)
         loss = criterion(output, labels)
         loss.backward()
         optimizer.step()
         
         total_loss += loss.item()
         total_acc += torch.sum(torch.max(output, dim=1)[1] == labels).item() * 1.0    
         c += 1
         
         # 更新进度条
         pbar.update(1)
    
   # validation部分同样可以加上tqdm来显示进度
   total_loss_val = 0
   total_acc_val = 0
   c = 0
   with tqdm(total=len(validation_loader), desc=f'Epoch {i+1}/{epoch} Validation') as pbar:
      for images, labels in validation_loader:
         # images = images.cuda()
         # labels = labels.cuda()
         output = model(images)
         loss = criterion(output, labels)
         
         total_loss_val += loss.item()
         total_acc_val += torch.sum(torch.max(output, dim=1)[1] == labels).item() * 1.0
         c += 1
         
         pbar.update(1)
   
   # 记录误差和准确率
   train_errors.append(total_loss / n_train)
   train_acc.append(total_acc / n_train)
   val_errors.append(total_loss_val / n_val)
   val_acc.append(total_acc_val / n_val)

   # 可以在这里打印每个epoch后的结果
   print(f"Epoch {i+1}, Train Loss: {total_loss/n_train:.4f}, Train Acc: {total_acc/n_train:.4f}, Val Loss: {total_loss_val/n_val:.4f}, Val Acc: {total_acc_val/n_val:.4f}")

print("Training complete")


total_acc = 0
for images,labels in test_loader:
   output = model(images)
   total_acc+=torch.sum(torch.max(output,dim=1)[1]==labels).item()*1.0

print("Test accuracy :",total_acc/len(test_loader.dataset))