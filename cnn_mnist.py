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

class CnnNet(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1,10,kernel_size=5,stride=1)
      self.conv2 = nn.Conv2d(10,10,kernel_size=5,stride=1)
      self.pool = nn.MaxPool2d(kernel_size=2,stride=2) #2x2 maxpool
      self.fc1 = nn.Linear(4*4*10,100)
      self.fc2 = nn.Linear(100,10)
  
   def forward(self,x):
      x = F.relu(self.conv1(x)) #24x24x10
      x = self.pool(x) #12x12x10
      x = F.relu(self.conv2(x)) #8x8x10
      x = self.pool(x) #4x4x10    
      x = x.view(-1, 4*4*10) #flattening
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x

train_ds = datasets.MNIST('data',train=True,download=True, transform=transforms.Compose([transforms.ToTensor()]))
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
      transform=transforms.Compose([transforms.ToTensor()])),batch_size=batch_size,shuffle=True)

model = CnnNet()
optimizer = optim.SGD(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

train_errors = []
train_acc = []
val_errors = []
val_acc = []
n_train = len(train_loader)*batch_size
n_val = len(validation_loader)*batch_size

for i in range(10):  # 假设进行10个epoch的训练
   total_loss = 0
   total_acc = 0  
   c = 0
    
   # 在训练数据加载器上加上tqdm包装器以显示进度
   with tqdm(total=len(train_loader), desc=f'Epoch {i+1}/10 Training') as pbar:
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
   with tqdm(total=len(validation_loader), desc=f'Epoch {i+1}/10 Validation') as pbar:
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