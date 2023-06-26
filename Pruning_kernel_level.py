'''
Author: ccbao 1689940525@qq.com
Date: 2023-06-25 21:18:11
LastEditors: ccbao 1689940525@qq.com
LastEditTime: 2023-06-26 07:59:25
FilePath: /Pruning/Pruning.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch.nn as nn
import torch
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,padding=1) # 输入通道为3
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True) # inplace=True表示直接修改输入值，而不是返回
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1) # 输入通道为3
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True) 
        self.conv4 = nn.Conv2d(128,128,kernel_size=3,padding=1) # 输入通道为3
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True) 
        self.fc1 = nn.Linear(128 * 4 *4,1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

## 卷积核剪枝
def prune_conv_layer(layer,prune_rate):
    if isinstance(layer,nn.Conv2d):
        weight = layer.weight.data.cpu().numpy()
        print(f"weight shape: { weight.shape }")
        num_weights = weight.size 
        num_prune = int(num_weights * prune_rate)
        # 计算一组卷积来求L2范数
        norm_per_filter = np.sqrt(np.sum(weight ** 2,axis=(1,2,3)))
        # 根据L2范数排序，选择一定比例的filter，将filter里的所有element置为零
        indices = np.argsort(norm_per_filter)[-num_prune:]
        print(f"indices: {indices}")
        weight[indices] = 0
        layer.weight.data = torch.from_numpy(weight).to(layer.weight.device)
        
    

net = Net()
prune_rate = 0.2

for layer in net.modules():
    prune_conv_layer(layer,prune_rate)
        