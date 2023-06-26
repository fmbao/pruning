'''
Author: ccbao 1689940525@qq.com
Date: 2023-06-25 21:18:11
LastEditors: ccbao 1689940525@qq.com
LastEditTime: 2023-06-26 12:48:05
FilePath: /Pruning/Pruning.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch.nn as nn
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
    

def visualize_tensor(tensor,batch_spacing=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for batch in range(tensor.shape[0]):
        for channel in range(tensor.shape[1]):
            for i in range(tensor.shape[2]):
                for j in range(tensor.shape[3]):
                    x,y,z = j + (batch * (tensor.shape[3] + batch_spacing)), i, channel
                    color = 'red' if tensor[batch,channel,i,j] == 0 else 'gray'
                    ax.bar3d(x,y,z,1,1,1,shade=True,color=color,edgecolor='black',alpha=0.9)
                    
    ax.set_xlabel('width')
    ax.set_zlabel('hight')
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.zaxis.labelpad = 15
    plt.show()


# 结构化剪枝
### 滤波器剪枝





### 通道剪枝


### 层剪枝


## 卷积核剪枝
def prune_conv_layer(conv_layer,prune_method, percentile=20, vis=True):
    prune_layer = conv_layer.copy()
    if prune_method == "fine_grained":
        prune_layer[np.abs(prune_layer) < 0.5] = 0
        pass
    
    if prune_method == "vector_level":
        # 计算沿着最后一个维度(col) 计算L2范数
        l2_norm = np.linalg.norm(prune_layer, axis=-1)
        pass
    
    if prune_method == "channel_level":
        # 计算每个channel的L2范数
        l2_norm = np.sqrt(np.sum(prune_layer ** 2, axis=(-4,-2,-1)))
        l2_norm = l2_norm.reshape(1,-1)
        l2_norm = np.repeat(l2_norm,prune_layer.shape[0],axis=0)
        pass
    
    if prune_method == "filter_level":
        # 计算每个filter的L2范数
        l2_norm = np.sqrt(np.sum(prune_layer ** 2, axis= (-3,-2,-1)))
        pass
    
    if prune_method == "kernel_level":
        # 计算每个kernle 的L2范数
        l2_norm = np.linalg.norm(prune_layer,axis=(-2,-1))
        pass
    # pass       
    

# net = Net()
# prune_rate = 0.2

    # percentile = 0.4
    
    threshold = np.percentile(l2_norm, percentile)
    mask = l2_norm < threshold
    print(prune_layer.shape)
    print(mask.shape)
    print("------------------------------------")
    prune_layer[mask] = 0
    if vis:
        visualize_tensor(prune_layer)
    return prune_layer

tensor = np.random.uniform(low=-1,high=1, size=(3,10,4,5))

# pruned_tensor = prune_conv_layer(tensor,'vector_level',vis=True)
# pruned_tensor = prune_conv_layer(tensor,'filter_level',vis=True)
pruned_tensor = prune_conv_layer(tensor,'channel_level',vis=True)
# pruned_tensor = prune_conv_layer(tensor,'layer_level',vis=True)
# for layer in net.modules():
#     prune_conv_layer(layer,prune_rate)
        