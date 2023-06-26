'''
Author: ccbao 1689940525@qq.com
Date: 2023-06-25 23:14:38
LastEditors: ccbao 1689940525@qq.com
LastEditTime: 2023-06-26 15:59:35
FilePath: /Pruning/Pruning_vector_level.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import matplotlib.pyplot as plt
import torch

from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)

X,Y = np.meshgrid(x,y)

Z = -1 * np.sin(np.sqrt(X **2 + Y ** 2)) / (np.sqrt(X**2 + Y **2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z,cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('loss')
plt.show()



pruning_rate = 50
model = torch.nn.Sequential(torch.nn.Linear(10,5),torch.nn.ReLU(),torch.nn.Linear(5,1))

input_tensor = torch.randn(1,10)
output_tensor = model(input_tensor)
loss = torch.sum(output_tensor)
loss.backward()

grad_weight_product_list = []
for name,param in model.named_parameters():
    # print(name)
    # print(param.data)
    # print(param.grad)
    # 基于梯度和权重大小的混合标准
    if 'weight' in name:
        grad_weight_product = torch.abs(param.grad * param.data)
        grad_weight_product_list.append(grad_weight_product)
        
all_product_values = torch.cat([torch.flatten(x) for x in grad_weight_product_list])
print(all_product_values)

threshold = np.percentile(all_product_values.cpu().detach().numpy(),pruning_rate)

for name,param in model.named_parameters():
    if 'weight' in name:
        mask = torch.where(torch.abs(param.grad * param.data) >= threshold,1,0)
        param.data *= mask.float()