'''
Author: ccbao 1689940525@qq.com
Date: 2023-06-25 21:18:11
LastEditors: ccbao 1689940525@qq.com
LastEditTime: 2023-06-26 20:06:13
FilePath: /Pruning/Pruning.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

"""
目前剪枝的框架有两种：
1. 训练，剪枝，微调
2. 训练时剪枝

下面这段代码实现的是：
第一种方案，训练，剪枝，微调
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,10)
    
    def forward(self,x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x
    

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
train_dataset = datasets.MNIST('./data',train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)


def train(model, dataloader, criterion, optimizer, device='cpu', num_epochs=10):
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs.view(inputs.size(0),-1))
            loss = criterion(outputs,targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
    return model

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
net = train(net, train_loader, criterion, optimizer,device='cpu', num_epochs=2)

# save model
torch.save(net.state_dict(),'net.pth')

# 剪枝
def prune_model(model,pruning_rate=0.5, method='global'):
    for name,param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if method == 'global':
                threshold = np.percentile(np.abs(tensor), pruning_rate * 100)
            else:
                threshold = np.percentile(np.abs(tensor),pruning_rate * 100,axis=1,keepdims=True)
            mask = np.abs(tensor) > threshold
            param.data = torch.FloatTensor(tensor * mask.astype(float).to(param.device))
        
        
net.load_state_dict(torch.load('net.pth'))
prune_model(net,pruning_rate=0.5, method='global')

torch.save(net.state_dict(),'pruned_model.pth')

# 用比较低的学习率去微调模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=1e-4)
finetuned_model = train(net, train_loader, criterion,optimizer,device='cpu',num_epochs=10)

# 保存模型
torch.save(finetuned_model.state_dict(),"finetuned_model.pth")


  