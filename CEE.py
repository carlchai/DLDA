import numpy as np
import torch
from torchvision import transforms, datasets
import torchvision
import torch.nn.functional as F
from torch import optim, device
import cv2
import torch.nn as nn
transformtrain = transforms.Compose(
    [

        transforms.Resize((28,28)),
        transforms.Normalize((0.1307,), (0.3081,)),  # 0.1307是均值，0.3081是标准差
        transforms.CenterCrop(24),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),

    ]
)
transform = transforms.Compose(
    [
        transforms.Resize((28,28)),
        transforms.ToTensor(),  # 把数据转换成张量
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)
tra = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

train_dataset = datasets.ImageFolder(root='./mnist/train',transform=tra)
test_dataset = datasets.ImageFolder(root='./mnist/test',transform=tra)

num_class = len(train_dataset.classes)

# 数据加载器 加载数据集
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=64)

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        out = self.relu(x + residual)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=num_class):
        super(ResNet18, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resblock(in_channels=64),
            resblock(in_channels=64),
            nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            resblock(in_channels=128),
            resblock(in_channels=128),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1),
            resblock(in_channels=256),
            resblock(in_channels=256),
            nn.Dropout(0.25)

        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=(2, 2), padding=1),
            resblock(in_channels=512),
            resblock(in_channels=512),

        )
        self.block6 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=outputs),
        )
    def initial(self, *models):
        for m in models:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.reshape(x.shape[0], -1)
        features = self.fc(x)
        return features

network = ResNet18(in_channels=3, resblock=ResidualBlock)
network = network.to(device=device)
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)




def train():
    network.train()
    correct = 0
    for batch_id,(data,target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        if batch_id % 100 == 0:
            print('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_id * len(data),
            len(train_dataloader.dataset),100. * batch_id / len(train_dataloader), loss.item()))
    acc = 100. * correct / len(train_dataloader.dataset)
    print(' Accuracy: {}/{} ({:.0f}%)\n'.format( correct,len(train_dataloader.dataset), acc))

def ttest():
    network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_dataloader:
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    test_loss /= len(test_dataloader.dataset)
    accuracy = 100. * correct / len(test_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
    len(test_dataloader.dataset),accuracy))

#交叉熵：99.56
#
for i in range(100):
    train()
    ttest()
    print("epoch:",i)
