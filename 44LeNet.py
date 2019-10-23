import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # 1, 32, 32

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 5, 0))   # 6, 28, 28
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))      # 6, 14, 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5, 0))  # 16, 10, 10
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))      # 16, 5, 5
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('fc2', nn.Linear(120, 84))
        layer3.add_module('fc3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        out = self.layer3(x)
        return out















