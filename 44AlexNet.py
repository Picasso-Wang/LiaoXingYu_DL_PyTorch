import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()  # 224, 224, 3

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, padding=2),     # 55, 55, 96
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),                     # 27, 27, 96
            nn.Conv2d(96, 256, 5, 1, padding=2),    # 27, 27, 256
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),                     # 13, 13, 256
            nn.Conv2d(256, 384, 3, 1, padding=1),   # 13, 13, 384
            nn.ReLU(True),
            nn.Conv2d(384, 384, 3, 1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, 1, padding=0),   # 13, 13, 256
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),                     # 6, 6, 256
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out










