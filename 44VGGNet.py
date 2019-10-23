import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()              # 224, 224, 3

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),      # 224, 224, 64
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, padding=1),     # 224, 224, 64
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                     # 112, 112, 64
            nn.Conv2d(64, 128, 3, 1, padding=1),    # 112, 112, 128
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, padding=1),   # 112, 112, 128
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                     # 56, 56, 128
            nn.Conv2d(128, 256, 3, 1, padding=1),   # 56, 56, 256
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, padding=1),   # 56, 56, 256
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, padding=1),   # 56, 56, 256
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                     # 28, 28, 256
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, padding=1),   # 28, 28, 512
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, padding=1),   # 28, 28, 512
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, padding=1),   # 28, 28, 512
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                     # 14, 14, 512
            nn.Conv2d(512, 512, 3, 1, padding=1),   # 14, 14, 512
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, padding=1),   # 14, 14, 512
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, padding=1),   # 14, 14, 512
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                     # 7, 7, 512
        )

        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

        self._initlize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out








