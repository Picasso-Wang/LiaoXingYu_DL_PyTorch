'''
input image size: 224, 224, 3

conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)      # (224-7+6)/2+1=112.         112, 112, 64
bn1 = nn.BatchNorm2d(64)
relu = nn.ReLU(inplace=True)
maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                                                 # 56, 56, 64

layer1:
    block1:
        nn.Conv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)             # 56, 56, 64
        bn1 = nn.BatchNorm2d(planes=64)
        self.relu = nn.ReLU(inplace=True)
        nn.Conv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)             # 56, 56, 64
        bn1 = nn.BatchNorm2d(planes=64)

        For x, there is no downsample, so x: (56, 56, 64) -> (56, 56, 64)
        self.relu(x+residual)

    block2:
        nn.Conv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)             # 56, 56, 64
        bn1 = nn.BatchNorm2d(planes=64)
        self.relu = nn.ReLU(inplace=True)
        nn.Conv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)             # 56, 56, 64
        bn1 = nn.BatchNorm2d(planes=64)

        For x, there is no downsample, so x: (56, 56, 64) -> (56, 56, 64)
        self.relu(x+residual)

layer2:
    block1:
        nn.Conv2d(in_planes=64, out_planes=128, kernel_size=3, stride=2, padding=1, bias=False)            # 28, 28, 128
        bn1 = nn.BatchNorm2d(planes=128)
        self.relu = nn.ReLU(inplace=True)
        nn.Conv2d(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1, bias=False)           # 28, 28, 128
        bn1 = nn.BatchNorm2d(planes=128)

        For x, there is a downsample:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes=64, planes * block.expansion=128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * block.expansion=128),)
        so x: (56, 56, 64) -> (28, 28, 128)
        self.relu(x+residual)

    block2:
        nn.Conv2d(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1, bias=False)           # 28, 28, 128
        bn1 = nn.BatchNorm2d(planes=128)
        self.relu = nn.ReLU(inplace=True)
        nn.Conv2d(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1, bias=False)           # 28, 28, 128
        bn1 = nn.BatchNorm2d(planes=128)
        self.relu(x+residual)

layer3:
    block1:
        nn.Conv2d(in_planes=128, out_planes=256, kernel_size=3, stride=2, padding=1, bias=False)           # 14, 14, 256
        bn1 = nn.BatchNorm2d(planes=256)
        self.relu = nn.ReLU(inplace=True)
        nn.Conv2d(in_planes=256, out_planes=256, kernel_size=3, stride=1, padding=1, bias=False)           # 14, 14, 256
        bn1 = nn.BatchNorm2d(planes=256)

        For x, there is a downsample:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes=128, planes * block.expansion=256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * block.expansion=256),)
        so x: (28, 28, 128) -> (14, 14, 256)
        self.relu(x+residual)

    block2:
        nn.Conv2d(in_planes=256, out_planes=256, kernel_size=3, stride=1, padding=1, bias=False)           # 14, 14, 256
        bn1 = nn.BatchNorm2d(planes=256)
        self.relu = nn.ReLU(inplace=True)
        nn.Conv2d(in_planes=256, out_planes=256, kernel_size=3, stride=1, padding=1, bias=False)           # 14, 14, 256
        bn1 = nn.BatchNorm2d(planes=256)
        self.relu(x+residual)

layer4:
    block1:
        nn.Conv2d(in_planes=256, out_planes=512, kernel_size=3, stride=2, padding=1, bias=False)            # 7, 7, 512
        bn1 = nn.BatchNorm2d(planes=512)
        self.relu = nn.ReLU(inplace=True)
        nn.Conv2d(in_planes=512, out_planes=512, kernel_size=3, stride=1, padding=1, bias=False)            # 7, 7, 512
        bn1 = nn.BatchNorm2d(planes=512)

        For x, there is a downsample:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes=256, planes * block.expansion=512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * block.expansion=512),)
        so x: (14, 14, 256) -> (7, 7, 512)
        self.relu(x+residual)

    block2:
        nn.Conv2d(in_planes=512, out_planes=512, kernel_size=3, stride=1, padding=1, bias=False)            # 7, 7, 512
        bn1 = nn.BatchNorm2d(planes=512)
        self.relu = nn.ReLU(inplace=True)
        nn.Conv2d(in_planes=512, out_planes=512, kernel_size=3, stride=1, padding=1, bias=False)            # 7, 7, 512
        bn1 = nn.BatchNorm2d(planes=512)
        self.relu(x+residual)

avgpool = nn.AvgPool2d(7, stride=1)                                                                             # 512
fc = nn.Linear(512 * block.expansion=512, num_classes=1000)                                                     # 1000
'''



