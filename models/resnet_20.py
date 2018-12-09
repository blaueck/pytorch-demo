import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import BasicBlock

class Resnet20(nn.Module):

    def __init__(self, in_shape, n_class, *args, **kwargs):
        super(Resnet20, self).__init__(*args, **kwargs)
        self.in_shape = in_shape

        self.conv1 = nn.Conv2d(in_shape[0], 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.in_planes = 16

        block = BasicBlock
        self.layer1 = self._make_layers(block, 16, 2, 1)
        self.layer2 = self._make_layers(block, 32, 2, 2)
        self.layer3 = self._make_layers(block, 64, 2, 2)

        # compute the height of layer3's feature map
        fh = in_shape[1]
        for _ in range(2):
            fh = math.ceil(fh / 2)
        self.fh = int(fh)

        self.fc = nn.Linear(64, n_class)
    
    def _make_layers(self, block, planes, n_block, stride=1):
        layers = []

        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers.append(block(self.in_planes, planes, stride, downsample))
        for _ in range(1, n_block):
            layers.append(block(planes, planes))

        self.in_planes = planes
        return nn.Sequential(*layers)
    

    def forward(self, images):
        out = self.conv1(images)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, self.fh)

        shape = out.shape
        out = torch.reshape(out, (shape[0], -1))
        out = self.fc(out)
        return out
        
