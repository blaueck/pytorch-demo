import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):

    def __init__(self, in_shape, n_class, *args, **kwargs):
        super(SimpleCNN, self).__init__(*args, **kwargs)
        self.in_shape = in_shape

        self.conv1 = nn.Conv2d(in_shape[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        fh = in_shape[1]
        for _ in range(3):
            fh = math.ceil(fh / 2)
        fh = int(fh)
        self.fc = nn.Linear(fh * fh * 128, n_class)
    
    def forward(self, images):
        out = self.conv1(images)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        shape = out.shape
        out = torch.reshape(out, (shape[0], -1))
        out = self.fc(out)
        return out
        
