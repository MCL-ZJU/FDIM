"""ResNet-18 feature extractor adapted from torchvision's ResNet implementation.

Reference:
    Torchvision model builder for ResNet-18:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
"""

import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.max_pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.max_pool(self.relu(self.bn1(self.conv1(x))))
        features = []
        for i in range(1, 5):
            x = getattr(self, f"layer{i}")(x)
            features.append(x)
        return features
