"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Challenge CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


__all__ = ["Challenge"]


class Challenge(nn.Module):
    def __init__(self) -> None:
        """Optimized CNN architecture with regularization"""
        super().__init__()

        # convolution layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)

        # we will use Global Average Pooling instead of FC layers
        self.gap = nn.AdaptiveAvgPool2d(1)

        # final classifier
        self.fc = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

        # initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
        
        # nn.init.kaiming_normal_(self.fc.weight,
        #                         mode='fan_in',
        #                         nonlinearity='linear')
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #input: (N, 3, 64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # x = F.relu(self.bn4(self.conv4(x)))
        # x = F.max_pool2d(x, 2)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
