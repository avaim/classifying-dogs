"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import Target
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Target"]


class Target(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3, # RGB
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2
        )

        # Max Pooling Layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2
        )

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=8,
            kernel_size=5,
            stride=2,
            padding=2
        )

        # Fully Connected Layer
        self.fc_1 = nn.Linear(
            in_features=8*2*2,
            out_features=2
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)

        # initialize convolutional layers
        for conv in [self.conv1, self.conv2, self.conv3]:
            # weight initialization
            num_input_channels = conv.weight.shape[1]
            std = (1.0 / (5 * 5 * num_input_channels)) ** 0.5
            nn.init.normal_(conv.weight, mean=0.0, std=std)
            
            # bias initialization
            nn.init.constant_(conv.bias, 0.0)

        # initialize the parameters for [self.fc_1]
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=(1.0/self.fc_1.in_features)**0.5)

        # bias initialization
        nn.init.constant_(self.fc_1.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: (N, 3, 64, 64)

        # Conv1 + ReLU
        x = F.relu(self.conv1(x)) # (N, 16, 32, 32)

        # Max Pooling
        x = self.pool(x) # (N, 16, 16, 16)

        # Conv2 + ReLU
        x = F.relu(self.conv2(x)) # (N, 64, 8, 8)

        # Max Pooling
        x = self.pool(x) # (N, 64, 4, 4)

        # Conv3 + ReLU
        x = F.relu(self.conv3(x)) # (N, 8, 2, 2)

        # Flattening
        x = torch.flatten(x, 1) # (N, 32)

        # FC Layer
        x = self.fc_1(x) # (N, 2)

        return x
