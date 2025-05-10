"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed


__all__ = ["Source"]


class Source(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
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
        self.fc1 = nn.Linear(
            in_features=8*2*2,
            out_features=8
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        set_random_seed()

        for conv in [self.conv1, self.conv2, self.conv3]:
            # weight initialization
            num_input_channels = conv.weight.shape[1]
            std = (1.0 / (5 * 5 * num_input_channels)) ** 0.5
            nn.init.normal_(conv.weight, mean=0.0, std=std)

            # bias initialization
            nn.init.constant_(conv.bias, 0.0)
        
        # weight initialization for FC Layer
        nn.init.normal_(self.fc1.weight, mean=0.0, std=(1.0/self.fc1.in_features)**0.5)

        # bias initalization for FC Layer
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward propagation for a batch of input examples. Pass the input array
        through layers of the model and return the output after the final layer.

        Args:
            x: array of shape (N, C, H, W) 
                N = number of samples
                C = number of channels
                H = height
                W = width

        Returns:
            z: array of shape (1, # output classes)
        """
        
        # Conv1 + ReLU
        x = F.relu(self.conv1(x))

        # Max Pooling
        x = self.pool(x)

        # Conv2 + ReLU
        x = F.relu(self.conv2(x))

        # Max Pooling
        x = self.pool(x)

        # Conv3 + ReLu
        x = F.relu(self.conv3(x))

        # Flattening
        x = torch.flatten(x, 1)

        # FC Layer
        x = self.fc1(x)

        return x
