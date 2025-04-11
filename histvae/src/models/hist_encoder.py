# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

histogram encoder with convolutional neural network

@author: tadahaya
"""
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        assert dim in [1, 2, 3], "dim must be 1 (1D), 2 (2D), or 3 (3D)"
        Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
        BatchNorm = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[dim]
        self.conv1 = Conv(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Skip connection
        out = self.relu(out)
        return out


class HistDeepConv(nn.Module):
    """ Deep Residual Network for Histogram Convolutional Encoder """
    def __init__(self, output_channels=64, num_blocks=4, dim=2):
        super().__init__()
        assert dim in [1, 2, 3], "dim must be 1 (1D), 2 (2D), or 3 (3D)"
        Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
        BatchNorm = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[dim]
        AdaptiveAvgPool = {1: nn.AdaptiveAvgPool1d, 2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}[dim]
        self.conv1 = Conv(in_channels=1, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*[ResidualBlock(output_channels, dim) for _ in range(num_blocks)])
        self.global_avg_pool = AdaptiveAvgPool(1)  # (B, C, 1...)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x
    

# HistEncoder
class HistEncoder(nn.Module):
    def __init__(self, hist_dim:int, hidden_dim:int, num_blocks:int, output_dim:int, prob_dropout=0.3):
        super().__init__()
        # convolutional layer
        assert hist_dim in [1, 2, 3], "hist_dim must be 1 (1D), 2 (2D), or 3 (3D)"
        self.hist_conv = HistDeepConv(output_channels=hidden_dim, num_blocks=num_blocks, dim=hist_dim)
        # MLP for output
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Linear
            nn.LayerNorm(hidden_dim),           # LayerNorm
            nn.ReLU(),                         # ReLU
            nn.Dropout(prob_dropout),          # Dropout
            nn.Linear(hidden_dim, output_dim)  # Linear for output
        )

    def forward(self, x):
        # convolution
        x = self.hist_conv(x)
        # MLP
        x = self.mlp(x)
        return x
    

# HistShallowEncoder
class HistShallowConv(nn.Module):
    def __init__(self, output_channels, dim):
        super().__init__()
        # convolutional layer
        assert dim in [1, 2, 3], "dim must be 1 (1D), 2 (2D), or 3 (3D)"
        Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
        self.conv = nn.Conv(
            in_channels=1, # input channel is 1 because of histogram
            out_channels=output_channels,
            kernel_size=3,  # filter size 3x3
            stride=1,       # stride 1
            padding=1
            )      # padding 'same'
        # global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # finally (1, 1)

    def forward(self, x):
        # convolution
        x = self.conv(x)
        # global average pooling
        x = self.global_avg_pool(x)
        # flatten as embedding vector
        x = x.view(x.size(0), -1)
        return x


class HistShallowEncoder(nn.Module):
    def __init__(self, hist_dim:int, hidden_dim:int, output_dim:int, prob_dropout=0.3):
        super().__init__()
        # convolutional layer
        assert hist_dim in [1, 2, 3], "hist_dim must be 1 (1D), 2 (2D), or 3 (3D)"
        self.hist_conv = HistShallowConv(output_channels=hidden_dim, dim=hist_dim)
        # MLP for output
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Linear
            nn.ReLU(),                         # ReLU
            nn.Dropout(prob_dropout),          # Dropout
            nn.Linear(hidden_dim, output_dim)  # Linear for output
        )

    def forward(self, x):
        # convolution
        x = self.hist_conv(x)
        # MLP
        x = self.mlp(x)
        return x