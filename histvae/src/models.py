# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019
 
Histogram Variational Autoencoder (VAE) for 1D, 2D, and 3D data.
route0

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

# Enum for clearly managing data dimensions
class DataDim(Enum):
    ONE_D = 1
    TWO_D = 2
    THREE_D = 3

# functions to get the number of dimensions from the input shape
def get_conv_layer(dim: DataDim):
    return {DataDim.ONE_D: nn.Conv1d, DataDim.TWO_D: nn.Conv2d, DataDim.THREE_D: nn.Conv3d}[dim]

def get_conv_transpose_layer(dim: DataDim):
    return {DataDim.ONE_D: nn.ConvTranspose1d, DataDim.TWO_D: nn.ConvTranspose2d, DataDim.THREE_D: nn.ConvTranspose3d}[dim]

def get_batchnorm_layer(dim: DataDim):
    return {DataDim.ONE_D: nn.BatchNorm1d, DataDim.TWO_D: nn.BatchNorm2d, DataDim.THREE_D: nn.BatchNorm3d}[dim]

# Encoder block (convolution + BatchNorm + activation)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dim):
        super().__init__()
        Conv = get_conv_layer(dim)
        BatchNorm = get_batchnorm_layer(dim)
        self.block = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size, stride, padding),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# Decoder block (deconvolution + BatchNorm + activation)
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dim, activation="relu"):
        super().__init__()
        ConvT = get_conv_transpose_layer(dim)
        BatchNorm = get_batchnorm_layer(dim)

        # 
        # 最終層のactivationはsigmoidで0〜1に制限し、他はrelu
        act_layer = nn.ReLU(inplace=True) if activation == "relu" else nn.Sigmoid()

        self.block = nn.Sequential(
            ConvT(in_channels, out_channels, kernel_size, stride, padding),
            BatchNorm(out_channels),
            act_layer,
        )

    def forward(self, x):
        return self.block(x)


# VAEのEncoder（潜在変数への符号化）
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, dim):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                ConvBlock(in_channels, h_dim, kernel_size=3, stride=2, padding=1, dim=dim)
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# VAEのDecoder（潜在変数からの復元）
class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dims, dim):
        super().__init__()
        hidden_dims = hidden_dims[::-1]
        layers = []
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(
                ConvTransposeBlock(in_channels, h_dim, kernel_size=4, stride=2, padding=1, dim=dim)
            )
            in_channels = h_dim
        layers.append(
            ConvTransposeBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dim=dim, activation="sigmoid")
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# VAEの本体モデル (Variational Autoencoder)
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim=128, hidden_dims=None):
        super().__init__()
        self.dim = DataDim(len(input_shape) - 1)

        hidden_dims = hidden_dims or [32, 64, 128, 256]

        # Encoderの構築
        self.encoder = Encoder(input_shape[0], hidden_dims, dim=self.dim)

        # Encoderの出力形状から中間層サイズを自動計算
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            enc_out = self.encoder(sample_input)
        self.enc_out_shape = enc_out.shape[1:]
        enc_out_dim = enc_out.numel()

        # VAE特有の潜在空間パラメータ (mu, logvar)
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # 潜在空間から復元用特徴量へのマッピング
        self.fc_decode = nn.Linear(latent_dim, enc_out_dim)

        # Decoderの構築
        self.decoder = Decoder(input_shape[0], hidden_dims, dim=self.dim)

    def encode(self, x):
        enc_out = self.encoder(x).flatten(start_dim=1)
        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        dec_input = self.fc_decode(z).view(-1, *self.enc_out_shape)
        return self.decoder(dec_input)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# VAE用の損失関数（再構成誤差＋潜在空間正則化）
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss