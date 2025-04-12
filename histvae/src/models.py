# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019
 
Histogram Variational Autoencoder (VAE) for 1D, 2D, and 3D data.

route2
- add dropout layer

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


# Encoder block (convolution + BatchNorm + activation + residual connection)
class ResidualConvBlock(nn.Module):
    """
    Convolutional block with residual connection (for Encoder)

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dim, dropout_conv=0.3):
        super().__init__()
        Conv = get_conv_layer(dim)
        BatchNorm = get_batchnorm_layer(dim)
        self.conv1 = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout_conv) if dropout_conv > 0 else nn.Identity()
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm(out_channels)
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            # dimension adjustment for skip connection (if needed)
            self.skip = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=stride),
                BatchNorm(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply dropout after the first convolution
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResidualConvTransposeBlock(nn.Module):
    """
    Deconvolutional block with residual connection (for Decoder)
 
    """
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding,
            dim, dropout_conv=0.3, activation="relu"
            ):
        super().__init__()
        ConvT = get_conv_transpose_layer(dim)
        BatchNorm = get_batchnorm_layer(dim)
        self.convt1 = ConvT(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout_conv) if dropout_conv > 0 else nn.Identity()
        self.convt2 = ConvT(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm(out_channels)
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                ConvT(in_channels, out_channels, kernel_size=stride, stride=stride),
                BatchNorm(out_channels)
            )
        # ReLU by default, but can be changed to Sigmoid for the last layer
        act_layer = nn.ReLU(inplace=True) if activation == "relu" else nn.Sigmoid()
        self.activation = act_layer

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.convt1(x)))
        out = self.dropout(out)  # Apply dropout after the first deconvolution
        out = self.bn2(self.convt2(out))
        out += identity
        return self.activation(out)


# VAE Encoder to latent space (encoding)
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, dim):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                ResidualConvBlock(in_channels, h_dim, kernel_size=3, stride=2, padding=1, dim=dim)
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

# VAE Decoder to reconstruct the input (decoding)
class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dims, dim):
        super().__init__()
        hidden_dims = hidden_dims[::-1]
        layers = []
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(
                ResidualConvTransposeBlock(in_channels, h_dim, kernel_size=4, stride=2, padding=1, dim=dim)
            )
            in_channels = h_dim
        layers.append(
            ResidualConvTransposeBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation="sigmoid", dim=dim)
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

# main ConvVAE class
class ConvVAE(nn.Module):
    def __init__(self, input_shape, latent_dim=128, hidden_dims=None):
        """
        Variational Autoencoder (VAE) for 1D, 2D, and 3D data.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input data like (channels, height, width) or (channels, length)

        latent_dim: int
            Dimension of the latent space

        hidden_dims: list of int
            List of hidden dimensions for the encoder and decoder

        """
        super().__init__()
        self.dim = DataDim(len(input_shape) - 1)
        hidden_dims = hidden_dims or [32, 64, 128, 256]
        # Construct Encoder
        self.encoder = Encoder(input_shape[0], hidden_dims, dim=self.dim)
        # calculate the output shape of the encoder
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            enc_out = self.encoder(sample_input)
        self.enc_out_shape = enc_out.shape[1:]
        enc_out_dim = enc_out.numel()
        # Latent space parameters (mu, logvar)
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        # mapping from latent space to reconstruction features
        self.fc_decode = nn.Linear(latent_dim, enc_out_dim)
        # Construct Decoder
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

    def vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        """
        Compute the VAE loss function.
 
        Parameters       
        ----------
        recon_x: torch.Tensor
            Reconstructed output from the decoder   
        
        x: torch.Tensor
            Original input data
        
        mu, logvar: torch.Tensor
            Latent space parameters (mean and log variance)

        beta: float
            Weight for the KL divergence term (default: 1.0)

        """

        batch_size = x.size(0)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / batch_size
        # for clear understanding, we use sum instead of mean
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss


class LinearHead(nn.Module):
    def __init__(
            self, pretrained, latent_dim:int, num_classes:int, num_layers:int=2,
            hidden_head:int=512, dropout_head:float=0.3, frozen:bool=False
            ):
        """
        Parameters
        ----------
        pretrained: pre-trained model

        latent_dim: dimension of the representation

        num_classes: number of classes

        num_layers: number of layers in MLP

        hidden_head: number of hidden units in MLP
            int or list of int

        dropout_head: dropout rate

        """
        super().__init__()
        # pretrained model
        self.pretrained = pretrained
        self.frozen = frozen
        if self.frozen:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        # MLP
        layers = []
        if isinstance(hidden_head, int):
            hidden_head = [hidden_head] * num_layers
        in_features = latent_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_head[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_head))
            in_features = hidden_head[i]
        layers.append(nn.Linear(hidden_head[i], num_classes))  # output layer
        self.linear_head = nn.Sequential(*layers)


    def forward(self, x):
        mu, logvar = self.pretrained.encode(x)
        z = self.pretrained.reparameterize(mu, logvar)
        recon = self.pretrained.decode(z)
        logits = self.linear_head(mu)  # use the latent representation for classification
        return logits, recon, mu, logvar


    def vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        batch_size = x.size(0)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / batch_size
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss