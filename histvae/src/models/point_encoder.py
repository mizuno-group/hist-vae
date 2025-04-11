# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

Point encoder with shared MLP and self-attention pooling

@author: tadahaya
"""
import torch
import torch.nn as nn

# Shared MLP
class SharedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, prob_dropout=0.5):
        super(SharedMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Linear
            nn.LayerNorm(hidden_dim),          # Layer Normalization
            nn.ReLU(),                         # ReLU
            nn.Dropout(prob_dropout),          # Dropout
            nn.Linear(hidden_dim, output_dim)  # Linear for output
        )

    # apply shared MLP to each point
    def forward(self, x):
        return self.mlp(x) # (batch, num_points, input_dim) -> (batch, num_points, output_dim)

# Self-Attention Pooling with non-linear transformation
class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, prob_dropout=0.3):
        super(SelfAttentionPooling, self).__init__()
        # attention layer (2 linear layers and ReLU)
        self.attn_layer1 = nn.Linear(input_dim, hidden_dim)  # Linear
        self.attn_relu = nn.ReLU()  # ReLU
        self.dropout = nn.Dropout(prob_dropout)  # Dropout
        self.attn_layer2 = nn.Linear(hidden_dim, input_dim)  # Linear
        self.softmax = nn.Softmax(dim=-1)  # Softmax getting attention weights

    def forward(self, x):
        # x (batch_size, num_points, feature_dim)
        # calculate attetion weights
        attn_weights = self.attn_layer1(x)  # (batch_size, num_points, hidden_dim)
        attn_weights = self.attn_relu(attn_weights)  # ReLU
        attn_weights = self.dropout(attn_weights)  # Dropout
        attn_weights = self.attn_layer2(attn_weights)  # (batch_size, num_points, feature_dim)
        attn_weights = self.softmax(attn_weights)  # (batch_size, num_points, feature_dim)
        # aggregate features
        aggregated_features = torch.sum(attn_weights * x, dim=1)  # (batch_size, feature_dim)
        return aggregated_features, attn_weights


class LinearSelfAttentionPooling(nn.Module):
    """ fully linear version of self-attention pooling """
    def __init__(self, input_dim, hidden_dim):
        super(LinearSelfAttentionPooling, self).__init__()
        self.attn = nn.Linear(input_dim, hidden_dim)  # Linear
        self.softmax = nn.Softmax(dim=-1)  # Softmax getting attention weights
        # attention weights adjustment linear layer
        self.attn_adjustment = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x (batch_size, num_points, feature_dim)
        # calculate attetion weights
        attn_weights = self.softmax(self.attn(x))  # (batch_size, num_points, hidden_dim)
        # attn_weights adjustment to fit feature_dim
        attn_weights = self.attn_adjustment(attn_weights)  # (batch_size, num_points, feature_dim)
        # aggregate features
        aggregated_features = torch.sum(attn_weights * x, dim=1)  # (batch_size, feature_dim)
        return aggregated_features, attn_weights


# PointEncoder
class PointEncoder(nn.Module):
    def __init__(self, input_dim, hidden_mlp, output_mlp, hidden_attn, dropout_mlp=0.5, dropout_attn=0.3):
        super(PointEncoder, self).__init__()
        self.shared_mlp = SharedMLP(input_dim, hidden_mlp, output_mlp, dropout_mlp)
        self.self_attention_pooling = SelfAttentionPooling(output_mlp, hidden_attn, dropout_attn)

    def forward(self, x):
        out = self.shared_mlp(x) # (batch, num_points, input_dim) -> (batch, num_points, output_mlp)
        out, weights = self.self_attention_pooling(out) # (batch, num_points, output_mlp) -> (batch, output_mlp)
        return out, weights