# -*- coding: utf-8 -*-
#               
#            ___________________________________
#           /___   _______/  /_   _____________/           
#              /  /    /_    _/  /                     
#        ___  /  /___   /  / /  /___    ____ 
#       /  / /  / __  \/  / /  / __  \/   _ /       
#      /  /_/  /   ___/  /_/  /   ___/  /__    
#      \______/ \____/\___/__/ \____/ \___/      
#       
#

"""
CNN-based Actor-Critic model for reinforcement learning.
Processes images to output action distributions and value estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActor(nn.Module):
    def __init__(self, input_channels=1, action_size=1, hidden_size=128, std_scale=1.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.std_scale = std_scale

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_output_size = 32 * 8 * 8
        self.actor_head = nn.Linear(conv_output_size, action_size)
        self.std_head = nn.Linear(conv_output_size, action_size)

        self.to(self.device)

    def forward(self, x, hidden=None):
        B, T, C, H, W = x.shape
        x = x[:, -1, :, :, :]  # on prend uniquement la dernière frame
        x = self.conv_layers(x)

        mu = torch.tanh(self.actor_head(x))
        log_std = self.std_head(x)
        std = (F.softplus(log_std) + 1e-3) * self.std_scale

        dist = torch.distributions.Normal(mu, std)
        return dist, None  # plus de hidden state

    def extract_feature_maps(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            for i in range(6):
                x = self.conv_layers[i](x)
        return x.squeeze(0)

class CNNCritic(nn.Module):
    def __init__(self, input_channels=1, hidden_size=128, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_output_size = 32 * 8 * 8
        self.critic_head = nn.Linear(conv_output_size, 1)

        self.to(self.device)

    def forward(self, x, hidden=None):
        B, T, C, H, W = x.shape
        x = x[:, -1, :, :, :]  # on garde la dernière image
        x = self.conv_layers(x)
        value = self.critic_head(x)
        return value, None
