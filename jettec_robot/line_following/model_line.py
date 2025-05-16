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

class CNNActorCritic(nn.Module):
    def __init__(self, input_channels=1, action_size=1, std=0.5, device=None):
        """
        Args:
            input_channels (int): Number of input channels (e.g., 1 for grayscale).
            action_size (int): Number of action outputs.
            std (float): Initial standard deviation for action distribution.
            device (torch.device, optional): Computation device.
        """
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_output_size = 32 * 8 * 8  # Expected output for 84x84 input size

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(128, action_size)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)

        # Move entire model to the selected device
        self.to(self.device)

    def forward(self, x):
        features = self.conv_layers(x)
        x = self.fc(features)
        mu = torch.tanh(self.actor_head(x))
        value = self.critic_head(x)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist, value

    def extract_features(self, x):
        """Returns extracted features after the flattening layer."""
        x = x.to(self.device)   # <<<<<< AJOUT
        with torch.no_grad():
            for layer in self.conv_layers:
                x = layer(x)
                print(f"[FEATURE DEBUG] After {layer.__class__.__name__}: {x.shape}")
                if isinstance(layer, nn.Flatten):
                    break
        return x

    def extract_feature_maps(self, x):
        """Returns feature maps before the flatten layer."""
        x = x.to(self.device)   # <<<<<< AJOUT
        with torch.no_grad():
            for i in range(6):
                x = self.conv_layers[i](x)
        return x.squeeze(0)
