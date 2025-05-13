# -*- coding: utf-8 -*-
#               
#            ___________________________________
#           /____   ____________   ____________/           
#              /  /    /_    _/  /                     
#        ___  /  /___   /  / /  /___    ____ 
#       /  / /  / __  \/  / /  / __  \/   _ /       
#      /  /_/  /   ___/  /_/  /   ___/  /__
#      \______/ \____/\___/__/ \____/ \___/      
#       
#

"""
Model  

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActorCritic(nn.Module):
    def __init__(self, input_channels=1, action_size=1, std=0.5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2),  # -> [16, 40, 40]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),               # -> [32, 18, 18]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),               # -> [32, 8, 8]
            nn.ReLU(),
            nn.Flatten()                                              # -> [2048]
        )

        conv_output_size = 32 * 8 * 8  # for 84x84

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(128, action_size)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)

    def forward(self, x):
        #print(f"[MODEL] Input shape: {x.shape}")
        x = x.to(self.device)
        features = self.conv_layers(x)
        #print(f"[MODEL] Features shape (after conv): {features.shape}")
        x = self.fc(features)
        #print(f"[MODEL] After FC shape: {x.shape}")
        mu = torch.tanh(self.actor_head(x))
        value = self.critic_head(x)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist, value

    def extract_features(self, x):
        # returns features after flatten (1D), useful to "visualise" what the robot extracts from images
        x = x.to(self.device)
        with torch.no_grad():
            for layer in self.conv_layers:
                x = layer(x)
                print(f"[FEATURE DEBUG] After {layer.__class__.__name__}: {x.shape}")
                if isinstance(layer, nn.Flatten):
                    break
        return x

    def extract_feature_maps(self, x):
        # Before flatten -> [C, H, W]
        x = x.to(self.device)
        with torch.no_grad():
            x = self.conv_layers[0](x)  # Conv1
            x = self.conv_layers[1](x)  # ReLU
            x = self.conv_layers[2](x)  # Conv2
            x = self.conv_layers[3](x)  # ReLU
            x = self.conv_layers[4](x)  # Conv3
            x = self.conv_layers[5](x)  # ReLU
        return x.squeeze(0)  # returns [C, H, W]

