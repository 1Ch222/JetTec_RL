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

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jettec_robot.line_following.model_line import CNNActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
GAMMA = 0.95
TAU = 0.95
GRADIENT_CLIP = 5
NUM_EPOCHS = 10
CLIP = 0.15
BETA = 0.001
LR = 1e-4
EPSILON = 1e-5


class Agent:
    def __init__(self, num_agents, input_channels, action_size):
        self.model = CNNActorCritic(input_channels=input_channels, action_size=action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=EPSILON)

    def act(self, states):
        states = states.to(device)
        dist, values = self.model(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
        return actions, log_probs, values

    def step(self, rollout, num_agents, writer=None, step_idx=0):
        storage = [None] * (len(rollout) - 1)
        advantage = torch.zeros((num_agents, 1), device=device)

        for i in reversed(range(len(rollout) - 1)):
            state, action, log_prob, reward, done, value = rollout[i]
            if i == len(rollout) - 2:
                next_return = rollout[i + 1][-1]

            reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(1)
            done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1)
            next_value = rollout[i + 1][-1]

            g_return = reward + GAMMA * next_return * done
            next_return = g_return

            td_error = reward + GAMMA * next_value - value
            advantage = advantage * TAU * GAMMA * done + td_error
            storage[i] = [state, action, log_prob, g_return, advantage]

        states, actions, log_probs, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*storage))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.learn(states, actions, log_probs, returns, advantages, writer, step_idx)

    def learn(self, states, actions, log_probs_old, returns, advantages, writer=None, step_idx=0):
        for _ in range(NUM_EPOCHS):
            for _ in range(states.size(0) // BATCH_SIZE):
                idx = np.random.randint(0, states.size(0), BATCH_SIZE)
                s, a, lp_old, r, adv = states[idx], actions[idx], log_probs_old[idx], returns[idx], advantages[idx]

                dist, values = self.model(s)
                log_probs = dist.log_prob(a).sum(dim=1, keepdim=True)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - lp_old)
                obj = ratio * adv
                obj_clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv

                policy_loss = -torch.min(obj, obj_clipped).mean() - BETA * entropy
                value_loss = (r - values).pow(2).mean()
                total_loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()

                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                self.optimizer.step()

                if writer:
                    writer.add_scalar("Loss/Policy", policy_loss.item(), step_idx)
                    writer.add_scalar("Loss/Value", value_loss.item(), step_idx)
                    writer.add_scalar("Loss/Entropy", entropy.item(), step_idx)
                    writer.add_scalar("gradients/norm", grad_norm, step_idx)
                    writer.add_scalar("actions/std", actions.std().item(), step_idx)

