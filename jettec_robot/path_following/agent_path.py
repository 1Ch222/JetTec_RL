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

"""
Agent module for training a line-following policy using Proximal Policy Optimization (PPO).
Handles action selection, advantage estimation, and policy/value network updates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jettec_robot.path_following.model_path import CNNPathActor, CNNPathCritic

# === Hyperparameters ===
BATCH_SIZE = 32
GAMMA = 0.90
TAU = 0.95
GRADIENT_CLIP = 5
NUM_EPOCHS = 15
CLIP = 0.1
LR = 1e-4
EPSILON = 1e-4

class Agent:
    def __init__(self, num_agents, input_channels, action_size, use_caps=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = CNNPathActor(input_channels, action_size, device=self.device)
        self.critic = CNNPathCritic(input_channels, device=self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR, eps=EPSILON)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR, eps=EPSILON)

        self.use_caps = use_caps
        self.BETA = 0.01  # Entropy weight
        self.lambda_grad_caps = 0.80  # Grad-CAPS weight

    def act(self, seq_states: torch.Tensor, hidden=None):
        seq_states = seq_states.to(self.device)
        dist, _ = self.actor(seq_states)
        value, _ = self.critic(seq_states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
        return actions, log_probs, value, None

    def step(self, rollout, num_agents, writer=None, step_idx=0):
        storage = [None] * (len(rollout) - 1)
        advantage = torch.zeros((num_agents, 1), device=self.device)

        for i in reversed(range(len(rollout) - 1)):
            seq, action, log_prob, reward, done, value = rollout[i]
            next_return = rollout[i + 1][5] if i == len(rollout) - 2 else next_return
            next_value = rollout[i + 1][5]

            reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
            done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

            g_return = reward + GAMMA * next_return * done
            next_return = g_return

            td_error = reward + GAMMA * next_value - value
            advantage = advantage * TAU * GAMMA * done + td_error

            storage[i] = [seq.detach(), action.detach(), log_prob.detach(), g_return.detach(), advantage.detach()]

        sequences, actions, log_probs_old, returns, advantages = zip(*storage)
        seq_batch = torch.cat(sequences, dim=0)
        actions = torch.cat(actions, dim=0)
        log_probs_old = torch.cat(log_probs_old, dim=0)
        returns = torch.cat(returns, dim=0)
        advantages = torch.cat(advantages, dim=0)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.learn(seq_batch, actions, log_probs_old, returns, advantages, writer, step_idx)

    def learn(self, seq_batch, actions, log_probs_old, returns, advantages, writer=None, step_idx=0):
        dataset_size = seq_batch.size(0)

        for _ in range(NUM_EPOCHS):
            perm = torch.randperm(dataset_size)
            for i in range(0, dataset_size, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]

                seqs = seq_batch[idx]
                a = actions[idx]
                lp_old = log_probs_old[idx]
                r = returns[idx]
                adv = advantages[idx]

                # === Grad-CAPS requires gradients w.r.t. state ===
                seqs.requires_grad_(True)
                dist, _ = self.actor(seqs)
                sampled_action = dist.rsample()  # Differentiable sampling

                log_probs = dist.log_prob(a).sum(dim=1, keepdim=True)
                entropy = dist.entropy().mean()

                values, _ = self.critic(seqs)

                ratio = torch.exp(log_probs - lp_old)
                obj = ratio * adv
                obj_clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv

                policy_loss = -torch.min(obj, obj_clipped).mean() - self.BETA * entropy
                value_loss = (r - values).pow(2).mean()

                total_loss = policy_loss + 0.5 * value_loss

                if self.use_caps:
                    grads = torch.autograd.grad(sampled_action, seqs,
                                                grad_outputs=torch.ones_like(sampled_action),
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                    grad_caps_loss = grads.pow(2).mean()
                    total_loss += self.lambda_grad_caps * grad_caps_loss

                    if writer:
                        writer.add_scalar("Loss/GradCAPS", grad_caps_loss.item(), step_idx)

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), GRADIENT_CLIP)
                nn.utils.clip_grad_norm_(self.critic.parameters(), GRADIENT_CLIP)
                self.actor_optim.step()
                self.critic_optim.step()

                if writer:
                    writer.add_scalar("Loss/Policy", policy_loss.item(), step_idx)
                    writer.add_scalar("Loss/Value", value_loss.item(), step_idx)
                    writer.add_scalar("Loss/Entropy", entropy.item(), step_idx)
                    writer.add_scalar("actions/std", a.std().item(), step_idx)

        torch.cuda.empty_cache()
