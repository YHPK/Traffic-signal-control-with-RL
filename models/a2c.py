import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions, device):
        super(Policy, self).__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device

        self.actor = Actor(obs_dim, n_actions)
        self.critic = Critic(obs_dim)


    def forward(self, x):
        probs = self.actor(x)
        value = self.critic(x)

        return probs, value

    def get_action_logprob_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        probs, value = self.forward(obs)
        categorical = Categorical(probs)
        action = categorical.sample()
        logprob = categorical.log_prob(action)

        return action.item(), logprob, value


def train(logprobs, returns, values, optim):
    optim.zero_grad()
    loss = 0
    # Cumulate gradients
    for ret, logprob, val in zip(returns, logprobs, values):
        adv = ret  - val.item()
        act_loss = -1 * logprob * adv
        val_loss = F.smooth_l1_loss(val, ret)

        loss += (act_loss + val_loss)

    loss.backward()
    optim.step()