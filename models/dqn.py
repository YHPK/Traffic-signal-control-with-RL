import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions, device):
        super(Policy, self).__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device

        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x

    def get_action(self, obs, epsilon=0):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        if random.random() > epsilon:
            with torch.no_grad():
                q = self.forward(obs)
                action = torch.argmax(q).item()
        else:
            action = random.randrange(self.n_actions)

        return action

    def get_output(self, obs, need_grad=True):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        if need_grad:
            output = self.forward(obs)
        else:
            with torch.no_grad():
                output = self.forward(obs)

        return output.cpu()


def train(q_model, target_q_model, batch, optim, gamma, device):
    obs = batch["obs"].to(device)
    next_obs = batch["obs2"].to(device)
    act = torch.tensor(batch["act"], dtype=torch.int64).to(device)
    rew = batch["rew"].to(device)
    done = batch["done"].to(device)

    obs_output = q_model(obs)
    q = torch.gather(obs_output, 1, act).squeeze(1)

    with torch.no_grad():
        next_obs_output = target_q_model(next_obs)
        next_q = torch.max(next_obs_output, dim=1)[0]

    td_target = rew + gamma * next_q * (1-done)
    td_error = (q - td_target).pow(2).mean()

    optim.zero_grad()
    td_error.backward()
    optim.step()

    return td_error.item()