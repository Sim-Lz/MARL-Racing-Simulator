# rl_models/policy_gradient.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class PolicyGradientAgent:
    def __init__(self):
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.002)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state_vector):
        state_t = torch.FloatTensor(state_vector)
        probs = self.policy(state_t)
        m = Categorical(probs)
        action_idx = m.sample()
        self.saved_log_probs.append(m.log_prob(action_idx))
        return int(action_idx.item() - 1)

    def update_policy(self):
        if not self.rewards:
            return
            
        # Calcolo dei totali dei ritorni scontati (REINFORCE)
        R = 0
        policy_loss = []
        returns = []
        for r in reversed(self.rewards):
            R = r + 0.98 * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
            
        for log_prob, Gt in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * Gt)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        self.saved_log_probs.clear()
        self.rewards.clear()