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
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 9),
            nn.Softmax(dim=-1)
        )
        self.last_grad_norm = 0.0  # Per monitorare la stabilità dell'addestramento
    def forward(self, x):
        return self.net(x)

class PolicyGradientAgent:
    def __init__(self):
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state_vector):
        # Normalizzazione cruciale dell'input per impedire la saturazione della Softmax
        norm_state = np.array(state_vector) / 250.0
        state_t = torch.FloatTensor(norm_state)
        
        probs = self.policy(state_t)
        m = Categorical(probs)
        action_idx = m.sample()
        self.saved_log_probs.append(m.log_prob(action_idx))
        return int(action_idx.item())

    def update_policy(self):
        if not self.rewards:
            return
            
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
        
        # Calcolo della norma L2 del gradiente per REINFORCE
        total_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.last_grad_norm = total_norm ** 0.5
        
        self.optimizer.step()
        
        self.saved_log_probs.clear()
        self.rewards.clear()