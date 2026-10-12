# rl_models/policy_gradient.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from config import OBSERVATION_SIZE

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 9),
        )

    def forward(self, x):
        return self.net(x)

class PolicyGradientAgent:
    def __init__(self):
        self.policy = PolicyNetwork()
        self.learning_rate = 0.0003
        self.gamma = 0.999
        self.entropy_coefficient = 0.01
        self.max_grad_norm = 1.0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.saved_log_probs = []
        self.saved_entropies = []
        self.rewards = []
        self.last_grad_norm = 0.0

    def select_action(self, state_vector):
        state = np.asarray(state_vector, dtype=np.float32)
        if state.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"Expected an observation of shape ({OBSERVATION_SIZE},)")
        state_t = torch.as_tensor(state)
        m = Categorical(logits=self.policy(state_t))
        action_idx = m.sample()
        self.saved_log_probs.append(m.log_prob(action_idx))
        self.saved_entropies.append(m.entropy())
        return int(action_idx.item())

    def update_policy(self):
        if not self.rewards:
            if self.saved_log_probs or self.saved_entropies:
                raise ValueError("Policy-gradient rewards and sampled actions are not aligned")
            return

        if not (len(self.rewards) == len(self.saved_log_probs) == len(self.saved_entropies)):
            raise ValueError("Policy-gradient rewards and sampled actions are not aligned")

        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns_t = torch.tensor(returns, dtype=torch.float32)
        advantages = returns_t - returns_t.mean()
        if len(advantages) > 1 and advantages.std(unbiased=False) > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
            
        self.optimizer.zero_grad()
        log_probs = torch.stack(self.saved_log_probs)
        entropies = torch.stack(self.saved_entropies)
        loss = -(log_probs * advantages).mean() - self.entropy_coefficient * entropies.mean()
        loss.backward()
        
        # Calcolo della norma L2 del gradiente per REINFORCE
        total_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.last_grad_norm = total_norm ** 0.5
        
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.saved_log_probs.clear()
        self.saved_entropies.clear()
        self.rewards.clear()