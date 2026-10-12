# rl_models/dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from config import OBSERVATION_SIZE

class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 9)
        )
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self):
        self.model = DQNNetwork()
        self.target_model = DQNNetwork()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.learning_rate = 0.0005
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.memory = deque(maxlen=10000)
        
        self.epsilon = 0.30
        self.epsilon_decay = 0.9997
        self.min_epsilon = 0.02
        self.gamma = 0.995
        self.minimum_replay_size = 8
        self.batch_size = 32
        self.training_interval = 4
        self.target_update_interval = 50
        self.steps_counter = 0
        self.training_calls = 0
        self.last_grad_norm = 0.0

    def select_action(self, state_vector):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randint(0, 8)
            
        state_t = torch.as_tensor(np.asarray(state_vector, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            actions_values = self.model(state_t)
        return int(torch.argmax(actions_values).item())

    def store_transition(self, s, a, r, s_prime, done):
        state = np.asarray(s, dtype=np.float32)
        next_state = np.asarray(s_prime, dtype=np.float32)
        if state.shape != (OBSERVATION_SIZE,) or next_state.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"Expected observations of shape ({OBSERVATION_SIZE},)")
        self.memory.append((state, a, r, next_state, done))

    def train_step(self, batch_size=None):
        if len(self.memory) < self.minimum_replay_size:
            return

        self.training_calls += 1
        if self.training_calls % self.training_interval != 0:
            return
            
        requested_batch_size = self.batch_size if batch_size is None else batch_size
        batch = random.sample(self.memory, min(requested_batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).view(-1, 1)
        rewards_t = torch.FloatTensor(rewards).view(-1, 1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).view(-1, 1)

        current_q = self.model(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            next_actions = self.model(next_states_t).argmax(1, keepdim=True)
            max_next_q = self.target_model(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        self.last_grad_norm = total_norm ** 0.5
        
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.steps_counter += 1
        if self.steps_counter % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())