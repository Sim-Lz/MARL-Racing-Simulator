# rl_models/dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self):
        self.model = DQNNetwork()
        self.target_model = DQNNetwork()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        
        self.epsilon = 0.30        # Partiamo più alti per esplorare la pista
        self.epsilon_decay = 0.9997 # Decadimento progressivo frame-by-frame
        self.min_epsilon = 0.02
        self.gamma = 0.98
        self.steps_counter = 0

    def select_action(self, state_vector):
        if random.random() < self.epsilon:
            return random.randint(0, 2) - 1
        with torch.no_grad():
            # Normalizzazione dell'input numerico in tempo reale
            norm_state = np.array(state_vector) / 250.0
            state_t = torch.FloatTensor(norm_state).unsqueeze(0)
            q_values = self.model(state_t)
            return int(torch.argmax(q_values).item()) - 1

    def store_transition(self, s, a, r, s_prime, done):
        # Normalizziamo gli stati prima di salvarli nel replay buffer
        norm_s = np.array(s) / 250.0
        norm_s_prime = np.array(s_prime) / 250.0
        self.memory.append((norm_s, a + 1, r, norm_s_prime, done))

    def train_step(self, batch_size=64):
        self.steps_counter += 1
        if len(self.memory) < batch_size:
            return
            
        # Decadimento dell'esplorazione
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            
        # Sincronizzazione della Target Network ogni 400 step di ottimizzazione
        if self.steps_counter % 400 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        if self.steps_counter % 4 != 0:
            return
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).view(-1, 1)
        rewards_t = torch.FloatTensor(rewards).view(-1, 1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).view(-1, 1)

        current_q = self.model(states_t).gather(1, actions_t)
        
        # Calcolo del target usando la Target Network fissa
        with torch.no_grad():
            max_next_q = self.target_model(next_states_t).max(1)[0].view(-1, 1)
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()