# rl_models/dqn.py
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import NUM_ACTIONS, OBSERVATION_SIZE
from rl_models.base import Agent, exploration_action


class DQNNetwork(nn.Module):
    """Architettura dueling: V(s) + A(s, a) - media(A). Stima del valore più stabile quando molte azioni
    hanno effetti simili (frequente qui: a 20 Hz una singola decisione cambia poco)."""

    def __init__(self):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, NUM_ACTIONS)

    def forward(self, x):
        features = self.fc(x)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=-1, keepdim=True)


class DQNAgent(Agent):
    """Double DQN dueling con replay condiviso tra le auto del team, warm-up, target soft e n-step return.

    Nelle 50 corse precedenti DQN imparava in fretta (vittorie alle corse 21-22) e poi collassava
    (da 250 a 4-20 checkpoint) con epsilon già al minimo: tipica instabilità da target che si muove
    troppo in fretta. Ora learning rate dimezzato, target più lento (tau 0.005), batch più grande e
    ripresa dell'esplorazione quando cambia il tracciato.
    """

    algorithm = "DQN"
    uses_training_respawn = True

    def __init__(self):
        self.model = DQNNetwork()
        self.target_model = DQNNetwork()
        self.target_model.load_state_dict(self.model.state_dict())

        self.learning_rate = 0.00025
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.memory = deque(maxlen=100000)

        self.epsilon_start = 1.0
        self.min_epsilon = 0.03
        self.new_track_epsilon = 0.3
        self.epsilon_decay_steps = 15000   # costante di tempo (decisioni) del decadimento esponenziale
        self.epsilon = self.epsilon_start
        self._schedule_start = self.epsilon_start
        self._schedule_decisions = 0
        self.gamma = 0.98                    # per decisione (3 frame): orizzonte ~2.5 s
        self.n_step = 3
        self.minimum_replay_size = 1000
        self.batch_size = 128
        self.training_interval = 1           # un passo di ottimizzazione per transizione
        self.tau = 0.005
        self.max_grad_norm = 10.0
        self.decisions = 0
        self.steps_counter = 0
        self.training_calls = 0
        self.last_grad_norm = 0.0
        self.last_loss = 0.0
        self._pending = {}

    def begin_track(self, sim):
        # Nuovo tracciato: riparte un breve ciclo di esplorazione (senza scendere sotto quella attuale).
        if self.epsilon < self.new_track_epsilon:
            self._schedule_start = self.new_track_epsilon
            self._schedule_decisions = 0

    def act(self, car_id, observation):
        return self.select_action(observation)

    def select_action(self, state_vector):
        self.decisions += 1
        self._schedule_decisions += 1
        self.epsilon = max(
            self.min_epsilon,
            self._schedule_start * math.exp(-self._schedule_decisions / self.epsilon_decay_steps),
        )
        if random.random() < self.epsilon:
            return exploration_action()
        state_t = torch.as_tensor(np.asarray(state_vector, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.model(state_t)).item())

    def observe(self, car_id, observation, action, reward, next_observation, terminated, truncated):
        """Accumula n-step return per auto; un troncamento chiude la finestra ma resta bootstrappato."""
        window = self._pending.setdefault(car_id, deque())
        window.append((np.asarray(observation, dtype=np.float32), action, float(reward)))
        next_state = np.asarray(next_observation, dtype=np.float32)
        if len(window) >= self.n_step:
            self._emit(window, next_state, terminated)
        if terminated or truncated:
            while window:
                self._emit(window, next_state, terminated)
            del self._pending[car_id]
        self.train_step()

    def _emit(self, window, next_state, terminated):
        state, action, _ = window[0]
        n_return = sum((self.gamma ** k) * item[2] for k, item in enumerate(window))
        self.store_transition(state, action, n_return, next_state, terminated, discount=self.gamma ** len(window))
        window.popleft()

    def store_transition(self, s, a, r, s_prime, done, discount=None):
        state = np.asarray(s, dtype=np.float32)
        next_state = np.asarray(s_prime, dtype=np.float32)
        if state.shape != (OBSERVATION_SIZE,) or next_state.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"Expected observations of shape ({OBSERVATION_SIZE},)")
        self.memory.append((state, int(a), float(r), next_state, float(done), self.gamma if discount is None else discount))

    def train_step(self, batch_size=None):
        if len(self.memory) < self.minimum_replay_size:
            return

        self.training_calls += 1
        if self.training_calls % self.training_interval != 0:
            return

        requested_batch_size = self.batch_size if batch_size is None else batch_size
        batch = random.sample(self.memory, min(requested_batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones, discounts = zip(*batch)

        states_t = torch.as_tensor(np.array(states))
        actions_t = torch.as_tensor(actions, dtype=torch.long).view(-1, 1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).view(-1, 1)
        next_states_t = torch.as_tensor(np.array(next_states))
        dones_t = torch.as_tensor(dones, dtype=torch.float32).view(-1, 1)
        discounts_t = torch.as_tensor(discounts, dtype=torch.float32).view(-1, 1)

        current_q = self.model(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = self.model(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_model(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1.0 - dones_t) * discounts_t * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.last_grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm))
        self.optimizer.step()
        self.last_loss = float(loss.item())
        self.steps_counter += 1

        with torch.no_grad():
            for target, source in zip(self.target_model.parameters(), self.model.parameters()):
                target.mul_(1.0 - self.tau).add_(source, alpha=self.tau)

    def diagnostics(self):
        return {
            "epsilon": float(self.epsilon),
            "replay_size": len(self.memory),
            "optimizer_steps": self.steps_counter,
            "last_grad_norm": self.last_grad_norm,
            "last_loss": self.last_loss,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "n_step": self.n_step,
        }
