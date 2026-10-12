# rl_models/policy_gradient.py
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from config import NUM_ACTIONS, OBSERVATION_SIZE
from rl_models.base import Agent


class PolicyNetwork(nn.Module):
    """Actor e critic separati: il critic fornisce la baseline V(s) per ridurre la varianza."""

    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, NUM_ACTIONS),
        )
        self.value = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Politica iniziale indipendente dallo stato, con lo stesso prior sul pedale dell'esplorazione
        # epsilon-greedy (50% accelera, 25% mantiene, 25% frena; sterzo uniforme): solo punto di partenza.
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([math.log(2.0)] * 3 + [0.0] * 6))

    def forward(self, x):
        return self.net(x)


class PolicyGradientAgent(Agent):
    """Policy gradient actor-critic con surrogato clippato (PPO) e vantaggi GAE.

    La versione precedente (REINFORCE) faceva un solo passo di gradiente a fine corsa con ritorni
    Monte Carlo a gamma 0.999 per frame: ~50 aggiornamenti in tutto, varianza enorme e baseline
    (media dell'episodio) correlata con il tempo. Qui l'aggiornamento avviene ogni rollout_size
    decisioni, con bootstrap del critic anche sui troncamenti di fine corsa.
    """

    algorithm = "POLICY_GRADIENT"
    uses_training_respawn = True

    def __init__(self):
        self.policy = PolicyNetwork()
        self.learning_rate = 0.0003
        self.gamma = 0.98            # per decisione (3 frame)
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_grad_norm = 0.5
        self.rollout_size = 1024
        self.epochs = 8
        self.minibatch_size = 128
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.last_grad_norm = 0.0
        self.last_entropy = 0.0
        self.updates = 0
        self._last_sample = {}
        self._trajectories = {}
        self._stored = 0

    def act(self, car_id, observation):
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"Expected an observation of shape ({OBSERVATION_SIZE},)")
        with torch.no_grad():
            state_t = torch.as_tensor(state)
            distribution = Categorical(logits=self.policy(state_t))
            action = distribution.sample()
            value = self.policy.value(state_t)
        self._last_sample[car_id] = (float(distribution.log_prob(action)), float(value))
        return int(action.item())

    def select_action(self, state_vector, car_id=0):
        return self.act(car_id, state_vector)

    def observe(self, car_id, observation, action, reward, next_observation, terminated, truncated):
        log_prob, value = self._last_sample.pop(car_id)
        with torch.no_grad():
            next_value = 0.0 if terminated else float(self.policy.value(torch.as_tensor(np.asarray(next_observation, dtype=np.float32))))
        self._trajectories.setdefault(car_id, []).append(
            (np.asarray(observation, dtype=np.float32), int(action), log_prob, value, float(reward), next_value, terminated or truncated)
        )
        self._stored += 1
        if self._stored >= self.rollout_size:
            self.update_policy()

    def _advantages(self, trajectory):
        advantages = np.zeros(len(trajectory), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(trajectory))):
            _, _, _, value, reward, next_value, episode_end = trajectory[t]
            delta = reward + self.gamma * next_value - value
            if episode_end or t == len(trajectory) - 1:
                running = 0.0
            running = delta + self.gamma * self.gae_lambda * running
            advantages[t] = running
        returns = advantages + np.asarray([item[3] for item in trajectory], dtype=np.float32)
        return advantages, returns

    def update_policy(self):
        trajectories = [t for t in self._trajectories.values() if t]
        if not trajectories:
            return
        states, actions, old_log_probs, advantages, returns = [], [], [], [], []
        for trajectory in trajectories:
            traj_advantages, traj_returns = self._advantages(trajectory)
            states.extend(item[0] for item in trajectory)
            actions.extend(item[1] for item in trajectory)
            old_log_probs.extend(item[2] for item in trajectory)
            advantages.append(traj_advantages)
            returns.append(traj_returns)

        states_t = torch.as_tensor(np.array(states))
        actions_t = torch.as_tensor(actions, dtype=torch.long)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32)
        advantages_t = torch.as_tensor(np.concatenate(advantages))
        returns_t = torch.as_tensor(np.concatenate(returns))
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        count = len(states_t)
        grad_norms, entropies = [], []
        for _ in range(self.epochs):
            permutation = torch.randperm(count)
            for start in range(0, count, self.minibatch_size):
                batch = permutation[start:start + self.minibatch_size]
                distribution = Categorical(logits=self.policy(states_t[batch]))
                log_probs = distribution.log_prob(actions_t[batch])
                ratio = torch.exp(log_probs - old_log_probs_t[batch])
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                policy_loss = -torch.min(ratio * advantages_t[batch], clipped * advantages_t[batch]).mean()
                value_loss = (self.policy.value(states_t[batch]).squeeze(-1) - returns_t[batch]).pow(2).mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norms.append(float(nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)))
                entropies.append(float(entropy.detach()))
                self.optimizer.step()

        self.last_grad_norm = float(np.mean(grad_norms))
        self.last_entropy = float(np.mean(entropies))
        self.updates += 1
        self._trajectories = {}
        self._stored = 0

    def end_race(self, results):
        # Le traiettorie sono già chiuse (terminated/truncated): restano nel buffer fino al rollout pieno.
        self._last_sample.clear()

    def diagnostics(self):
        return {
            "policy_updates": self.updates,
            "pending_transitions": self._stored,
            "last_grad_norm": self.last_grad_norm,
            "last_entropy": self.last_entropy,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "entropy_coefficient": self.entropy_coefficient,
        }
