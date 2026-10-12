# rl_models/qlearning.py
import numpy as np
import random

class TabularQLearner:
    def __init__(self):
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.999
        self.epsilon = 0.1
        self.epsilon_decay = 0.9995
        self.min_epsilon = 0.02

        self.last_td_error = 0.0

    def discretize(self, state_vector):
        left = min(state_vector[0], state_vector[1], state_vector[2])
        center = state_vector[3]
        right = min(state_vector[4], state_vector[5], state_vector[6])

        def get_bin(val):
            if val < 0.125:
                return 3
            if val < 0.25:
                return 2
            if val < 0.4125:
                return 1
            return 0

        speed_bin = min(3, int(float(state_vector[7]) * 4.0))
        bearing = np.arctan2(state_vector[8], state_vector[9])
        bearing_bin = min(7, int((bearing + np.pi) / (np.pi / 4.0)))
        return (get_bin(left), get_bin(center), get_bin(right), speed_bin, bearing_bin)

    def select_action(self, disc_state):
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(9)
        
        if random.random() < self.epsilon:
            return random.randint(0, 8)
        values = self.q_table[disc_state]
        best_actions = np.flatnonzero(np.isclose(values, np.max(values)))
        return int(random.choice(best_actions.tolist()))

    def update(self, s, action, reward, s_prime, done=False):
        if s not in self.q_table:
            self.q_table[s] = np.zeros(9)
        if s_prime not in self.q_table:
            self.q_table[s_prime] = np.zeros(9)
            
        old_q = self.q_table[s][action]
        next_value = 0.0 if done else self.gamma * np.max(self.q_table[s_prime])
        td_error = reward + next_value - old_q
        self.q_table[s][action] += self.lr * td_error
        self.last_td_error = float(td_error)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)