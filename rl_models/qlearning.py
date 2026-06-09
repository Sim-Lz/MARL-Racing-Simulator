# rl_models/qlearning.py
import numpy as np
import random

class TabularQLearner:
    def __init__(self):
        # Spazio degli stati discretizzato: 2 macro-stati (Vicino/Lontano) per 7 sensori = 128 combinazioni
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.90
        self.epsilon = 0.1

    def discretize(self, state_vector):
        # Se la distanza letta è minore di 60px restituisce 1 (Pericolo), altrimenti 0 (Libero)
        return tuple([1 if val < 60.0 else 0 for val in state_vector])

    def select_action(self, disc_state):
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(3) # 3 azioni possibili
        
        if random.random() < self.epsilon:
            return random.randint(0, 2) - 1
        return int(np.argmax(self.q_table[disc_state]) - 1)

    def update(self, s, action, reward, s_prime):
        a_idx = action + 1
        if s not in self.q_table: self.q_table[s] = np.zeros(3)
        if s_prime not in self.q_table: self.q_table[s_prime] = np.zeros(3)
        
        best_next_q = np.max(self.q_table[s_prime])
        self.q_table[s][a_idx] += self.lr * (reward + self.gamma * best_next_q - self.q_table[s][a_idx])