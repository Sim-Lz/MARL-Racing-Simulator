# rl_models/qlearning.py
import numpy as np
import random

class TabularQLearner:
    def __init__(self):
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1

        self.last_grad_norm = 0.0  # Mappa l'errore di differenza temporale (TD Error)

    def discretize(self, state_vector):
        # Accorpiamo i 7 sensori in 3 macro-aree fondamentali
        left = min(state_vector[0], state_vector[1], state_vector[2])
        center = state_vector[3]
        right = min(state_vector[4], state_vector[5], state_vector[6])
        
        # Discretizzazione a 4 livelli (da critico a libero)
        def get_bin(val):
            if val < 50.0:  return 3  # Pericolo Imminente
            if val < 100.0: return 2  # Vicino
            if val < 165.0: return 1  # Distanza Media
            return 0                  # Rettilineo Libero
            
        return (get_bin(left), get_bin(center), get_bin(right))

    def select_action(self, disc_state):
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(9)
        
        if random.random() < self.epsilon:
            return random.randint(0, 8)
        return int(np.argmax(self.q_table[disc_state]))

    def update(self, s, action, reward, s_prime):
        if s not in self.q_table:
            self.q_table[s] = np.zeros(9)
        if s_prime not in self.q_table:
            self.q_table[s_prime] = np.zeros(9)
            
        max_next_q = np.max(self.q_table[s_prime])
        old_q = self.q_table[s][action]
        
        td_error = reward + self.gamma * max_next_q - old_q
        self.q_table[s][action] += self.lr * td_error
        
        # Il TD Error rappresenta l'equivalente del gradiente per i metodi tabulari
        self.last_grad_norm = float(abs(td_error))