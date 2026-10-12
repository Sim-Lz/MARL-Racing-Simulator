# rl_models/genetic.py
import numpy as np
from config import OBSERVATION_SIZE

class GeneticNetwork:
    def __init__(self):
        self.W1 = np.random.randn(OBSERVATION_SIZE, 16) * 0.3
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(16, 9) * 0.3
        self.b2 = np.zeros(9)
        self.last_mutation_norm = 0.0
        self.mutation_rate = 0.05
        self.mutation_fraction = 0.15

    def forward(self, state_vector):
        x = np.asarray(state_vector, dtype=np.float32)
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return int(np.argmax(out))

    def mutate(self, rate=None):
        mutation_rate = self.mutation_rate if rate is None else rate
        dW1 = np.random.randn(*self.W1.shape) * mutation_rate * (np.random.rand(*self.W1.shape) < self.mutation_fraction)
        db1 = np.random.randn(*self.b1.shape) * mutation_rate * (np.random.rand(*self.b1.shape) < self.mutation_fraction)
        dW2 = np.random.randn(*self.W2.shape) * mutation_rate * (np.random.rand(*self.W2.shape) < self.mutation_fraction)
        db2 = np.random.randn(*self.b2.shape) * mutation_rate * (np.random.rand(*self.b2.shape) < self.mutation_fraction)
        
        self.W1 += dW1
        self.b1 += db1
        self.W2 += dW2
        self.b2 += db2
        
        total_mutation = np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2)
        self.last_mutation_norm = float(np.sqrt(total_mutation))