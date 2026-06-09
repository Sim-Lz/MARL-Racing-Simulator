# rl_models/genetic.py
import numpy as np

class GeneticNetwork:
    def __init__(self):
        self.W1 = np.random.randn(7, 16) * 0.2
        self.W2 = np.random.randn(16, 3) * 0.2

    def forward(self, state_vector):
        # Standardizzazione interna dell'input numerico
        x = np.array(state_vector) / 250.0 
        h = np.tanh(np.dot(x, self.W1))
        out = np.dot(h, self.W2)
        return int(np.argmax(out) - 1)

    def mutate(self, rate=0.07):
        self.W1 += np.random.randn(*self.W1.shape) * rate * (np.random.rand(*self.W1.shape) < 0.2)
        self.W2 += np.random.randn(*self.W2.shape) * rate * (np.random.rand(*self.W2.shape) < 0.2)