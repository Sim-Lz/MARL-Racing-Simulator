# rl_models/genetic.py
import numpy as np

class GeneticNetwork:
    def __init__(self):
        self.W1 = np.random.randn(7, 16) * 0.3
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(16, 3) * 0.3
        self.b2 = np.zeros(3)

    def forward(self, state_vector):
        x = np.array(state_vector) / 250.0 
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return int(np.argmax(out) - 1)

    def mutate(self, rate=0.1):
        # Mutazione selettiva applicata sia a pesi che a bias
        self.W1 += np.random.randn(*self.W1.shape) * rate * (np.random.rand(*self.W1.shape) < 0.25)
        self.b1 += np.random.randn(*self.b1.shape) * rate * (np.random.rand(*self.b1.shape) < 0.25)
        self.W2 += np.random.randn(*self.W2.shape) * rate * (np.random.rand(*self.W2.shape) < 0.25)
        self.b2 += np.random.randn(*self.b2.shape) * rate * (np.random.rand(*self.b2.shape) < 0.25)