# rl_models/qlearning.py
import math
import random
from collections import deque

import numpy as np

from config import (
    NUM_ACTIONS,
    OBS_HEADING_COS,
    OBS_HEADING_SIN,
    OBS_LATERAL,
    OBS_SPEED,
    OBS_TRACK_HEADING,
    OBS_TRAFFIC_CLOSING,
    OBS_TRAFFIC_PROXIMITY,
    SENSOR_ANGLES,
    SENSOR_RANGES,
)
from rl_models.base import Agent, exploration_action

LATERAL_EDGES = [-0.55, -0.2, 0.2, 0.55]                                 # 5 bin
HEADING_EDGES = [math.radians(d) for d in (-30, -12, -4, 4, 12, 30)]     # 7 bin
NEAR_CURVE_EDGES = [-0.25, -0.08, 0.08, 0.25]                            # 5 bin, curva a 50 px (con segno)
FAR_SHARPNESS_EDGES = [0.15, 0.5]                                        # 3 bin, |curva| a 300 px
SPEED_EDGES = [0.2, 0.4, 0.6, 0.8]                                       # 5 bin
FRONT_WALL_EDGES_PX = [40.0, 120.0]                                      # 3 bin, muro davanti (cono ±10°)
FRONT_CONE = [i for i, angle in enumerate(SENSOR_ANGLES) if abs(angle) <= 10]


class TabularQLearner(Agent):
    """Q-learning tabellare con ritorni a n passi.

    Stato: offset laterale, errore di direzione, curva imminente (con verso), severità della curva
    lontana (per anticipare la frenata), velocità, distanza dal muro davanti e traffico (~15000 stati al
    massimo, ~1500 visitati in pratica). Senza il muro davanti lo stato non conteneva alcun sensore e il
    64% delle auto usciva nei primi 8 s, con errori di direzione mediani di 37°.
    Con l'aggiornamento a 1 passo la penalità di uscita risaliva di uno stato ogni volta che veniva
    rivisitato: nelle 50 corse precedenti la tabella non ha mai imparato a frenare prima delle curve
    (5-16 checkpoint per corsa). Con n=3 il segnale copre ~0.15 s per aggiornamento.
    """

    algorithm = "Q_LEARNING"
    uses_training_respawn = True

    def __init__(self):
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.98            # per decisione (3 frame): orizzonte ~2.5 s
        self.n_step = 3
        self.epsilon = 0.4
        self.epsilon_decay = 0.99995  # per aggiornamento
        self.min_epsilon = 0.03
        self.new_track_epsilon = 0.15
        self.last_td_error = 0.0
        self.updates = 0
        self._windows = {}

    def discretize(self, state_vector):
        heading = math.atan2(float(state_vector[OBS_HEADING_SIN]), float(state_vector[OBS_HEADING_COS]))
        traffic = int(float(state_vector[OBS_TRAFFIC_PROXIMITY]) > 0.5 and float(state_vector[OBS_TRAFFIC_CLOSING]) > 0.0)
        return (
            int(np.searchsorted(LATERAL_EDGES, float(state_vector[OBS_LATERAL]))),
            int(np.searchsorted(HEADING_EDGES, heading)),
            int(np.searchsorted(NEAR_CURVE_EDGES, float(state_vector[OBS_TRACK_HEADING]))),
            int(np.searchsorted(FAR_SHARPNESS_EDGES, abs(float(state_vector[OBS_TRACK_HEADING + 2])))),
            int(np.searchsorted(SPEED_EDGES, float(state_vector[OBS_SPEED]))),
            int(np.searchsorted(FRONT_WALL_EDGES_PX, min(float(state_vector[i]) * SENSOR_RANGES[i] for i in FRONT_CONE))),
            traffic,
        )

    def _values(self, state):
        if state not in self.q_table:
            # Uno stato con traffico mai visto parte dai valori dello stesso stato senza traffico: con zeri
            # l'argmax era un pareggio tra tutte le azioni e in gruppo (partenza) l'auto guidava a caso.
            free = state[:-1] + (0,)
            self.q_table[state] = self.q_table[free].copy() if state[-1] and free in self.q_table else np.zeros(NUM_ACTIONS)
        return self.q_table[state]

    def begin_track(self, sim):
        # Su un tracciato nuovo serve di nuovo un po' di esplorazione.
        self.epsilon = max(self.epsilon, self.new_track_epsilon)

    def act(self, car_id, observation):
        return self.select_action(self.discretize(observation))

    def select_action(self, disc_state):
        values = self._values(disc_state)
        if random.random() < self.epsilon:
            return exploration_action()
        best_actions = np.flatnonzero(np.isclose(values, np.max(values)))
        return int(random.choice(best_actions.tolist()))

    def observe(self, car_id, observation, action, reward, next_observation, terminated, truncated):
        window = self._windows.setdefault(car_id, deque())
        window.append((self.discretize(observation), int(action), float(reward)))
        next_state = self.discretize(next_observation)
        if len(window) >= self.n_step:
            self._emit(window, next_state, terminated)
        if terminated or truncated:
            while window:
                self._emit(window, next_state, terminated)
            del self._windows[car_id]

    def _emit(self, window, next_state, terminated):
        state, action, _ = window[0]
        n_return = sum((self.gamma ** k) * item[2] for k, item in enumerate(window))
        self.update(state, action, n_return, next_state, done=terminated, discount=self.gamma ** len(window))
        window.popleft()

    def update(self, s, action, reward, s_prime, done=False, discount=None):
        """done deve essere True solo per terminazioni reali: un troncamento va bootstrappato."""
        values = self._values(s)
        discount = self.gamma if discount is None else discount
        next_value = 0.0 if done else discount * float(np.max(self._values(s_prime)))
        td_error = reward + next_value - values[action]
        values[action] += self.lr * td_error
        self.last_td_error = float(td_error)
        self.updates += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def diagnostics(self):
        return {
            "epsilon": float(self.epsilon),
            "q_table_states": len(self.q_table),
            "updates": self.updates,
            "last_td_error": self.last_td_error,
            "lr": self.lr,
            "gamma": self.gamma,
            "n_step": self.n_step,
        }
