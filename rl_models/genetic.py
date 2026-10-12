# rl_models/genetic.py
import numpy as np

from car import controls_to_action
from config import OBS_TRAFFIC_PROXIMITY, OBSERVATION_SIZE
from rl_models.base import Agent

HIDDEN_UNITS = 16
CONTROL_DEADZONE = 0.33


class GeneticNetwork:
    """Rete con due uscite continue (sterzo, acceleratore) mappate sulle 9 azioni con una zona morta.

    Con un argmax su 9 uscite, piccole mutazioni cambiavano l'azione in modo discontinuo; con due uscite
    l'effetto di una mutazione sul comportamento è graduale e il paesaggio di fitness più regolare.
    """

    def __init__(self, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        self.W1 = rng.standard_normal((OBSERVATION_SIZE, HIDDEN_UNITS)) / np.sqrt(OBSERVATION_SIZE)
        self.b1 = np.zeros(HIDDEN_UNITS)
        self.W2 = rng.standard_normal((HIDDEN_UNITS, 2)) / np.sqrt(HIDDEN_UNITS)
        self.b2 = np.zeros(2)
        self.last_mutation_norm = 0.0
        self.mutation_rate = 0.1
        self.mutation_fraction = 0.2

    def controls(self, state_vector):
        x = np.asarray(state_vector, dtype=np.float64)
        h = np.tanh(x @ self.W1 + self.b1)
        steer_value, throttle_value = np.tanh(h @ self.W2 + self.b2)
        steer = 0 if abs(steer_value) < CONTROL_DEADZONE else int(np.sign(steer_value))
        throttle = 0 if abs(throttle_value) < CONTROL_DEADZONE else int(np.sign(throttle_value))
        return steer, throttle

    def forward(self, state_vector):
        return controls_to_action(*self.controls(state_vector))

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def copy(self):
        clone = GeneticNetwork.__new__(GeneticNetwork)
        clone.W1, clone.b1, clone.W2, clone.b2 = (p.copy() for p in self.parameters())
        clone.last_mutation_norm = 0.0
        clone.mutation_rate = self.mutation_rate
        clone.mutation_fraction = self.mutation_fraction
        return clone

    def mutate(self, rate=None, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        mutation_rate = self.mutation_rate if rate is None else rate
        total = 0.0
        for parameter in self.parameters():
            delta = rng.standard_normal(parameter.shape) * mutation_rate * (rng.random(parameter.shape) < self.mutation_fraction)
            parameter += delta
            total += float(np.sum(delta ** 2))
        self.last_mutation_norm = float(np.sqrt(total))

    @staticmethod
    def crossover(parent_a, parent_b, rng):
        """Crossover uniforme per neurone nascosto: mantiene insieme pesi in ingresso e in uscita del neurone."""
        child = parent_a.copy()
        take_b = rng.random(HIDDEN_UNITS) < 0.5
        child.W1[:, take_b] = parent_b.W1[:, take_b]
        child.b1[take_b] = parent_b.b1[take_b]
        child.W2[take_b, :] = parent_b.W2[take_b, :]
        if rng.random() < 0.5:
            child.b2 = parent_b.b2.copy()
        return child


class GeneticAgent(Agent):
    """Algoritmo genetico con popolazione propria del team e valutazione deterministica su benchmark fisso.

    In gara corre sempre il miglior individuo noto. In parallelo, per tutta la corsa, alcuni fantasmi per
    auto valutano i candidati. Ogni individuo viene provato da fermo su benchmark_starts punti di partenza
    fissi per tracciato (distribuiti lungo il circuito), per max_training_episode_s secondi ciascuno;
    la fitness è la somma delle reward delle prove (progresso meno penalità di uscita).
    I fantasmi non interagiscono con le altre auto e reti e fisica sono deterministiche: la fitness è
    esatta e confrontabile tra generazioni, quindi gli elitari la conservano senza essere rivalutati e il
    miglior individuo non può peggiorare.
    (Prima ogni valutazione partiva da un punto diverso: la fitness misurava il punto di partenza e la
    popolazione è rimasta ferma per 355 generazioni.)
    """

    algorithm = "GENETIC"
    uses_training_respawn = True
    continuous_training_ghost = True
    training_ghosts_per_car = 3
    max_training_episode_s = 10.0
    benchmark_starts = 4

    def __init__(self, population_size=16, elite_count=2, tournament_size=3, seed=None):
        self.rng = np.random.default_rng(seed)
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.mutation_rate = 0.1
        self.mutation_fraction = 0.2
        self.population = [self._new_individual() for _ in range(population_size)]
        self.scores = [[None] * self.benchmark_starts for _ in range(population_size)]
        self.best = 0
        self.assignments = {}
        self.evaluations = 0
        self.generation = 0
        self.best_fitness = float("-inf")
        self.last_mutation_norm = 0.0
        self.start_indices = [0] * self.benchmark_starts

    def _new_individual(self):
        network = GeneticNetwork(self.rng)
        # Lieve spinta iniziale sull'acceleratore: senza, buona parte delle reti casuali resta ferma
        # e viene squalificata per stallo, senza produrre alcun segnale di fitness utile.
        network.b2[1] = 0.5
        return network

    def fitness(self, index):
        scores = self.scores[index]
        return float(sum(scores)) if all(score is not None for score in scores) else None

    def begin_track(self, sim):
        # Nuovo tracciato: nuovo benchmark, tutte le fitness vanno ricalcolate (elitari compresi).
        count = len(sim.track.center)
        offset = int(self.rng.integers(count))
        self.start_indices = [(offset + k * count // self.benchmark_starts) % count for k in range(self.benchmark_starts)]
        self.scores = [[None] * self.benchmark_starts for _ in range(self.population_size)]
        self.assignments = {}

    def begin_race(self, car_ids):
        # In gara corre il miglior individuo noto (sfruttamento); i candidati si valutano nei fantasmi.
        self.assignments = {}

    def begin_episode(self, car_id):
        in_flight = {(task[1], task[2]) for task in self.assignments.values() if task[0] == self.generation}
        for index in range(self.population_size):
            for start, score in enumerate(self.scores[index]):
                if score is None and (index, start) not in in_flight:
                    self.assignments[car_id] = (self.generation, index, start)
                    return
        # Tutte le prove sono già in corso: l'episodio guida il migliore e non viene conteggiato.
        self.assignments[car_id] = (self.generation, None, 0)

    def training_spawn_index(self, slot, default_index, track):
        _, _, start = self.assignments.get(slot, (None, None, 0))
        return self.start_indices[start]

    def end_episode(self, car_id, result):
        generation, index, start = self.assignments.pop(car_id, (None, None, None))
        if generation != self.generation or index is None:
            return
        self.scores[index][start] = float(result["fitness"])
        self.evaluations += 1
        if all(self.fitness(i) is not None for i in range(self.population_size)):
            self._evolve()

    def end_race(self, results):
        # Le prove interrotte dalla fine corsa non contano: tornano in coda.
        self.assignments = {}

    def act(self, car_id, observation):
        _, index, _ = self.assignments.get(car_id, (None, None, None))
        # Le valutazioni avvengono senza traffico (fantasmi isolati), quindi i pesi sugli input di traffico
        # non vengono mai selezionati: in gara produrrebbero comportamenti casuali (es. stallo in griglia).
        # Il genetico guida quindi senza quegli input.
        masked = np.array(observation, dtype=np.float64, copy=True)
        masked[OBS_TRAFFIC_PROXIMITY:] = 0.0
        return self.population[self.best if index is None else index].forward(masked)

    def _tournament(self, fitness):
        contenders = self.rng.choice(self.population_size, size=self.tournament_size, replace=False)
        return self.population[max(contenders, key=lambda i: fitness[i])]

    def _evolve(self):
        fitness = [self.fitness(i) for i in range(self.population_size)]
        order = np.argsort(fitness)[::-1]
        self.best_fitness = float(fitness[order[0]])

        new_population = [self.population[i] for i in order[:self.elite_count]]
        new_scores = [list(self.scores[i]) for i in order[:self.elite_count]]   # deterministiche: si conservano
        mutation_norms = []
        while len(new_population) < self.population_size:
            child = GeneticNetwork.crossover(self._tournament(fitness), self._tournament(fitness), self.rng)
            child.mutation_fraction = self.mutation_fraction
            child.mutate(self.mutation_rate, self.rng)
            mutation_norms.append(child.last_mutation_norm)
            new_population.append(child)
            new_scores.append([None] * self.benchmark_starts)

        self.population = new_population
        self.scores = new_scores
        self.best = 0   # l'elitario migliore è in testa alla nuova popolazione
        self.last_mutation_norm = float(np.mean(mutation_norms)) if mutation_norms else 0.0
        self.generation += 1

    def diagnostics(self):
        return {
            "ga_generation": self.generation,
            "evaluations": self.evaluations,
            "population_size": self.population_size,
            "best_fitness": self.best_fitness if np.isfinite(self.best_fitness) else None,
            "last_mutation_norm": self.last_mutation_norm,
            "mutation_rate": self.mutation_rate,
            "mutation_fraction": self.mutation_fraction,
            "benchmark_starts": self.start_indices,
        }
