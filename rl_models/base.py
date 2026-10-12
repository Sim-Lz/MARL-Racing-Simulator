"""Interfaccia comune a tutti i controllori.

Ogni agente è condiviso dalle auto dello stesso team (parameter sharing): le transizioni arrivano
etichettate con car_id, così i modelli che ne hanno bisogno (PPO, genetico) tengono buffer separati.
Una transizione copre un intervallo decisionale (l'azione viene ripetuta per N frame e le reward sommate).
"""
import random

from car import controls_to_action

# Prior di esplorazione sul pedale: con azioni uniformi frenata (-260 px/s^2) e accelerazione (+180)
# si annullano e l'auto resta quasi ferma, quindi l'esplorazione casuale non vede mai curve né progresso.
# Riguarda solo la raccolta dati: reward e valori appresi non vengono toccati.
EXPLORATION_LONGITUDINAL_WEIGHTS = {1: 0.5, 0: 0.25, -1: 0.25}


def exploration_action():
    steer = random.choice((-1, 0, 1))
    longitudinal = random.choices(
        list(EXPLORATION_LONGITUDINAL_WEIGHTS), weights=list(EXPLORATION_LONGITUDINAL_WEIGHTS.values())
    )[0]
    return controls_to_action(steer, longitudinal)


class Agent:
    algorithm = "BASE"
    # True per i modelli che si addestrano anche sui respawn "fantasma" dopo un'eliminazione.
    uses_training_respawn = False
    # Durata massima di un episodio fantasma (None = fino all'uscita di pista o a fine corsa).
    max_training_episode_s = None
    # True se il fantasma deve girare per tutta la corsa, in parallelo all'auto in gara (es. genetico,
    # che valuta i candidati nei fantasmi mentre in gara corre il miglior individuo).
    continuous_training_ghost = False
    # Numero di fantasmi contemporanei per auto (episodi indipendenti, utile per valutazioni parallele).
    training_ghosts_per_car = 1

    def begin_track(self, sim):
        """Chiamato quando inizia una serie di corse su un nuovo tracciato."""

    def training_spawn_index(self, slot, default_index, track):
        """Indice della mezzeria da cui far partire un episodio fantasma (default: poco prima dell'uscita)."""
        return default_index

    def begin_race(self, car_ids):
        """Chiamato all'inizio di ogni corsa con le auto controllate da questo agente."""

    def begin_episode(self, car_id):
        """Chiamato quando parte un episodio di addestramento fantasma per car_id."""

    def end_episode(self, car_id, result):
        """Fine di un episodio fantasma. result: {"fitness": reward dell'episodio, "progress": checkpoint}."""

    def act(self, car_id, observation):
        raise NotImplementedError

    def observe(self, car_id, observation, action, reward, next_observation, terminated, truncated):
        """terminated: fine reale dell'episodio (crash/DSQ). truncated: corsa chiusa per timeout o vittoria
        altrui, lo stato successivo ha ancora valore e va usato per il bootstrap."""

    def end_race(self, results):
        """results: {car_id: {"progress": checkpoint percorsi, "fitness": reward individuale, ...}}."""

    def diagnostics(self):
        return {}
