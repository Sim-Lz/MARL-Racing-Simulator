"""Reward per frame, scomposta per termini così che ogni componente sia tracciabile nei log.

Principi:
- il segnale principale è il progresso lungo la mezzeria (potential-based): non premia la velocità
  in sé né una specifica azione di controllo, e tornare indietro costa esattamente quanto si era guadagnato;
- ogni terminazione anticipata (crash, stallo, contromano, nessun progresso) ha la stessa penalità,
  così nessuna modalità di uscita è "più conveniente" delle altre;
- le collisioni tra auto penalizzano solo chi tampona, in proporzione alla velocità d'impatto;
- la penalità di squalifica (stallo, nessun progresso) si paga a rate mentre la situazione dura e alla
  squalifica solo il resto: stesso costo totale, ma visibile subito;
- nessuna penalità temporale di default (con la terminazione creerebbe un incentivo a uscire presto);
- un piccolo bonus quando l'auto migliora il proprio record di settore (solo positivo e con tetto).
"""
from dataclasses import dataclass

from config import (
    COLLISION_PENALTY,
    FINISH_BONUS,
    PERSONAL_BEST_REWARD_CAP,
    PERSONAL_BEST_REWARD_SCALE,
    PROGRESS_REWARD_SCALE,
    TERMINATION_PENALTY,
    TIME_PENALTY_PER_SECOND,
)


@dataclass
class RewardTerms:
    progress: float = 0.0
    time_penalty: float = 0.0
    collision_penalty: float = 0.0
    termination_penalty: float = 0.0
    idle_penalty: float = 0.0
    finish_bonus: float = 0.0
    personal_best_bonus: float = 0.0

    @property
    def total(self):
        return (
            self.progress
            - self.time_penalty
            - self.collision_penalty
            - self.termination_penalty
            - self.idle_penalty
            + self.finish_bonus
            + self.personal_best_bonus
        )


def calculate_step_reward(
    progress_checkpoints,
    elapsed_ms,
    terminated=False,
    collisions_at_fault=0,
    finished=False,
    personal_best_improvements=(),
    idle_penalty=0.0,
    termination_paid=0.0,
):
    """progress_checkpoints: avanzamento del frame espresso in checkpoint (può essere negativo).
    personal_best_improvements: miglioramenti relativi (0-1) dei record personali di settore battuti nel frame.
    collisions_at_fault: somma delle gravità (0-1, velocità d'impatto) dei tamponamenti causati nel frame.
    idle_penalty: quota anticipata della penalità di squalifica pagata nel frame.
    termination_paid: parte della penalità di terminazione già pagata come anticipo."""
    return RewardTerms(
        progress=PROGRESS_REWARD_SCALE * progress_checkpoints,
        time_penalty=TIME_PENALTY_PER_SECOND * elapsed_ms / 1000.0,
        collision_penalty=COLLISION_PENALTY * collisions_at_fault,
        termination_penalty=max(0.0, TERMINATION_PENALTY - termination_paid) if terminated else 0.0,
        idle_penalty=idle_penalty,
        finish_bonus=FINISH_BONUS if finished else 0.0,
        personal_best_bonus=sum(
            min(PERSONAL_BEST_REWARD_CAP, PERSONAL_BEST_REWARD_SCALE * max(0.0, improvement))
            for improvement in personal_best_improvements
        ),
    )
