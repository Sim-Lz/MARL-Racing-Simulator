"""Reward per frame, scomposta per termini così che ogni componente sia tracciabile nei log.

Principi:
- il segnale principale è il progresso lungo la mezzeria (potential-based): non premia la velocità
  in sé né una specifica azione di controllo, e tornare indietro costa esattamente quanto si era guadagnato;
- ogni terminazione anticipata (crash, stallo, contromano, nessun progresso) ha la stessa penalità,
  così nessuna modalità di uscita è "più conveniente" delle altre;
- le collisioni tra auto penalizzano solo chi tampona;
- nessuna penalità temporale di default (con la terminazione creerebbe un incentivo a uscire presto).
"""
from dataclasses import dataclass

from config import (
    COLLISION_PENALTY,
    FINISH_BONUS,
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
    finish_bonus: float = 0.0

    @property
    def total(self):
        return (
            self.progress
            - self.time_penalty
            - self.collision_penalty
            - self.termination_penalty
            + self.finish_bonus
        )


def calculate_step_reward(
    progress_checkpoints,
    elapsed_ms,
    terminated=False,
    collisions_at_fault=0,
    finished=False,
):
    """progress_checkpoints: avanzamento del frame espresso in checkpoint (può essere negativo)."""
    return RewardTerms(
        progress=PROGRESS_REWARD_SCALE * progress_checkpoints,
        time_penalty=TIME_PENALTY_PER_SECOND * elapsed_ms / 1000.0,
        collision_penalty=COLLISION_PENALTY * collisions_at_fault,
        termination_penalty=TERMINATION_PENALTY if terminated else 0.0,
        finish_bonus=FINISH_BONUS if finished else 0.0,
    )
