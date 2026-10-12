from config import FINISH_BONUS, PROGRESS_REWARD_SCALE, TIME_PENALTY_PER_SECOND


def calculate_step_reward(progress_delta, elapsed_ms, finished=False):
    reward = (
        PROGRESS_REWARD_SCALE * progress_delta
        - TIME_PENALTY_PER_SECOND * elapsed_ms / 1000.0
    )
    if finished:
        reward += FINISH_BONUS
    return reward
