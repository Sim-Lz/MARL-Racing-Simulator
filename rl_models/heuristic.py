# rl_models/heuristic.py
from config import (
    DECELERATION,
    FPS,
    MAX_RAY_DISTANCE,
    MAX_SPEED_PX,
)


class HeuristicController:
    def select_action(self, state_vector):
        left_clearance = (state_vector[0] + state_vector[1] + state_vector[2]) / 3.0
        right_clearance = (state_vector[4] + state_vector[5] + state_vector[6]) / 3.0
        speed = float(state_vector[7]) * MAX_SPEED_PX * 1.15
        bearing_sin = float(state_vector[8])
        bearing_cos = float(state_vector[9])

        steer = 0
        if abs(bearing_sin) > 0.08 or bearing_cos < 0.0:
            steer = 1 if bearing_sin > 0.0 else -1

        front_left = float(state_vector[2])
        front_right = float(state_vector[4])
        if min(front_left, front_right) < 0.12:
            if front_left > front_right:
                steer = -1
            elif front_right > front_left:
                steer = 1
        elif abs(left_clearance - right_clearance) > 0.12 and abs(bearing_sin) <= 0.08:
            steer = -1 if left_clearance > right_clearance else 1

        front_distance = float(state_vector[3]) * MAX_RAY_DISTANCE
        stopping_distance = speed * speed / (2.0 * DECELERATION) + speed / FPS + 12.0
        if front_distance < stopping_distance:
            longitudinal = -1
        elif speed < MAX_SPEED_PX * 0.65 or front_distance > stopping_distance * 1.8:
            longitudinal = 1
        else:
            longitudinal = 0

        if speed < 12.0 and front_distance > 24.0:
            longitudinal = 1

        longitudinal_row = {1: 0, 0: 1, -1: 2}[longitudinal]
        return longitudinal_row * 3 + steer + 1