# rl_models/heuristic.py
import math

from car import controls_to_action
from config import (
    DECELERATION,
    FPS,
    OBS_TRAFFIC_BEARING,
    OBS_TRAFFIC_CLOSING,
    OBS_TRAFFIC_PROXIMITY,
    SENSOR_ANGLES,
    SENSOR_RANGES,
)
from rl_models.base import Agent

FRONT_CONE = [i for i, angle in enumerate(SENSOR_ANGLES) if abs(angle) <= 10]
LEFT_DIAGONALS = [i for i, angle in enumerate(SENSOR_ANGLES) if -60 <= angle <= -20]
RIGHT_DIAGONALS = [i for i, angle in enumerate(SENSOR_ANGLES) if 20 <= angle <= 60]


class HeuristicController(Agent):
    """Controllore di riferimento con conoscenza del tracciato.

    Insegue la traiettoria ideale (curvatura minima, esterno-interno-esterno) calcolata da Track con
    pure pursuit e segue il profilo di velocità limite (curvatura, frenata, accelerazione). In traffico
    tenta il sorpasso spostandosi sul lato opposto all'auto davanti invece di accodarsi; frena solo se
    il tamponamento è imminente.
    A differenza dei modelli che apprendono, legge direttamente stato dell'auto e geometria (begin_track).
    """

    algorithm = "HEURISTIC"
    speed_hysteresis = 3.0
    pass_hold_frames = 45
    pass_shift_fraction = 0.7
    pass_room_px = 16.0              # spazio laterale minimo per tentare il sorpasso (auto larga 10 px)
    wall_clearance_px = 5.0
    lookahead_base_px = 30.0
    lookahead_gain_s = 0.12          # il punto inseguito si allontana con la velocità (px per px/s)
    steer_threshold = 0.3            # frazione della sterzata massima sotto la quale si va dritti

    def __init__(self):
        self.sim = None
        self._pass_state = {}

    def begin_track(self, sim):
        self.sim = sim

    def begin_race(self, car_ids):
        self._pass_state = {}

    def act(self, car_id, observation):
        if self.sim is None or self.sim.track.racing_offset is None:
            return controls_to_action(0, 1)
        return self.drive(self.sim.cars[car_id], self.sim.track, observation)

    def _choose_pass_side(self, car, track, bearing, index):
        """Lato di sorpasso con spazio sufficiente (0 = nessuno). Preferisce il lato opposto all'auto davanti."""
        room_left = car.lateral_offset - float(track.lateral_lower[index])
        room_right = float(track.lateral_upper[index]) - car.lateral_offset
        preferred = -1 if bearing > 0.05 else 1 if bearing < -0.05 else (1 if room_right > room_left else -1)
        for side in (preferred, -preferred):
            if (room_right if side > 0 else room_left) >= self.pass_room_px:
                return side
        return 0

    def drive(self, car, track, observation):
        speed = car.speed_px
        here = car.track_index or 0

        # --- Traffico: sorpassa solo se c'è spazio laterale e non si è in staccata; altrimenti segue a distanza.
        proximity = float(observation[OBS_TRAFFIC_PROXIMITY])
        closing = float(observation[OBS_TRAFFIC_CLOSING])
        bearing = float(observation[OBS_TRAFFIC_BEARING])
        side, frames = self._pass_state.get(car.car_id, (0, 0))
        braking_zone = min(float(track.racing_speed[(here + k) % len(track.racing_speed)]) for k in range(0, 60, 5)) < 0.8 * speed
        following = False
        if proximity > 0.35 and closing > 0.0 and abs(bearing) < 0.6:
            if frames <= 0:
                side = 0 if braking_zone else self._choose_pass_side(car, track, bearing, here)
            if side:
                frames = self.pass_hold_frames
            else:
                following = True
        else:
            frames -= 1
        self._pass_state[car.car_id] = (side if frames > 0 else 0, frames)

        # --- Sterzo: pure pursuit su un punto della traiettoria ideale (o della traiettoria di sorpasso).
        lookahead = self.lookahead_base_px + self.lookahead_gain_s * speed
        index = track.index_at(car.progress_px + lookahead)
        if side and frames > 0:
            offset = car.lateral_offset + side * self.pass_shift_fraction * track.brush_radius
            offset = max(float(track.lateral_lower[index]), min(float(track.lateral_upper[index]), offset))
            target_x = track.center[index, 0] + track.normal[index, 0] * offset
            target_y = track.center[index, 1] + track.normal[index, 1] * offset
        else:
            target_x, target_y = track.racing_points[index]
        dx, dy = target_x - car.x, target_y - car.y
        distance = max(1.0, math.hypot(dx, dy))
        alpha = (math.atan2(dy, dx) - math.radians(car.angle) + math.pi) % (2.0 * math.pi) - math.pi

        rate = car.steering_rate()
        if rate < 1e-3:
            steer = 0
        else:
            needed = math.degrees(2.0 * math.sin(alpha) / distance * speed / FPS)
            steer = 0 if abs(needed) < self.steer_threshold * rate else (1 if needed > 0 else -1)

        distances = [float(observation[i]) * SENSOR_RANGES[i] for i in range(len(SENSOR_RANGES))]
        left_gap = min(distances[i] for i in LEFT_DIAGONALS)
        right_gap = min(distances[i] for i in RIGHT_DIAGONALS)
        if min(left_gap, right_gap) < self.wall_clearance_px:
            steer = 1 if left_gap < right_gap else -1

        # --- Velocità: profilo limite della traiettoria ideale nel tratto immediatamente davanti.
        window = [(here + k) % len(track.racing_speed) for k in range(0, 12)]
        target_speed = min(float(track.racing_speed[i]) for i in window)
        front_distance = min(distances[i] for i in FRONT_CONE)
        # Solo frenata d'emergenza: in curva i raggi frontali vedono sempre il muro esterno, e un controllo
        # sullo spazio di arresto completo (come nel vecchio euristico) farebbe frenare per tutta la curva.
        # Le curve sono già gestite dal profilo di velocità.
        emergency_distance = speed * 0.12 + 6.0

        if front_distance < emergency_distance or speed > target_speed + self.speed_hysteresis:
            longitudinal = -1
        elif speed < target_speed - self.speed_hysteresis:
            longitudinal = 1
        else:
            longitudinal = 0

        # Senza spazio per passare si resta in scia a distanza di sicurezza invece di tamponare.
        if following and closing > 0.0:
            if proximity > 0.7 and speed > 40.0:
                longitudinal = -1
            elif proximity > 0.55 and longitudinal == 1:
                longitudinal = 0
        # Tamponamento imminente con l'auto proprio davanti.
        if proximity > 0.8 and closing > 0.0 and abs(bearing) < 0.25 and speed > 40.0:
            longitudinal = -1

        if speed < 12.0 and front_distance > 24.0:
            longitudinal = 1

        return controls_to_action(steer, longitudinal)
