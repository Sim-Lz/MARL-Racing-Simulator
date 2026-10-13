# rl_models/heuristic.py
import math

from car import controls_to_action
from config import (
    DECELERATION,
    FPS,
    LOOKAHEAD_DISTANCES_PX,
    MAX_SPEED_PX,
    MAX_STEER_ANGLE,
    MIN_SPEED_PX,
    OBS_HEADING_COS,
    OBS_HEADING_SIN,
    OBS_LOOKAHEAD,
    OBS_SPEED,
    OBS_TRACK_HEADING,
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
LEFT_SIDE = SENSOR_ANGLES.index(-90)
RIGHT_SIDE = SENSOR_ANGLES.index(90)
SPEED_SCALE = MAX_SPEED_PX * 1.15   # normalizzazione della velocità nell'osservazione (Car.get_observation)
# Capacità di curvatura del modello di sterzo (vedi Car.steering_rate): sopra MIN_SPEED k_max = K / v^2.
CURVATURE_CAPACITY = math.radians(MAX_STEER_ANGLE) * MIN_SPEED_PX * FPS


class HeuristicController(Agent):
    """Controllore di riferimento a regole fisse che usa SOLO l'osservazione dei modelli che apprendono.

    Non conosce la traiettoria ideale né il suo profilo di velocità: vede gli stessi sensori dei muri,
    la stessa velocità, posizione rispetto alla mezzeria, punti di look-ahead e traffico. Conosce invece
    la fisica dell'auto (raggio di sterzata in funzione della velocità, decelerazione in frenata), che usa
    per decidere quanto frenare prima di una curva. Non impara: è la stessa regola a ogni giro, utile come
    termine di paragone per capire quanto i modelli si avvicinano a una guida "ragionevole".

    - Sterzo: pure pursuit verso un punto della mezzeria davanti all'auto (più lontano a velocità alta).
      Inseguire un punto davanti taglia naturalmente le curve verso l'interno, senza che la traiettoria
      sia stata calcolata a priori.
    - Velocità: la curvatura della pista nei tratti davanti si stima dalla direzione della mezzeria nei
      punti di look-ahead; per ogni tratto la velocità massima percorribile e lo spazio di frenata
      fissano la velocità ammessa ora. Frenata d'emergenza se il muro davanti è vicinissimo.
    - Traffico: sposta il punto inseguito dal lato opposto all'auto davanti se c'è spazio, altrimenti
      resta in scia; frena solo se il tamponamento è imminente.
    """

    algorithm = "HEURISTIC"
    curvature_margin = 0.85          # frazione della capacità di sterzo usata in curva
    braking_margin = 0.8             # frazione della decelerazione usata per pianificare la frenata
    speed_hysteresis = 3.0
    wall_clearance_px = 9.0
    steer_threshold = 0.3            # frazione della sterzata massima sotto la quale si va dritti
    pass_bias_rad = 0.12             # spostamento angolare del punto inseguito durante il sorpasso
    pass_room_px = 18.0
    saturation = 0.98                # |sin| oltre cui la direzione della pista è considerata satura
    tight_bend_rad = 0.6             # svolta (entro 150 px) oltre la quale si insegue solo il punto vicino

    def act(self, car_id, observation):
        return self.drive(observation)

    @staticmethod
    def _angle(sin_value):
        return math.asin(max(-1.0, min(1.0, float(sin_value))))

    def _target_speed(self, observation, distances):
        """Velocità ammessa ora: limiti di curvatura dei tratti davanti, raggiungibili frenando, e muro frontale."""
        heading_error = math.atan2(float(observation[OBS_HEADING_SIN]), float(observation[OBS_HEADING_COS]))
        previous_angle, previous_distance = -heading_error, 0.0   # direzione della pista qui, rispetto all'auto
        allowed = MAX_SPEED_PX
        for k, distance in enumerate(LOOKAHEAD_DISTANCES_PX):
            value = float(observation[OBS_TRACK_HEADING + k])
            # Oltre ±90° l'osservazione satura: per prudenza la svolta si considera un tornante (±180°).
            angle = math.copysign(math.pi, value) if abs(value) >= self.saturation else self._angle(value)
            curvature = abs(angle - previous_angle) / max(1.0, distance - previous_distance)
            limit = math.sqrt(self.curvature_margin * CURVATURE_CAPACITY / max(curvature, 1e-9))
            limit = max(MIN_SPEED_PX, min(MAX_SPEED_PX, limit))
            allowed = min(allowed, math.sqrt(limit ** 2 + 2.0 * DECELERATION * self.braking_margin * previous_distance))
            previous_angle, previous_distance = angle, distance
        return allowed

    def drive(self, observation):
        speed = float(observation[OBS_SPEED]) * SPEED_SCALE
        distances = [float(observation[i]) * SENSOR_RANGES[i] for i in range(len(SENSOR_RANGES))]

        # --- Sterzo: pure pursuit sulla mezzeria, punto vicino a bassa velocità e medio ad alta. Nelle curve
        # strette il punto torna vicino: un punto lontano taglierebbe la curva fino al muro interno.
        heading_error = math.atan2(float(observation[OBS_HEADING_SIN]), float(observation[OBS_HEADING_COS]))
        bend = abs(self._angle(observation[OBS_TRACK_HEADING + 1]) + heading_error)   # svolta entro 150 px
        blend = min(1.0, max(0.0, (speed - MIN_SPEED_PX) / (MAX_SPEED_PX - MIN_SPEED_PX)))
        blend *= max(0.0, 1.0 - bend / self.tight_bend_rad)
        near, mid = self._angle(observation[OBS_LOOKAHEAD]), self._angle(observation[OBS_LOOKAHEAD + 1])
        alpha = (1.0 - blend) * near + blend * mid
        aim_distance = (1.0 - blend) * LOOKAHEAD_DISTANCES_PX[0] + blend * LOOKAHEAD_DISTANCES_PX[1]

        proximity = float(observation[OBS_TRAFFIC_PROXIMITY])
        closing = float(observation[OBS_TRAFFIC_CLOSING])
        bearing = float(observation[OBS_TRAFFIC_BEARING])
        traffic_ahead = proximity > 0.35 and closing > 0.0 and abs(bearing) < 0.6
        following = False
        if traffic_ahead:
            side = -1 if bearing > 0 else 1                     # lato opposto all'auto davanti (+ = destra)
            room = distances[RIGHT_SIDE] if side > 0 else distances[LEFT_SIDE]
            if room >= self.pass_room_px:
                alpha += side * self.pass_bias_rad
            else:
                following = True

        rate = MAX_STEER_ANGLE * (speed / MIN_SPEED_PX if speed <= MIN_SPEED_PX else MIN_SPEED_PX / max(speed, 1e-6))
        if rate < 1e-3:
            steer = 0
        else:
            needed = math.degrees(2.0 * math.sin(alpha) / aim_distance * speed / FPS)
            steer = 0 if abs(needed) < self.steer_threshold * rate else (1 if needed > 0 else -1)

        left_gap = min(distances[i] for i in LEFT_DIAGONALS)
        right_gap = min(distances[i] for i in RIGHT_DIAGONALS)
        if min(left_gap, right_gap) < self.wall_clearance_px:
            steer = 1 if left_gap < right_gap else -1

        # --- Velocità.
        target_speed = self._target_speed(observation, distances)
        # Solo frenata d'emergenza sul muro frontale: in curva i raggi frontali vedono sempre il muro esterno,
        # e un limite sullo spazio di arresto completo farebbe frenare per tutta la curva.
        front_distance = min(distances[i] for i in FRONT_CONE)
        if front_distance < speed * 0.12 + 6.0 or speed > target_speed + self.speed_hysteresis:
            longitudinal = -1
        elif speed < target_speed - self.speed_hysteresis:
            longitudinal = 1
        else:
            longitudinal = 0

        if following:
            if proximity > 0.7 and speed > 40.0:
                longitudinal = -1
            elif proximity > 0.55 and longitudinal == 1:
                longitudinal = 0
        if proximity > 0.8 and closing > 0.0 and abs(bearing) < 0.25 and speed > 40.0:
            longitudinal = -1
        if speed < 12.0 and front_distance > 24.0:
            longitudinal = 1

        return controls_to_action(steer, longitudinal)
