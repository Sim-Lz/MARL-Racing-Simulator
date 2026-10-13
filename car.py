# car.py
import math

import numpy as np
import pygame

from config import *

ACTION_MAP = {
    0: (-1, 1), 1: (0, 1), 2: (1, 1),    # Accelerazione combinata
    3: (-1, 0), 4: (0, 0), 5: (1, 0),    # Mantenimento / Inerzia
    6: (-1, -1), 7: (0, -1), 8: (1, -1),  # Frenata combinata
}

_SENSOR_RADIANS = np.radians(np.asarray(SENSOR_ANGLES, dtype=np.float64))
_SENSOR_RANGES = np.asarray(SENSOR_RANGES, dtype=np.float64)
# Distanza tra il centro e il bordo della carrozzeria lungo ciascun raggio.
_BODY_EXTENT = np.minimum(
    CAR_HALF_LENGTH / np.maximum(np.abs(np.cos(_SENSOR_RADIANS)), 1e-9),
    CAR_HALF_WIDTH / np.maximum(np.abs(np.sin(_SENSOR_RADIANS)), 1e-9),
)
_RAY_SAMPLES = np.arange(0.0, MAX_RAY_DISTANCE + CAR_HALF_LENGTH + SENSOR_STEP_PX, SENSOR_STEP_PX)


def saturated_sin(angle):
    """sin dell'angolo, saturato a ±1 oltre ±90° così che "dietro" non si confonda con "davanti"."""
    return math.sin(angle) if math.cos(angle) >= 0.0 else math.copysign(1.0, math.sin(angle))


def action_to_controls(action):
    """Traduce l'indice discreto (0-8) in (sterzo, longitudinale)."""
    return ACTION_MAP.get(int(action), (0, 0))


def controls_to_action(steer, longitudinal):
    row = {1: 0, 0: 1, -1: 2}[longitudinal]
    return row * 3 + steer + 1


class Car:
    def __init__(self, car_id, team_id, index_in_team, start_pos, start_angle, color):
        self.car_id = car_id
        self.team_id = team_id
        self.index_in_team = index_in_team
        self.color = color

        self.x = float(start_pos[0])
        self.y = float(start_pos[1])
        self.angle = float(start_angle)

        self.speed_px = 0.0
        self.speed = 0.0

        self.alive = True
        self.distance_traveled = 0.0
        self.laps = 0
        self.next_checkpoint = 0
        self.sensor_distances = np.zeros(NUM_SENSORS)
        self.traffic_features = (0.0, 0.0, 0.0)

        # Progresso lungo l'ascissa curvilinea (px). Negativo dietro la linea di partenza.
        self.track_index = None
        self.progress_px = 0.0
        self.max_progress_px = 0.0
        self.lateral_offset = 0.0
        self.heading_error = 0.0
        self.last_lap_timer = 0.0
        self.lap_times = []

        # --- METRICHE DI VALUTAZIONE SCIENTIFICA ---
        self.max_speed_reached = 0.0
        self.speed_sum = 0.0
        self.survival_time_s = 0.0
        self.total_steps = 0
        self.decisions = 0
        self.slipstream_frames = 0
        self.overtakes_performed = 0
        self.collisions_caused = 0
        self.collisions_suffered = 0
        self.checkpoints_passed_total = 0
        self.cumulative_reward = 0.0
        self.individual_reward_total = 0.0
        self.progress_reward_total = 0.0
        self.time_penalty_total = 0.0
        self.terminal_penalty_total = 0.0
        self.collision_penalty_total = 0.0
        self.completion_bonus_total = 0.0
        self.personal_best_bonus_total = 0.0
        self.personal_bests_beaten = 0
        self.slipstream_timer = 0
        self.stationary_frames = 0
        self.wrong_way_frames = 0
        self.no_progress_frames = 0
        self.progress_mark_px = 0.0
        self.termination_reason = "ALIVE"
        self.death_info = None

        # Respawn di addestramento (vedi TRAINING_RESPAWN): i fantasmi non contano in classifica.
        self.is_ghost = False
        self.slot = car_id
        self.line_stats = {
            "frames": 0, "abs_lateral": 0.0, "corner_frames": 0, "inside": 0.0,
            "speed_ratio": 0.0, "line_error": 0.0, "traffic_frames": 0,
        }
        self.training_respawns = 0
        self.training_reward_total = 0.0
        self.training_progress_total = 0.0

    # ------------------------------------------------------------------ sensori
    def cast_rays(self, track):
        """Distanza libera (dal bordo della carrozzeria) verso il muro lungo ogni raggio."""
        if not self.alive:
            return
        self._rays_stale = False
        angles = math.radians(self.angle) + _SENSOR_RADIANS
        reach = _RAY_SAMPLES[None, :]
        xs = self.x + np.cos(angles)[:, None] * reach
        ys = self.y + np.sin(angles)[:, None] * reach
        safe = track.drivable(xs, ys)
        within_range = reach <= (_SENSOR_RANGES + _BODY_EXTENT)[:, None]
        blocked = ~safe & within_range
        first_hit = np.where(blocked.any(axis=1), blocked.argmax(axis=1), within_range.sum(axis=1))
        hit_distance = _RAY_SAMPLES[np.minimum(first_hit, len(_RAY_SAMPLES) - 1)]
        hit_distance = np.where(blocked.any(axis=1), hit_distance, _SENSOR_RANGES + _BODY_EXTENT)
        self.sensor_distances[:] = np.clip(hit_distance - _BODY_EXTENT, 0.0, _SENSOR_RANGES)

    def update_track_position(self, track, initial=False):
        """Aggiorna progresso continuo, offset laterale ed errore di direzione rispetto alla mezzeria."""
        if track.center is None:
            return 0.0
        hint = None if initial or self.track_index is None else self.track_index
        index = track.project(self.x, self.y, hint)
        arc = index * track.step
        if initial:
            # Le auto partono dietro la linea: progresso negativo fino al primo passaggio.
            self.progress_px = arc - track.length if arc > track.length / 2.0 else arc
            self.max_progress_px = max(0.0, self.progress_px)
            self.progress_mark_px = self.progress_px
            delta = 0.0
        else:
            previous_arc = self.track_index * track.step
            delta = (arc - previous_arc + track.length / 2.0) % track.length - track.length / 2.0
            self.progress_px += delta
        self.track_index = index
        self.lateral_offset = track.lateral_offset(self.x, self.y, index)
        track_heading = math.atan2(track.tangent[index, 1], track.tangent[index, 0])
        self.heading_error = (math.radians(self.angle) - track_heading + math.pi) % (2.0 * math.pi) - math.pi
        return delta

    def race_progress(self, track):
        """Progresso continuo espresso in checkpoint (giri * N + checkpoint + frazione)."""
        if track.center is None or not track.checkpoints:
            return 0.0
        return self.progress_px / track.checkpoint_spacing

    def get_observation(self, track):
        """Muri (per raggio), velocità, posizione/direzione rispetto alla mezzeria, look-ahead e traffico."""
        if getattr(self, "_rays_stale", False):
            self.cast_rays(track)
        sensors = self.sensor_distances / _SENSOR_RANGES
        speed = min(1.0, self.speed_px / (MAX_SPEED_PX * 1.15))
        half_width = max(1.0, float(track.brush_radius))
        lateral = max(-1.0, min(1.0, self.lateral_offset / half_width))

        lookahead = []
        track_heading = []
        heading = math.radians(self.angle)
        for distance in LOOKAHEAD_DISTANCES_PX:
            if track.center is None:
                lookahead.append(0.0)
                track_heading.append(0.0)
                continue
            index = track.index_at(self.progress_px + distance)
            target = track.center[index]
            bearing = math.atan2(target[1] - self.y, target[0] - self.x) - heading
            lookahead.append(saturated_sin(bearing))
            relative = math.atan2(track.tangent[index, 1], track.tangent[index, 0]) - heading
            track_heading.append(saturated_sin(relative))

        return np.concatenate((
            sensors,
            [speed, lateral, math.sin(self.heading_error), math.cos(self.heading_error)],
            lookahead,
            track_heading,
            self.traffic_features,
        )).astype(np.float32)

    def is_on_track(self, track):
        """Test the car's footprint rather than treating side sensor rays as collisions."""
        heading = math.radians(self.angle)
        forward = (math.cos(heading), math.sin(heading))
        lateral = (-forward[1], forward[0])
        samples = [(0.0, 0.0), (CAR_HALF_LENGTH, 0.0), (-CAR_HALF_LENGTH, 0.0),
                   (0.0, -CAR_HALF_WIDTH), (0.0, CAR_HALF_WIDTH)]
        for longitudinal in (-CAR_HALF_LENGTH, CAR_HALF_LENGTH):
            for sideways in (-CAR_HALF_WIDTH, CAR_HALF_WIDTH):
                samples.append((longitudinal, sideways))
        xs = [self.x + forward[0] * lo + lateral[0] * si for lo, si in samples]
        ys = [self.y + forward[1] * lo + lateral[1] * si for lo, si in samples]
        return bool(np.all(track.drivable(xs, ys)))

    # ------------------------------------------------------------------ dinamica
    def steering_rate(self):
        """Gradi per frame. Sotto MIN_SPEED il raggio minimo resta costante: niente rotazioni da fermi."""
        if self.speed_px <= MIN_SPEED_PX:
            return MAX_STEER_ANGLE * self.speed_px / MIN_SPEED_PX
        return MAX_STEER_ANGLE * MIN_SPEED_PX / self.speed_px

    def kill(self, reason):
        if self.alive:
            self.alive = False
            self.termination_reason = reason

    def update(self, action, track, has_slipstream=False):
        if not self.alive:
            return

        self.total_steps += 1

        if has_slipstream:
            self.slipstream_timer = 30
            self.slipstream_frames += 1
        elif self.slipstream_timer > 0:
            self.slipstream_timer -= 1

        is_currently_boosted = (self.slipstream_timer > 0)
        top_speed = MAX_SPEED_PX * 1.15 if is_currently_boosted else MAX_SPEED_PX
        current_accel = ACCELERATION * 1.5 if is_currently_boosted else ACCELERATION

        if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
            steer_act, long_act = int(action[0]), int(action[1])
        else:
            steer_act, long_act = action_to_controls(action)

        # 1. Sterzo (vincolo fisico legato alla velocità)
        self.angle = (self.angle + steer_act * self.steering_rate()) % 360

        # 2. Dinamica longitudinale
        if long_act == 1:
            self.speed_px += current_accel / FPS
        elif long_act == -1:
            self.speed_px -= DECELERATION / FPS
        else:
            self.speed_px -= COASTING_DECELERATION / FPS

        # Perdita di efficienza graduale in curva (prima era un taglio istantaneo al 85%, che
        # penalizzava in modo discontinuo qualunque sterzata ad alta velocità).
        if steer_act != 0 and self.speed_px > top_speed * 0.85:
            self.speed_px -= SCRUB_DECELERATION / FPS
        self.speed_px = max(0.0, min(self.speed_px, top_speed))

        self.speed = self.speed_px / FPS
        self.max_speed_reached = max(self.max_speed_reached, self.speed_px)
        self.speed_sum += self.speed_px

        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.distance_traveled += self.speed
        self._rays_stale = True  # i raggi vengono ricalcolati solo quando serve un'osservazione

        if not self.is_on_track(track):
            self.kill("CRASHED")

    def check_disqualifications(self, track):
        """Stallo, contromano prolungato e assenza di progresso. Ritorna True se la vettura è squalificata."""
        if not self.alive:
            return False
        self.stationary_frames = self.stationary_frames + 1 if self.speed_px <= STALL_SPEED_PX else 0
        self.wrong_way_frames = self.wrong_way_frames + 1 if math.cos(self.heading_error) < -0.2 else 0
        if self.progress_px >= self.progress_mark_px + track.checkpoint_spacing:
            self.progress_mark_px = self.progress_px
            self.no_progress_frames = 0
        else:
            self.no_progress_frames += 1

        if self.stationary_frames >= STALL_TIMEOUT_S * FPS:
            self.kill("DSQ_STALLED")
        elif self.wrong_way_frames >= WRONG_WAY_TIMEOUT_S * FPS:
            self.kill("DSQ_BACK")
        elif self.no_progress_frames >= NO_PROGRESS_TIMEOUT_S * FPS:
            self.kill("DSQ_NO_PROGRESS")
        return not self.alive

    def average_speed(self):
        return self.speed_sum / max(1, self.total_steps)

    def draw(self, screen):
        if not self.alive:
            return
        car_surface = pygame.Surface((int(CAR_HALF_LENGTH * 2), int(CAR_HALF_WIDTH * 2)), pygame.SRCALPHA)
        if self.is_ghost:
            pygame.draw.rect(car_surface, (*self.color, 110), car_surface.get_rect(), 1)
        else:
            car_surface.fill(self.color)
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        new_rect = rotated_surface.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated_surface, new_rect.topleft)
