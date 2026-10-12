# car.py
import pygame
import numpy as np
import math
from config import *

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
        self.sensor_distances = np.zeros(len(SENSOR_ANGLES))

        self.last_lap_timer = 0.0
        self.lap_bonus_reward = 0.0
        self.crash_triggered = False
        
        # --- METRICHE DI VALUTAZIONE SCIENTIFICA ---
        self.max_speed_reached = 0.0
        self.survival_time_s = 0.0
        self.total_steps = 0
        self.slipstream_frames = 0
        self.overtakes_performed = 0
        self.checkpoints_passed_total = 0
        self.cumulative_reward = 0.0  
        self.penalties_accumulated = 0.0
        self.progress_reward_total = 0.0
        self.time_penalty_total = 0.0
        self.terminal_penalty_total = 0.0
        self.completion_bonus_total = 0.0
        self.slipstream_timer = 0 
        self.stationary_frames = 0
        self.termination_reason = "ALIVE"

    def cast_rays(self, track):
        if not self.alive:
            return
        for i, alpha in enumerate(SENSOR_ANGLES):
            rad_angle = math.radians(self.angle + alpha)
            dx = math.cos(rad_angle)
            dy = math.sin(rad_angle)
            
            distance = 0.0
            while distance < MAX_RAY_DISTANCE:
                target_x = self.x + dx * distance
                target_y = self.y + dy * distance
                if not track.get_pixel_safety(target_x, target_y):
                    break
                distance += 2.0  
            self.sensor_distances[i] = distance

    def get_observation(self, track):
        """Return normalized wall, speed, and next-checkpoint direction features."""
        sensors = np.clip(self.sensor_distances / MAX_RAY_DISTANCE, 0.0, 1.0)
        speed = np.clip(self.speed_px / (MAX_SPEED_PX * 1.15), 0.0, 1.0)

        if track.checkpoints:
            target = track.checkpoints[self.next_checkpoint % len(track.checkpoints)]
            dx = target[0] - self.x
            dy = target[1] - self.y
            bearing = math.atan2(dy, dx) - math.radians(self.angle)
            bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
            bearing_sin = math.sin(bearing)
            bearing_cos = math.cos(bearing)
        else:
            bearing_sin, bearing_cos = 0.0, 1.0

        return np.concatenate((sensors, [speed, bearing_sin, bearing_cos])).astype(np.float32)

    def race_progress(self, track):
        """Estimate continuous checkpoint progress, including motion within a segment."""
        if not track.checkpoints:
            return 0.0

        count = len(track.checkpoints)
        checkpoint_index = self.next_checkpoint % count
        previous = track.checkpoints[(checkpoint_index - 1) % count]
        target = track.checkpoints[checkpoint_index]
        segment_length = math.hypot(target[0] - previous[0], target[1] - previous[1])
        if segment_length <= 1e-6:
            fraction = 0.0
        else:
            distance = math.hypot(target[0] - self.x, target[1] - self.y)
            fraction = max(-1.0, min(1.0, 1.0 - distance / segment_length))
        return self.laps * count + self.next_checkpoint + fraction

    def is_on_track(self, track):
        """Test the car's footprint rather than treating side sensor rays as collisions."""
        heading = math.radians(self.angle)
        forward = (math.cos(heading), math.sin(heading))
        lateral = (-forward[1], forward[0])
        half_length, half_width = 8.0, 5.0

        samples = [(0.0, 0.0), (half_length, 0.0), (-half_length, 0.0)]
        for longitudinal in (-half_length, half_length):
            for sideways in (-half_width, half_width):
                samples.append((longitudinal, sideways))
        for sideways in (-half_width, half_width):
            samples.append((0.0, sideways))

        return all(
            track.get_pixel_safety(
                self.x + forward[0] * longitudinal + lateral[0] * sideways,
                self.y + forward[1] * longitudinal + lateral[1] * sideways,
            )
            for longitudinal, sideways in samples
        )

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

        current_max_steer = MAX_STEER_ANGLE * (MIN_SPEED_PX / max(self.speed_px, MIN_SPEED_PX)) if self.speed_px > 0 else MAX_STEER_ANGLE
        
        # Mappatura delle 9 combinazioni discrete: [Sterzo, Longitudinale]
        # Sterzo: -1 = Sinistra, 0 = Dritto, 1 = Destra
        # Longitudinale: 1 = Accelera, 0 = Coasting, -1 = Frena
        action_map = {
            0: (-1,  1), 1: (0,  1), 2: (1,  1),  # Accelerazione combinata
            3: (-1,  0), 4: (0,  0), 5: (1,  0),  # Mantenimento / Inerzia
            6: (-1, -1), 7: (0, -1), 8: (1, -1)   # Frenata combinata
        }
        
        # Fallback di sicurezza se i vecchi modelli estraggono indici fuori range o formati non mappati
        if action in action_map:
            steer_act, long_act = action_map[action]
        elif isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
            steer_act, long_act = action[0], action[1]
        else:
            steer_act, long_act = 0, 1  # Fail-safe: procedi dritto accelerando

        # 1. Controllo Sterzo (Vincolo fisico legato alla velocità)
        if steer_act == -1:
            self.angle -= current_max_steer
        elif steer_act == 1:
            self.angle += current_max_steer
        self.angle = self.angle % 360

        # 2. Dinamica Longitudinale Diretta
        if long_act == 1:
            # Spinta massima erogata dal motore
            self.speed_px += (current_accel / FPS)
        elif long_act == -1:
            # Decelerazione meccanica dell'impianto frenante
            self.speed_px -= (DECELERATION / FPS)
        else:
            # Attrito naturale (Coasting) al rilascio dei pedali
            self.speed_px -= (25.0 / FPS)

        # Vincolo Logico: Impedisce la retromarcia e applica la perdita di efficienza (Tire Scrub) in curva
        scrub_limit = top_speed * 0.85 if steer_act != 0 else top_speed
        self.speed_px = max(0.0, min(self.speed_px, scrub_limit))
        
        self.speed = self.speed_px / FPS
        if self.speed_px > self.max_speed_reached:
            self.max_speed_reached = self.speed_px

        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.distance_traveled += self.speed
        
        self.cast_rays(track)
        
        # Controllo anti-stallo: squalifica se ferma per più di 3 secondi (180 frame a 60 FPS)
        if self.speed_px <= 5.0: # consideriamo stallo una velocità di 5px/s o inferiore, così da indurre comunque l'agente a muoversi 
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0

        if self.stationary_frames >= 180:
            self.alive = False
            self.termination_reason = "DSQ_STALLED"

        if not self.is_on_track(track):
            self.alive = False
            self.termination_reason = "CRASHED"
            

    def draw(self, screen):
        if not self.alive:
            return
        car_surface = pygame.Surface((16, 10), pygame.SRCALPHA)
        car_surface.fill(self.color)
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        new_rect = rotated_surface.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated_surface, new_rect.topleft)