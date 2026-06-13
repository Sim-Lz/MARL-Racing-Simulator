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
        
        # --- METRICHE DI VALUTAZIONE SCIENTIFICA E REWARD ---
        self.max_speed_reached = 0.0
        self.total_steps = 0
        self.slipstream_frames = 0
        self.overtakes_performed = 0
        self.checkpoints_passed_total = 0
        self.cumulative_reward = 0.0  # Problema 7: Tracciamento Gradiente/Reward Finale
        
        self.slipstream_timer = 0 

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
        
        # Problema 5: Incremento massiccio dell'accelerazione sotto effetto scia
        current_accel = ACCELERATION * 1.25 if is_currently_boosted else ACCELERATION

        current_max_steer = MAX_STEER_ANGLE * (MIN_SPEED_PX / max(self.speed_px, MIN_SPEED_PX)) if self.speed_px > 0 else MAX_STEER_ANGLE
        if action == -1:
            self.angle -= current_max_steer
        elif action == 1:
            self.angle += current_max_steer
        self.angle = self.angle % 360

        # Problema 4: Analisi corretta del radar per la V-MAX
        center_distance = self.sensor_distances[3]  # Solo il raggio 0°
        
        if center_distance > 140.0:
            target_speed = top_speed
        else:
            front_min = min(self.sensor_distances[2:5])
            safe_ratio = front_min / 140.0
            target_speed = 30.0 + (top_speed - 30.0) * safe_ratio
        
        if action != 0:
            target_speed *= 0.80  
            
        target_speed = max(15.0, min(target_speed, top_speed))

        # Inerzia lineare con accelerazione boostata
        if self.speed_px < target_speed:
            self.speed_px = min(target_speed, self.speed_px + (current_accel / FPS))
        elif self.speed_px > target_speed:
            self.speed_px = max(target_speed, self.speed_px - (DECELERATION / FPS))

        self.speed = self.speed_px / FPS
        if self.speed_px > self.max_speed_reached:
            self.max_speed_reached = self.speed_px

        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.distance_traveled += self.speed
        
        self.cast_rays(track)
        
        if np.any(self.sensor_distances[:5] < CRASH_THRESHOLD) or not track.get_pixel_safety(self.x, self.y):
            self.alive = False

    def draw(self, screen):
        if not self.alive:
            return
        car_surface = pygame.Surface((16, 10), pygame.SRCALPHA)
        car_surface.fill(self.color)
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        new_rect = rotated_surface.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated_surface, new_rect.topleft)