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
        
        self.speed_px = MIN_SPEED_PX  
        self.speed = self.speed_px / FPS
        
        self.alive = True
        self.distance_traveled = 0.0
        self.laps = 0  
        self.next_checkpoint = 0  
        self.sensor_distances = np.zeros(len(SENSOR_ANGLES))

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

    def update(self, action, track):
        if not self.alive:
            return

        # 1. Applicazione dello sterzo dinamico (Sottosterzo ad alta velocità)
        current_max_steer = MAX_STEER_ANGLE * (MIN_SPEED_PX / max(self.speed_px, MIN_SPEED_PX))

        if action == -1:
            self.angle -= current_max_steer
        elif action == 1:
            self.angle += current_max_steer
            
        self.angle = self.angle % 360

        # 2. Calcolo della velocità target predittiva tramite Sensor Fusion frontale
        # Estrae i dati dei sensori: indice 1 (-45°), 2 (-22.5°), 3 (0°), 4 (+22.5°), 5 (+45°)
        front_distance = min(self.sensor_distances[1:6])
        ray_ratio = front_distance / MAX_RAY_DISTANCE
        target_speed = MIN_SPEED_PX + (MAX_SPEED_PX - MIN_SPEED_PX) * ray_ratio
        
        if action != 0:
            target_speed *= 0.85  # Attrito da curvatura (Tire Scrub)
            
        target_speed = max(MIN_SPEED_PX, min(target_speed, MAX_SPEED_PX))

        # 3. Simulazione dell'inerzia lineare (Accelerazione e Frenata)
        if self.speed_px < target_speed:
            self.speed_px = min(target_speed, self.speed_px + (ACCELERATION / FPS))
        elif self.speed_px > target_speed:
            self.speed_px = max(target_speed, self.speed_px - (DECELERATION / FPS))

        self.speed = self.speed_px / FPS

        # 4. Spostamento cinematico vettoriale
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        
        self.distance_traveled += self.speed
        self.cast_rays(track)
        
        # Validazione stato vitale
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