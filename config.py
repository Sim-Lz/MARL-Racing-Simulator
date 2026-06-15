# config.py
import pygame

# Dimensioni Schermo
WIDTH = 1500
HEIGHT = 1000
FPS = 60

# Configurazione Simulazione MARL
CARS_PER_TEAM = 2
NUM_TEAMS = 10
TOTAL_CARS = CARS_PER_TEAM * NUM_TEAMS
TARGET_LAPS = 5
NUM_SIMULATIONS = 50
SESSION_DURATION_MS = 180000  # 180 Secondi di timeout sessione

# Algoritmi assegnati ai rispettivi Team
TEAM_ALGORITHMS = [
    "HEURISTIC",       # Team 0
    "GENETIC",         # Team 1
    "Q_LEARNING",      # Team 2
    "DQN",             # Team 3
    "POLICY_GRADIENT", # Team 4
    "GENETIC",         # Team 5
    "Q_LEARNING",      # Team 6
    "DQN",             # Team 7
    "POLICY_GRADIENT", # Team 8
    "HEURISTIC"        # Team 9
]

# Palette Colori Identificativi Team
TEAM_COLORS = [
    (230, 25, 75),   # Rosso
    (60, 180, 75),   # Verde
    (255, 225, 25),  # Giallo
    (0, 130, 200),   # Blu
    (245, 130, 48),  # Arancione
    (145, 30, 180),  # Viola
    (70, 240, 240),  # Ciano
    (240, 50, 230),  # Magenta
    (210, 245, 60),  # Lime
    (0, 128, 128)    # Verde Scuro
]

# Configurazione Sensori (Raggi di Prossimità)
SENSOR_ANGLES = [-90, -45, -22.5, 0, 22.5, 45, 90]
MAX_RAY_DISTANCE = 250.0
CRASH_THRESHOLD = 8.0

# Dinamica del Veicolo e Limiti di Tolleranza
MAX_STEER_ANGLE = 5.0       # Angolo massimo di sterzata (gradi)
MAX_SPEED_PX = 360.0       # Velocità massima nei rettilinei (px/s)
MIN_SPEED_PX = 80.0       # Velocità minima di sicurezza nelle curve (px/s)
ACCELERATION = 180.0       # Rateo incremento velocità (px/s^2)
DECELERATION = 260.0       # Rateo decremento velocità (px/s^2)