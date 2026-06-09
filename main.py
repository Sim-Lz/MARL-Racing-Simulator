# main.py
import subprocess
import sys

# Controllo e installazione automatica delle dipendenze
try:
    import pygame
    import numpy
    import torch
    import math
    import random
except ImportError:
    print("Dipendenze mancanti rilevate. Avvio l'installazione automatica...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Installazione completata con successo!")
    except Exception as e:
        print(f"Errore durante l'installazione automatica: {e}")
        print("Prova a installare manualmente con: pip install pygame numpy torch")
        sys.exit(1)

from config import *
from track import Track
from car import Car
from team import Team

from rl_models.heuristic import HeuristicController
from rl_models.genetic import GeneticNetwork
from rl_models.qlearning import TabularQLearner
from rl_models.dqn import DQNAgent
from rl_models.policy_gradient import PolicyGradientAgent

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MARL Racing Simulator - Full Synchronized Stable Build")
clock = pygame.time.Clock()

track = Track(WIDTH, HEIGHT)
state = "DRAWING"

cars = []
teams = []
agents = {}  # Mappatura univoca car_id -> Istanza IA dedicata

# Instanziazione isolata dei controller IA per impedire cross-contamination di memoria
for car_id in range(TOTAL_CARS):
    team_id = car_id // CARS_PER_TEAM
    algo = TEAM_ALGORITHMS[team_id]
    if algo == "HEURISTIC": agents[car_id] = HeuristicController()
    elif algo == "GENETIC": agents[car_id] = GeneticNetwork()
    elif algo == "Q_LEARNING": agents[car_id] = TabularQLearner()
    elif algo == "DQN": agents[car_id] = DQNAgent()
    elif algo == "POLICY_GRADIENT": agents[car_id] = PolicyGradientAgent()

def initialize_race_session():
    global cars, teams
    cars.clear()
    teams.clear()
    
    # 1. GENERAZIONE ORDINE RANDOMICO DELLA GRIGLIA
    shuffled_car_ids = list(range(TOTAL_CARS))
    random.shuffle(shuffled_car_ids)
    
    total_pts = len(track.drawing_points)
    temp_cars = [None] * TOTAL_CARS
    
    # 2. POSIZIONAMENTO SICURO LUNGO LA MEZZERIA DEL TRACCIATO
    for slot, car_id in enumerate(shuffled_car_ids):
        team_id = car_id // CARS_PER_TEAM
        color = TEAM_COLORS[team_id]
        
        # Spaziamo le auto a ritroso dall'ultimo punto disegnato (passo di 6 pixel per auto)
        pt_idx = (total_pts - 1 - (slot * 6)) % total_pts
        spawn_x, spawn_y = track.drawing_points[pt_idx]
        
        # 3. ALLINEAMENTO AUTOMATICO ALLA DIREZIONE DEL CIRCUITO
        next_idx = (pt_idx + 5) % total_pts
        p_curr = track.drawing_points[pt_idx]
        p_next = track.drawing_points[next_idx]
        local_angle = math.degrees(math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0]))
        
        # Istanziamo l'auto con la posizione e l'orientamento perfetto
        new_car = Car(car_id, team_id, car_id % CARS_PER_TEAM, (spawn_x, spawn_y), local_angle, color)
        new_car.cast_rays(track)
        temp_cars[car_id] = new_car
        
    cars = temp_cars
    
    for team_id in range(NUM_TEAMS):
        team_cars = [cars[team_id * 2], cars[team_id * 2 + 1]]
        teams.append(Team(team_id, TEAM_ALGORITHMS[team_id], team_cars))

def check_car_to_car_collisions():
    num_checkpoints = len(track.checkpoints)
    if num_checkpoints == 0: return

    for i in range(TOTAL_CARS):
        if not cars[i].alive: continue  
        for j in range(i + 1, TOTAL_CARS):
            if not cars[j].alive: continue
            
            if math.hypot(cars[i].x - cars[j].x, cars[i].y - cars[j].y) < 12.0:
                # Calcolo della vicinanza telemetrica in base ai checkpoint
                cp_dist = abs(cars[i].next_checkpoint - cars[j].next_checkpoint)
                cp_dist = min(cp_dist, num_checkpoints - cp_dist)
                
                # Se la distanza è minima sono sullo stesso piano, altrimenti si tratta del ponte Suzuka
                if cp_dist <= 2:
                    cars[i].alive = False
                    cars[j].alive = False

session_timer = 0
generation_count = 1
race_finished = False
winner_info = ""

while True:
    dt = clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN and state == "DRAWING" and len(track.drawing_points) > 20:
                track.close_track()
                track.generate_checkpoints()
                state = "RACING"
                initialize_race_session()
                session_timer = 0
                race_finished = False
            
            if state == "DRAWING":
                if event.key == pygame.K_UP:
                    track.brush_radius = min(80, track.brush_radius + 5)
                elif event.key == pygame.K_DOWN:
                    track.brush_radius = max(20, track.brush_radius - 5)

    if state == "DRAWING":
        track.draw_step(screen)
    elif state == "RACING":
        session_timer += dt
        screen.blit(track.surface, (0, 0))
        
        # Snapshot dei vettori di stato correnti prima dell'applicazione delle decisioni IA
        current_states = {car.car_id: car.sensor_distances.copy() for car in cars if car.alive}
        actions = {}
        
        for car in cars:
            if not car.alive: continue
            agent = agents[car.car_id]
            algo = TEAM_ALGORITHMS[car.team_id]
            s = current_states[car.car_id]
            
            if algo == "HEURISTIC": actions[car.car_id] = agent.select_action(s)
            elif algo == "GENETIC": actions[car.car_id] = agent.forward(s)
            elif algo == "Q_LEARNING": actions[car.car_id] = agent.select_action(agent.discretize(s))
            elif algo == "DQN": actions[car.car_id] = agent.select_action(s)
            elif algo == "POLICY_GRADIENT": actions[car.car_id] = agent.select_action(s)

        for car in cars:
            if car.alive:
                car.update(actions[car.car_id], track)
                
                if track.checkpoints:
                    target_cp = track.checkpoints[car.next_checkpoint]
                    if math.hypot(car.x - target_cp[0], car.y - target_cp[1]) < track.brush_radius * 1.3:
                        car.next_checkpoint += 1
                        if car.next_checkpoint >= len(track.checkpoints):
                            car.next_checkpoint = 0
                            car.laps += 1
                            if car.laps >= TARGET_LAPS:
                                race_finished = True
                                winner_info = f"Vettura {car.car_id} ({TEAM_ALGORITHMS[car.team_id]})"
                
                car.draw(screen)

        check_car_to_car_collisions()

        # Step di Computazione ed Addestramento dei Reward Online
        for team in teams:
            step_rewards = team.compute_cooperative_rewards()
            for car in team.cars:
                if car.car_id not in current_states: continue  
                
                agent = agents[car.car_id]
                algo = TEAM_ALGORITHMS[car.team_id]
                r = step_rewards[car.car_id]
                s = current_states[car.car_id]
                s_prime = car.sensor_distances.copy()
                a = actions[car.car_id]
                done = not car.alive
                
                # VERIFICA E PENALITÀ CONTROMANO
                if track.checkpoints and car.alive:
                    target_cp = track.checkpoints[car.next_checkpoint]
                    dx = target_cp[0] - car.x
                    dy = target_cp[1] - car.y
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        # Vettore normalizzato verso il checkpoint obiettivo
                        target_dir_x = dx / dist
                        target_dir_y = dy / dist
                        # Vettore di orientamento anteriore dell'auto
                        car_dir_x = math.cos(math.radians(car.angle))
                        car_dir_y = math.sin(math.radians(car.angle))
                        
                        # Se il prodotto scalare è negativo, l'auto guarda dalla parte opposta al checkpoint
                        dot_product = target_dir_x * car_dir_x + target_dir_y * car_dir_y
                        if dot_product < 0:
                            r -= 45.0  # Pesante penalità per scoraggiare il senso di marcia invertito
                
                # REWARD SHAPING ANTI-SCIA (Già esistente nel tuo file)
                if s[3] < 55.0 and not done:
                    if a != 0: r += 25.0
                    else: r -= 15.0
                
                if algo == "Q_LEARNING":
                    agent.update(agent.discretize(s), a, r, agent.discretize(s_prime))
                elif algo == "DQN":
                    agent.store_transition(s, a, r, s_prime, done)
                    agent.train_step()
                elif algo == "POLICY_GRADIENT":
                    agent.rewards.append(r)

        active_cars = sum(1 for car in cars if car.alive)
        
        font = pygame.font.SysFont("Arial", 16)
        ui_text = font.render(f"Simulazione {generation_count} | Attive: {active_cars}/20 | Target: {TARGET_LAPS} Giri | Tempo: {session_timer//1000}s", True, (255, 255, 0))
        screen.blit(ui_text, (20, 20))
        
        # Controllo Fine Sessione (Traguardo raggiunto, estinzione o timeout di gara)
        if race_finished or active_cars == 0 or session_timer >= SESSION_DURATION_MS:
            if race_finished:
                print(f"--- SESSIONE TERMINATA: {winner_info} vince completando {TARGET_LAPS} giri! ---")
            
            # Calcolo e distribuzione finale dei canali ad evoluzione genetica (Offline step)
            genetic_car_ids = [c.car_id for c in cars if TEAM_ALGORITHMS[c.team_id] == "GENETIC"]
            if genetic_car_ids:
                best_genetic_id = max(genetic_car_ids, key=lambda cid: cars[cid].distance_traveled)
                best_net = agents[best_genetic_id]
                for cid in genetic_car_ids:
                    if cid != best_genetic_id:
                        agents[cid].W1 = best_net.W1.copy()
                        agents[cid].W2 = best_net.W2.copy()
                        agents[cid].mutate()

            # Ottimizzazione tramite Policy Gradient a fine corsa
            for car_id, agent in agents.items():
                if TEAM_ALGORITHMS[cars[car_id].team_id] == "POLICY_GRADIENT":
                    agent.update_policy()
                
            generation_count += 1

            # INTERRUZIONE DEL SIMULATORE DOPO 10 GENERAZIONI
            if generation_count > 10:
                print("\n=======================================================")
                print(" SIMULAZIONE COMPLETATA: Raggiunto il limite di 10 sessioni.")
                print("=======================================================")
                pygame.quit()
                sys.exit()

            initialize_race_session()
            session_timer = 0
            race_finished = False

    pygame.display.flip()