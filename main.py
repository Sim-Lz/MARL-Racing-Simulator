# main.py
import subprocess
import sys

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

# ==============================================================================
# CONFIGURAZIONE DINAMICA TRACCIATO E REWARDS
# ==============================================================================
NUM_CHECKPOINTS = 100          # Numero di checkpoint uniformemente distribuiti lungo il tracciato, indipendentemente dalla lunghezza totale
BASE_CHECKPOINT_REWARD = 200.0  # Reward totale distribuito lungo un intero giro

# Calcolo automatico degli indici di settore in base al percentile
IDX_S1 = int(NUM_CHECKPOINTS * 0.33)
IDX_S2 = int(NUM_CHECKPOINTS * 0.67)
# ==============================================================================

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MARL Racing Simulator - Full Synchronized Stable Build")
clock = pygame.time.Clock()

track = Track(WIDTH, HEIGHT)
state = "DRAWING"

cars = []
teams = []
agents = {}

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
    
    shuffled_car_ids = list(range(TOTAL_CARS))
    random.shuffle(shuffled_car_ids)
    
    total_pts = len(track.drawing_points)
    temp_cars = [None] * TOTAL_CARS
    
    for slot, car_id in enumerate(shuffled_car_ids):
        team_id = car_id // CARS_PER_TEAM
        color = TEAM_COLORS[team_id]
        
        pt_idx = (total_pts - 1 - (slot * 6)) % total_pts
        spawn_x, spawn_y = track.drawing_points[pt_idx]
        
        next_idx = (pt_idx + 5) % total_pts
        p_curr = track.drawing_points[pt_idx]
        p_next = track.drawing_points[next_idx]
        local_angle = math.degrees(math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0]))
        
        new_car = Car(car_id, team_id, car_id % CARS_PER_TEAM, (spawn_x, spawn_y), local_angle, color)
        new_car.cast_rays(track)
        
        new_car.sector_timer_start = 0
        new_car.sector_bonus_reward = 0.0
        new_car.checkpoint_bonus = 0.0
        
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
                cp_dist = abs(cars[i].next_checkpoint - cars[j].next_checkpoint)
                cp_dist = min(cp_dist, num_checkpoints - cp_dist)
                
                if cp_dist <= 2:
                    cars[i].alive = False
                    cars[j].alive = False

session_timer = 0
generation_count = 1
race_finished = False
winner_info = ""

best_lap_time_current_track = float('inf')
best_s1_time = float('inf')
best_s2_time = float('inf')
best_s3_time = float('inf')

while True:
    dt = clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN and state == "DRAWING" and len(track.drawing_points) > 20:
                
                # 1. AUTOCOMPLETAMENTO TRACCIATO (In Bianco Puro)
                p_first = track.drawing_points[0]
                p_last = track.drawing_points[-1]
                dist = math.hypot(p_first[0] - p_last[0], p_first[1] - p_last[1])
                steps = max(1, int(dist / 5)) 
                
                for i in range(1, steps + 1):
                    frac = i / steps
                    nx = p_last[0] + (p_first[0] - p_last[0]) * frac
                    ny = p_last[1] + (p_first[1] - p_last[1]) * frac
                    pygame.draw.circle(track.surface, (255, 255, 255), (int(nx), int(ny)), track.brush_radius)
                    track.drawing_points.append((int(nx), int(ny)))
                
                track.close_track()
                track.generate_checkpoints()
                
                # 2. RICAMPIONAMENTO UNIFORME BASATO SU NUM_CHECKPOINTS
                total_dist = 0
                dists = [0]
                pts = track.drawing_points
                for i in range(1, len(pts)):
                    d = math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
                    total_dist += d
                    dists.append(total_dist)
                
                interval = total_dist / float(NUM_CHECKPOINTS)
                new_cps = []
                for i in range(NUM_CHECKPOINTS):
                    target = i * interval
                    for j in range(len(dists)-1):
                        if dists[j] <= target <= dists[j+1]:
                            frac = (target - dists[j]) / (dists[j+1] - dists[j]) if dists[j+1] != dists[j] else 0
                            x = pts[j][0] + frac * (pts[j+1][0] - pts[j][0])
                            y = pts[j][1] + frac * (pts[j+1][1] - pts[j][1])
                            new_cps.append((x, y))
                            break
                
                while len(new_cps) < NUM_CHECKPOINTS:
                    new_cps.append(pts[-1])
                
                track.checkpoints = new_cps
                
                state = "RACING"
                initialize_race_session()
                session_timer = 0
                race_finished = False
            
            elif state == "POST_RACE":
                if event.key == pygame.K_r:
                    track = Track(WIDTH, HEIGHT)
                    state = "DRAWING"
                    best_lap_time_current_track = float('inf')
                    best_s1_time = float('inf')
                    best_s2_time = float('inf')
                    best_s3_time = float('inf')
                elif event.key == pygame.K_SPACE:
                    state = "RACING"
                    initialize_race_session()
                    session_timer = 0
                    race_finished = False

            if state == "DRAWING":
                if event.key == pygame.K_UP:
                    track.brush_radius = min(80, track.brush_radius + 5)
                elif event.key == pygame.K_DOWN:
                    track.brush_radius = max(12, track.brush_radius - 5)

    if state == "DRAWING":
        track.draw_step(screen)
    
    elif state == "RACING":
        session_timer += dt
        screen.blit(track.surface, (0, 0))
        
        if len(track.drawing_points) > 0:
            p_start = track.drawing_points[0]
            rad_angle = math.radians(track.start_angle + 90)
            x1 = p_start[0] + math.cos(rad_angle) * track.brush_radius
            y1 = p_start[1] + math.sin(rad_angle) * track.brush_radius
            x2 = p_start[0] - math.cos(rad_angle) * track.brush_radius
            y2 = p_start[1] - math.sin(rad_angle) * track.brush_radius
            pygame.draw.line(screen, (220, 53, 69), (x1, y1), (x2, y2), 5)
            
            # DISEGNO CHECKPOINT DINAMICI DEI SETTORI S1 E S2 (Celeste)
            if len(track.checkpoints) == NUM_CHECKPOINTS:
                for cp_idx in [IDX_S1, IDX_S2]:
                    cp = track.checkpoints[cp_idx]
                    prev_cp = track.checkpoints[cp_idx - 1]
                    angle = math.atan2(cp[1] - prev_cp[1], cp[0] - prev_cp[0])
                    rad_cp_angle = angle + math.pi/2
                    cx1 = cp[0] + math.cos(rad_cp_angle) * track.brush_radius
                    cy1 = cp[1] + math.sin(rad_cp_angle) * track.brush_radius
                    cx2 = cp[0] - math.cos(rad_cp_angle) * track.brush_radius
                    cy2 = cp[1] - math.sin(rad_cp_angle) * track.brush_radius
                    pygame.draw.line(screen, (0, 255, 255), (cx1, cy1), (cx2, cy2), 4)
        
        slipstream_flags = {car.car_id: False for car in cars if car.alive}
        for car_a in cars:
            if not car_a.alive: continue
            for car_b in cars:
                if not car_b.alive or car_a.car_id == car_b.car_id: continue
                
                dx = car_b.x - car_a.x
                dy = car_b.y - car_a.y
                dist = math.hypot(dx, dy)
                
                if 15.0 < dist < 90.0:
                    heading = math.degrees(math.atan2(dy, dx)) % 360
                    angle_diff = abs(heading - car_a.angle)
                    if angle_diff > 180: angle_diff = 360 - angle_diff
                    
                    if angle_diff < 22.5:
                        slipstream_flags[car_a.car_id] = True
                        break

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
                car.update(actions[car.car_id], track, slipstream_flags.get(car.car_id, False))
                
                if track.checkpoints:
                    target_cp = track.checkpoints[car.next_checkpoint]
                    if math.hypot(car.x - target_cp[0], car.y - target_cp[1]) < track.brush_radius * 1.3:
                        
                        # Assegnazione del reward bilanciato
                        car.checkpoint_bonus = BASE_CHECKPOINT_REWARD / NUM_CHECKPOINTS
                        
                        # MONITORAGGIO TEMPI SETTORI
                        if car.next_checkpoint == IDX_S1:
                            s1_time = (session_timer - car.sector_timer_start) / 1000.0
                            car.sector_timer_start = session_timer
                            if s1_time < best_s1_time:
                                best_s1_time = s1_time
                                car.sector_bonus_reward = 0.0  # Sistema pronto per futuri reward
                        
                        elif car.next_checkpoint == IDX_S2:
                            s2_time = (session_timer - car.sector_timer_start) / 1000.0
                            car.sector_timer_start = session_timer
                            if s2_time < best_s2_time:
                                best_s2_time = s2_time
                                car.sector_bonus_reward = 0.0 
                        
                        car.next_checkpoint += 1
                        if car.next_checkpoint >= len(track.checkpoints):
                            
                            s3_time = (session_timer - car.sector_timer_start) / 1000.0
                            car.sector_timer_start = session_timer
                            if s3_time < best_s3_time:
                                best_s3_time = s3_time
                                car.sector_bonus_reward = 0.0 
                            
                            car.next_checkpoint = 0
                            
                            lap_time_ms = session_timer - car.last_lap_timer
                            car.last_lap_timer = session_timer
                            car.laps += 1
                            lap_time_s = max(0.5, lap_time_ms / 1000.0)
                            
                            if best_lap_time_current_track == float('inf'):
                                car.lap_bonus_reward = 1000.0
                                best_lap_time_current_track = lap_time_s
                            else:
                                car.lap_bonus_reward = 1000.0 * (best_lap_time_current_track / lap_time_s)
                                if lap_time_s < best_lap_time_current_track:
                                    best_lap_time_current_track = lap_time_s
                            
                            if car.laps >= TARGET_LAPS:
                                race_finished = True
                                winner_info = f"Vettura {car.car_id} ({TEAM_ALGORITHMS[car.team_id]})"                

                car.draw(screen)

        check_car_to_car_collisions()

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
                
                if car.alive:
                    # Recupera lo stato della scia per calcolare la top speed reale di quel frame
                    has_tow = slipstream_flags.get(car.car_id, False)
                    current_top_speed = MAX_SPEED_PX * 1.15 if has_tow else MAX_SPEED_PX
                    
                    # 1. Reward quadratico basato sulla velocità px/s reale della vettura
                    r += (car.speed_px ** 2) / 75.0  
                    
                    # 2. Bonus spinto quando l'auto spinge oltre il 95% della sua velocità limite attuale
                    if car.speed_px >= current_top_speed * 0.95:
                        r += 15.0  # Premiazione costante frame-by-frame per la permanenza in V-max
                        
                    r -= 0.5  # Step penalty costante contro lo stallo e le rotazioni vuote

                if not car.alive and not car.crash_triggered:
                    car.crash_triggered = True
                    laps_missing = max(0, TARGET_LAPS - car.laps)
                    remancy_ratio = laps_missing / TARGET_LAPS
                    r -= (400.0 + 400.0 * remancy_ratio)
                    done = True
                else:
                    done = not car.alive

                if car.lap_bonus_reward > 0:
                    r += car.lap_bonus_reward
                    car.lap_bonus_reward = 0.0  
                
                if hasattr(car, 'checkpoint_bonus') and car.checkpoint_bonus > 0:
                    r += car.checkpoint_bonus
                    car.checkpoint_bonus = 0.0
                    
                if hasattr(car, 'sector_bonus_reward') and car.sector_bonus_reward > 0:
                    r += car.sector_bonus_reward
                    car.sector_bonus_reward = 0.0
                
                if not hasattr(car, 'backwards_timer'): car.backwards_timer = 0.0
                if not hasattr(car, 'distance_at_backwards_start'): car.distance_at_backwards_start = car.distance_traveled
                if not hasattr(car, 'dsq_triggered'): car.dsq_triggered = False

                if track.checkpoints and car.alive:
                    target_cp = track.checkpoints[car.next_checkpoint]
                    dx = target_cp[0] - car.x
                    dy = target_cp[1] - car.y
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        target_dir_x = dx / dist
                        target_dir_y = dy / dist
                        car_dir_x = math.cos(math.radians(car.angle))
                        car_dir_y = math.sin(math.radians(car.angle))
                        
                        if (target_dir_x * car_dir_x + target_dir_y * car_dir_y) < -0.2:
                            r -= 60.0
                            if car.backwards_timer == 0.0:
                                car.distance_at_backwards_start = car.distance_traveled
                            car.backwards_timer += dt
                            
                            if car.backwards_timer >= 3000:
                                car.alive = False
                                car.dsq_triggered = True
                                car.distance_traveled = car.distance_at_backwards_start
                                done = True
                        else:
                            car.backwards_timer = 0.0

                if slipstream_flags.get(car.car_id, False) and not done:
                    r += 10.0 

                if algo == "Q_LEARNING":
                    agent.update(agent.discretize(s), a, r, agent.discretize(s_prime))
                elif algo == "DQN":
                    agent.store_transition(s, a, r, s_prime, done)
                    agent.train_step()
                elif algo == "POLICY_GRADIENT":
                    agent.rewards.append(r)

        # Ordinamento deterministico basato sul progresso effettivo dei checkpoint
        def get_race_progress(c):
            if not track.checkpoints:
                return 0
            # Sincronizza l'indice per evitare out-of-bound isolati
            cp_idx = c.next_checkpoint % len(track.checkpoints)
            target_cp = track.checkpoints[cp_idx]
            dist_to_cp = math.hypot(c.x - target_cp[0], c.y - target_cp[1])
            
            # Valore principale: totale dei checkpoint superati nella sessione
            total_checkpoints_passed = c.laps * len(track.checkpoints) + c.next_checkpoint
            
            # Sottraiamo la distanza mancante come tie-breaker per premiare chi è più vicino al prossimo checkpoint
            return total_checkpoints_passed * 10000 - dist_to_cp

        sorted_cars = sorted(cars, key=get_race_progress, reverse=True)
        
        panel_x = WIDTH - 260
        pygame.draw.rect(screen, (20, 20, 20), (panel_x, 10, 250, 260))
        pygame.draw.rect(screen, (70, 70, 70), (panel_x, 10, 250, 260), 2)
        
        font_title = pygame.font.SysFont("Arial", 12, bold=True)
        screen.blit(font_title.render("CLASSIFICA LIVE / GIRI RIM.", True, (255, 255, 255)), (panel_x + 10, 16))
        
        font_lb = pygame.font.SysFont("Arial", 10)
        for rank, c in enumerate(sorted_cars[:10]):
            laps_remaining = max(0, TARGET_LAPS - c.laps)
            
            if getattr(c, 'dsq_triggered', False):
                status_text = "DSQ"
            elif not c.alive:
                status_text = "DNF"
            else:
                status_text = f"{int(c.speed_px)} px/s"
            
            if slipstream_flags.get(c.car_id, False) and c.alive:
                status_text += " +TOW"
                
            info_str = f"{rank+1}. ID {c.car_id:02d} ({TEAM_ALGORITHMS[c.team_id][:4]}): {laps_remaining} laps to go | {status_text}"
            color = c.color if c.alive else (110, 110, 110)
            screen.blit(font_lb.render(info_str, True, color), (panel_x + 10, 42 + rank * 21))

        active_cars = sum(1 for car in cars if car.alive)
        font = pygame.font.SysFont("Arial", 16)
        
        track_record_str = f"{best_lap_time_current_track:.2f}s" if best_lap_time_current_track != float('inf') else "N/D"
        str_s1 = f"{best_s1_time:.2f}s" if best_s1_time != float('inf') else "N/D"
        str_s2 = f"{best_s2_time:.2f}s" if best_s2_time != float('inf') else "N/D"
        str_s3 = f"{best_s3_time:.2f}s" if best_s3_time != float('inf') else "N/D"
        
        ui_text = font.render(f"Simulazione {generation_count}/10 | Attive: {active_cars}/20 | Tempo: {session_timer//1000}s", True, (255, 255, 0))
        record_text = font.render(f"Lap: {track_record_str} | S1: {str_s1} | S2: {str_s2} | S3: {str_s3}", True, (0, 255, 255))
        
        screen.blit(ui_text, (20, 20))
        screen.blit(record_text, (20, 45))
        
        if race_finished or active_cars == 0 or session_timer >= SESSION_DURATION_MS:
            if active_cars == 0:
                print(f"--- SESSIONE {generation_count} TERMINATA: TUTTE LE AUTO ELIMINATE ---")
            if session_timer >= SESSION_DURATION_MS:
                print(f"--- SESSIONE {generation_count} TERMINATA PER TIMEOUT ---")
            elif race_finished:
                print(f"--- SESSIONE {generation_count} TERMINATA: {winner_info} vince! ---")
            
            for car in cars:
                if car.laps < TARGET_LAPS:
                    agent = agents[car.car_id]
                    algo = TEAM_ALGORITHMS[car.team_id]
                    s = car.sensor_distances.copy()
                    
                    timeout_penalty = -700.0 - (TARGET_LAPS - car.laps) * 200.0
                    
                    if car.alive:
                        if algo == "DQN":
                            agent.store_transition(s, 0, timeout_penalty, s, True)
                            agent.train_step()
                        elif algo == "Q_LEARNING":
                            agent.update(agent.discretize(s), 0, timeout_penalty, agent.discretize(s))
                        elif algo == "POLICY_GRADIENT":
                            agent.rewards.append(timeout_penalty)
            
            genetic_car_ids = [c.car_id for c in cars if TEAM_ALGORITHMS[c.team_id] == "GENETIC"]
            if genetic_car_ids:
                num_cps = len(track.checkpoints) if track.checkpoints else 1
                best_genetic_id = max(genetic_car_ids, key=lambda cid: (cars[cid].laps * num_cps + cars[cid].next_checkpoint))
                
                best_net = agents[best_genetic_id]
                for cid in genetic_car_ids:
                    if cid != best_genetic_id:
                        agents[cid].W1 = best_net.W1.copy()
                        agents[cid].W2 = best_net.W2.copy()
                        agents[cid].mutate()

            for car_id, agent in agents.items():
                if TEAM_ALGORITHMS[cars[car_id].team_id] == "POLICY_GRADIENT":
                    agent.update_policy()
                
            generation_count += 1
            if generation_count > 10:
                pygame.quit()
                sys.exit()
                
            state = "POST_RACE"

    elif state == "POST_RACE":
        screen.blit(track.surface, (0, 0))
        pygame.draw.rect(screen, (15, 15, 15), (WIDTH//2 - 250, HEIGHT//2 - 70, 500, 140))
        pygame.draw.rect(screen, (255, 255, 255), (WIDTH//2 - 250, HEIGHT//2 - 70, 500, 140), 2)
        
        f_menu = pygame.font.SysFont("Arial", 18, bold=True)
        txt_gen = f_menu.render(f"GENERAZIONE {generation_count - 1} COMPLETATA", True, (0, 255, 0))
        txt_opt1 = f_menu.render("Premi R per RIDISEGNARE un nuovo tracciato", True, (255, 255, 255))
        txt_opt2 = f_menu.render("Premi SPAZIO per RIPARTIRE sulla stessa pista", True, (255, 255, 255))
        
        screen.blit(txt_gen, (WIDTH//2 - 140, HEIGHT//2 - 45))
        screen.blit(txt_opt1, (WIDTH//2 - 210, HEIGHT//2 - 5))
        screen.blit(txt_opt2, (WIDTH//2 - 210, HEIGHT//2 + 25))

    pygame.display.flip()



# Problemi e miglioramenti:
# 1) le auto hanno troppa paura dei limiti del tracciato laterali che non raggiungono mai la velocità massima o, a velocità elevate, tendono a non seguire la traiettoria ideale, ma si spostano verso il centro del tracciato, perdendo tempo prezioso.
# 2) le auto prioritizzano la distanza percorsa all'avanzamento nei ceckpoint, con conseguente rotazione su se stessi perenne (consentita forse da un numero di checkpoint troppo elevato, trigger non attivato)
# 2.2) le auto che ruotano su sé stesse forse superano checkpoint andando contromano. per la singola auto l'unico checkpoint attivo da raggiungere è quello successivo e non quelli che dovranno superare dopo aver superato il successivo, altrimenti si rischia di avere auto che ruotano su se stesse per ore senza mai superare un checkpoint, ma solo avvicinandosi a quello più vicino (che però non è quello attivo da superare) e quindi accumulando reward per la distanza percorsa senza mai avanzare realmente nel tracciato.
# 3) le vetture non sembrano superare le vetture avversarie, bensì, attraverso la scia, seguono la vettura precedente solo per poi finire contro di essa, terminando per essere entrambe eliminare
# 4) vanno migliorati anche i singoli algoritmi per renderli più competitivi e meno propensi a rimanere bloccati in situazioni di stallo o rotazione su se stessi
# 5) l'effetto scia deve continuare anche mentre l'auto tenta il sorpasso e la percentuale di velocità deve diminuire come da resistenza dell'aria (effetto scia scompare lentamente)
# 6) le auto che terminano la gara completando tutti i giri lo fanno prima della linea del traguardo