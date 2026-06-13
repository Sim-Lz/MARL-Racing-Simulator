# main.py
import subprocess
import sys

try:
    import pygame
    import numpy as np
    import torch
    import math
    import random
except ImportError:
    print("Dipendenze mancanti rilevate. Avvio l'installazione automatica...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
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

NUM_CHECKPOINTS = 100          
BASE_CHECKPOINT_REWARD = 200.0  
IDX_S1 = int(NUM_CHECKPOINTS * 0.33)
IDX_S2 = int(NUM_CHECKPOINTS * 0.67)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MARL Racing Simulator - Full Restored & Scientific Build")
clock = pygame.time.Clock()

track = Track(WIDTH, HEIGHT)
state = "DRAWING"

cars = []
teams = []
agents = {}
global_simulation_stats = []  # Database storico per il report finale (Problema 7)

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
    # Ordinamento opzionale se vuoi assegnare i posti in griglia in base ai tempi, al momento casuale
    random.shuffle(shuffled_car_ids)
    
    total_pts = len(track.drawing_points)
    temp_cars = [None] * TOTAL_CARS
    
    for slot, car_id in enumerate(shuffled_car_ids):
        team_id = car_id // CARS_PER_TEAM
        color = TEAM_COLORS[team_id]
        
        # Problema 3: Formazione Griglia F1 (Sfalsata su 2 Colonne)
        row = slot // 2
        col = slot % 2
        
        pt_idx = (total_pts - 1 - (row * 12)) % total_pts
        p_curr = track.drawing_points[pt_idx]
        p_next = track.drawing_points[(pt_idx + 5) % total_pts]
        
        # Vettori per l'orientamento
        angle = math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0])
        nx = -math.sin(angle)
        ny = math.cos(angle)
        
        # Offset laterale: colonna 0 a destra, colonna 1 a sinistra rispetto al centro
        offset = 15 if col == 0 else -15
        
        spawn_x = p_curr[0] + nx * offset
        spawn_y = p_curr[1] + ny * offset
        local_angle = math.degrees(angle)
        
        new_car = Car(car_id, team_id, car_id % CARS_PER_TEAM, (spawn_x, spawn_y), local_angle, color)
        new_car.cast_rays(track)
        
        new_car.sector_timer_start = 0
        new_car.sector_bonus_reward = 0.0
        
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

def print_mini_report(gen_num, car_list):
    print(f"\n--- SINTESI GEN {gen_num} COMPLETATA ---")
    active = sum(1 for c in car_list if c.alive or c.laps >= TARGET_LAPS)
    print(f"Auto al traguardo/attive: {active}/{TOTAL_CARS}. Avvio post-elaborazione...")

def print_grand_final_report():
    print(f"\n==========================================================================================")
    print(f"🔬 GRAND SCIENTIFIC REPORT: METRICHE AGGREGATE POST-TRAINING (10 GENERAZIONI)")
    print(f"==========================================================================================")
    
    algo_metrics = {algo: {'aes': [], 'speed': [], 'reward': [], 'overtakes': []} for algo in set(TEAM_ALGORITHMS)}
    
    for stat in global_simulation_stats:
        algo_metrics[stat['algo']]['aes'].append(stat['aes'])
        algo_metrics[stat['algo']]['speed'].append(stat['max_speed'])
        algo_metrics[stat['algo']]['reward'].append(stat['cumulative_reward'])
        algo_metrics[stat['algo']]['overtakes'].append(stat['overtakes'])
        
    print(f"| ALGORITMO       | ACCURACY (AES) | V-MAX MEDIA | SORPASSI TOT | REWARD MEDIO FINALE |")
    print(f"------------------------------------------------------------------------------------------")
    for algo, metrics in algo_metrics.items():
        if not metrics['aes']: continue
        mean_aes = np.mean(metrics['aes'])
        mean_speed = np.mean(metrics['speed'])
        tot_overtakes = np.sum(metrics['overtakes'])
        mean_reward = np.mean(metrics['reward'])
        print(f"| {algo:<15} | {mean_aes:>13.2f}% | {int(mean_speed):>7} px/s | {tot_overtakes:>12} | {mean_reward:>19.1f} |")
    print(f"==========================================================================================\n")

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
                track.close_track()
                
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
                
                # Incrementata la distanza massima di scia a 120 (Problema 5)
                if 15.0 < dist < 120.0:
                    heading = math.degrees(math.atan2(dy, dx)) % 360
                    angle_diff = abs(heading - car_a.angle)
                    if angle_diff > 180: angle_diff = 360 - angle_diff
                    
                    if angle_diff < 22.5:
                        slipstream_flags[car_a.car_id] = True
                        if car_a.next_checkpoint > car_b.next_checkpoint and hasattr(car_a, 'was_behind_last_frame') and car_a.was_behind_last_frame == car_b.car_id:
                            car_a.overtakes_performed += 1
                        car_a.was_behind_last_frame = car_b.car_id
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

        individual_raw_rewards = {}
        cars_alive_snapshot = {c.car_id: c.alive for c in cars}

        for car in cars:
            if not car.alive:
                individual_raw_rewards[car.car_id] = -15.0
                continue
                
            r = 0.0
            car.update(actions[car.car_id], track, slipstream_flags.get(car.car_id, False))
            
            target_cp = track.checkpoints[car.next_checkpoint]
            cx_dir = target_cp[0] - car.x
            cy_dir = target_cp[1] - car.y
            c_dist = math.hypot(cx_dir, cy_dir)
            
            alignment_factor = 0.0
            if c_dist > 0:
                cx_dir /= c_dist
                cy_dir /= c_dist
                car_x_dir = math.cos(math.radians(car.angle))
                car_x_dir_y = math.sin(math.radians(car.angle))
                alignment_factor = cx_dir * car_x_dir + cy_dir * car_x_dir_y

            if track.checkpoints:
                if c_dist < track.brush_radius * 1.3:
                    # Problema 6: Reward Checkpoint sommate istantaneamente 
                    r += BASE_CHECKPOINT_REWARD / NUM_CHECKPOINTS
                    car.checkpoints_passed_total += 1
                    
                    if car.next_checkpoint == IDX_S1:
                        s1_time = (session_timer - car.sector_timer_start) / 1000.0
                        car.sector_timer_start = session_timer
                        if s1_time < best_s1_time: best_s1_time = s1_time
                    elif car.next_checkpoint == IDX_S2:
                        s2_time = (session_timer - car.sector_timer_start) / 1000.0
                        car.sector_timer_start = session_timer
                        if s2_time < best_s2_time: best_s2_time = s2_time
                    
                    car.next_checkpoint += 1
                    
                    # Problema 2: Ripristinato calcolo settore S3 originale
                    if car.next_checkpoint >= len(track.checkpoints):
                        car.next_checkpoint = 0
                        
                        s3_time = (session_timer - car.sector_timer_start) / 1000.0
                        car.sector_timer_start = session_timer
                        if s3_time < best_s3_time: best_s3_time = s3_time
                        
                        car.laps += 1
                        
                        lap_time_ms = session_timer - car.last_lap_timer
                        car.last_lap_timer = session_timer
                        lap_time_s = max(0.5, lap_time_ms / 1000.0)
                        
                        if best_lap_time_current_track == float('inf'):
                            r += 1000.0
                            best_lap_time_current_track = lap_time_s
                        else:
                            r += 1000.0 * (best_lap_time_current_track / lap_time_s)
                            if lap_time_s < best_lap_time_current_track:
                                best_lap_time_current_track = lap_time_s
                        
                        if car.laps >= TARGET_LAPS:
                            race_finished = True
                            winner_info = f"Vettura {car.car_id} ({TEAM_ALGORITHMS[car.team_id]})"

            forward_velocity_incentive = max(0.0, alignment_factor)
            r += ((car.speed_px ** 2) / 75.0) * forward_velocity_incentive
            
            has_tow = slipstream_flags.get(car.car_id, False)
            current_top_speed = MAX_SPEED_PX * 1.15 if has_tow else MAX_SPEED_PX
            if car.speed_px >= current_top_speed * 0.95:
                r += 15.0
                
            r -= 0.5  
            
            if has_tow and alignment_factor > 0.9:
                for target_car in cars:
                    if target_car.alive and target_car.car_id != car.car_id and target_car.team_id != car.team_id:
                        if math.hypot(target_car.x - car.x, target_car.y - car.y) < 45.0:
                            r -= 3.0  

            if alignment_factor < -0.2:
                r -= 60.0
                if not hasattr(car, 'backwards_timer'): car.backwards_timer = 0.0
                car.backwards_timer += dt
                if car.backwards_timer >= 3000:
                    car.alive = False
                    car.dsq_triggered = True
                    r -= 500.0
            else:
                car.backwards_timer = 0.0

            if has_tow:
                r += 10.0

            individual_raw_rewards[car.car_id] = r
            car.draw(screen)

        check_car_to_car_collisions()

        for team in teams:
            team_rewards = team.compute_cooperative_rewards(individual_raw_rewards, cars_alive_snapshot)
            for car in team.cars:
                if car.car_id not in current_states: continue
                
                agent = agents[car.car_id]
                algo = TEAM_ALGORITHMS[car.team_id]
                s = current_states[car.car_id]
                s_prime = car.sensor_distances.copy()
                a = actions[car.car_id]
                
                final_r = team_rewards[car.car_id]
                
                if not car.alive and not car.crash_triggered:
                    car.crash_triggered = True
                    laps_missing = max(0, TARGET_LAPS - car.laps)
                    final_r -= (400.0 + 400.0 * (laps_missing / TARGET_LAPS))
                    done = True
                else:
                    done = not car.alive

                car.cumulative_reward += final_r  # Tracking continuo reward

                if algo == "Q_LEARNING":
                    agent.update(agent.discretize(s), a, final_r, agent.discretize(s_prime))
                elif algo == "DQN":
                    agent.store_transition(s, a, final_r, s_prime, done)
                    agent.train_step()
                elif algo == "POLICY_GRADIENT":
                    agent.rewards.append(final_r)

        def get_race_progress(c):
            if not track.checkpoints: return 0
            cp_idx = c.next_checkpoint % len(track.checkpoints)
            target_cp = track.checkpoints[cp_idx]
            dist_to_cp = math.hypot(c.x - target_cp[0], c.y - target_cp[1])
            total_checkpoints_passed = c.laps * len(track.checkpoints) + c.next_checkpoint
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
            if getattr(c, 'dsq_triggered', False): status_text = "DSQ"
            elif not c.alive: status_text = "DNF"
            else: status_text = f"{int(c.speed_px)} px/s"
            if slipstream_flags.get(c.car_id, False) and c.alive: status_text += " +TOW"
                
            info_str = f"{rank+1}. ID {c.car_id:02d} ({TEAM_ALGORITHMS[c.team_id][:4]}): {laps_remaining} laps | {status_text}"
            screen.blit(font_lb.render(info_str, True, c.color if c.alive else (110, 110, 110)), (panel_x + 10, 42 + rank * 21))

        active_cars = sum(1 for car in cars if car.alive)
        font = pygame.font.SysFont("Arial", 16)
        
        track_record_str = f"{best_lap_time_current_track:.2f}s" if best_lap_time_current_track != float('inf') else "N/D"
        ui_text = font.render(f"Simulazione {generation_count}/10 | Attive: {active_cars}/20 | Tempo: {session_timer//1000}s", True, (255, 255, 0))
        record_text = font.render(f"Lap Record: {track_record_str} | S1: {best_s1_time:.2f}s | S2: {best_s2_time:.2f}s | S3: {best_s3_time:.2f}s", True, (0, 255, 255))
        screen.blit(ui_text, (20, 20))
        screen.blit(record_text, (20, 45))
        
        if race_finished or active_cars == 0 or session_timer >= SESSION_DURATION_MS:
            print_mini_report(generation_count, cars)
            
            for car in cars:
                # Salvataggio Metriche per il Report Finale
                accuracy_aes = (car.checkpoints_passed_total / max(1, car.total_steps)) * 100.0
                global_simulation_stats.append({
                    'gen': generation_count,
                    'car_id': car.car_id,
                    'algo': TEAM_ALGORITHMS[car.team_id],
                    'aes': accuracy_aes,
                    'max_speed': car.max_speed_reached,
                    'cumulative_reward': car.cumulative_reward,
                    'overtakes': car.overtakes_performed
                })

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
                
            if generation_count == 10:
                print_grand_final_report()
                pygame.quit()
                sys.exit()
                
            generation_count += 1
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