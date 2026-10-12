# main.py
import sys

try:
    import pygame
    import numpy as np
    import torch
    import math
    import random
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Dipendenza Python mancante ({exc.name}). "
        f"Installa i pacchetti del progetto con "
        f'"{sys.executable}" -m pip install -r requirements.txt.'
    ) from exc

from config import *
from track import Track
from car import Car
from team import Team

from rl_models.heuristic import HeuristicController
from rl_models.genetic import GeneticNetwork
from rl_models.qlearning import TabularQLearner
from rl_models.dqn import DQNAgent
from rl_models.policy_gradient import PolicyGradientAgent
from rl_models.rewards import calculate_step_reward
from race_logger import RaceLogger
from race_flow import should_auto_start_next_race, should_exit_early

NUM_CHECKPOINTS = 50
IDX_S1 = int(NUM_CHECKPOINTS * 0.33)
IDX_S2 = int(NUM_CHECKPOINTS * 0.67)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MARL Racing Simulator - Gradient Tracker & High-Fidelity Reports")
clock = pygame.time.Clock()

track = Track(WIDTH, HEIGHT)
state = "DRAWING"

cars = []
teams = []
agents = {}
global_simulation_stats = [] 

# Struttura dedicata al tracciamento storico dei gradienti per generazione
global_gradient_registry = {
    "POLICY_GRADIENT": [],
    "DQN": []
}

RUN_SEED = random.SystemRandom().randrange(2**32)
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)
torch.manual_seed(RUN_SEED)
race_logger = RaceLogger()

for car_id in range(TOTAL_CARS):
    team_id = car_id // CARS_PER_TEAM
    algo = TEAM_ALGORITHMS[team_id]
    if algo == "HEURISTIC": agents[car_id] = HeuristicController()
    elif algo == "GENETIC": agents[car_id] = GeneticNetwork()
    elif algo == "Q_LEARNING": agents[car_id] = TabularQLearner()
    elif algo == "DQN": agents[car_id] = DQNAgent()
    elif algo == "POLICY_GRADIENT": agents[car_id] = PolicyGradientAgent()

def get_gradient_norm(agent):
    """Calcola la norma L2 dei gradienti per i modelli basati su PyTorch."""
    total_norm = 0.0
    # Ispezione dinamica dell'architettura dell'agente passato
    model_attr = None
    if hasattr(agent, 'model'): model_attr = agent.model
    elif hasattr(agent, 'policy_net'): model_attr = agent.policy_net
    elif hasattr(agent, 'policy'): model_attr = agent.policy
    
    if model_attr and isinstance(model_attr, torch.nn.Module):
        for p in model_attr.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    return 0.0 # Ritorna 0 per Heuristic, Genetic, Tabular Q-Learning (Non basati su backpropagation diretta)

def initialize_race_session():
    global cars, teams, best_lap_time_current_race
    cars.clear()
    teams.clear()
    best_lap_time_current_race = float('inf')
    
    shuffled_car_ids = list(range(TOTAL_CARS))
    random.shuffle(shuffled_car_ids)
    
    total_pts = len(track.drawing_points)
    temp_cars = [None] * TOTAL_CARS
    
    for slot, car_id in enumerate(shuffled_car_ids):
        team_id = car_id // CARS_PER_TEAM
        color = TEAM_COLORS[team_id]
        
        row = slot // 2
        col = slot % 2
        
        pt_idx = (total_pts - 1 - (row * 12)) % total_pts
        p_curr = track.drawing_points[pt_idx]
        p_next = track.drawing_points[(pt_idx + 5) % total_pts]
        
        angle = math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0])
        nx = -math.sin(angle)
        ny = math.cos(angle)
        
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

def start_next_race():
    global state, session_timer, race_finished, post_race_elapsed_ms
    initialize_race_session()
    session_timer = 0
    race_finished = False
    post_race_elapsed_ms = 0
    state = "RACING"

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
                    cars[i].termination_reason = "COLLISION"
                    cars[j].termination_reason = "COLLISION"

def print_deep_simulation_report(gen_num, car_list, reason):
    total_sim_checkpoints = NUM_CHECKPOINTS * TARGET_LAPS
    
    print(f"\n=====================================================================================================================")
    print(f"REPORT SIMULAZIONE N° {gen_num} | MOTIVAZIONE CHIUSURA: {reason}")
    print(f"=====================================================================================================================")
    print(f"| ID | ALGORITMO       | STATO       | GIRI COMPL. | CHECKPOINTS | V-MAX     | T-SURVIVAL | REWARD TOT | GRAD. L2  |")
    print(f"---------------------------------------------------------------------------------------------------------------------")
    for c in sorted(car_list, key=lambda x: x.car_id):
        g_norm = get_gradient_norm(agents[c.car_id])
        g_str = f"{g_norm:.4f}" if g_norm > 0 else "N/A"
        status = "COMPLETED" if c.laps >= TARGET_LAPS else c.termination_reason
        
        # Stampa riga con inclusione del tempo di sopravvivenza formattato a due cifre decimali
        print(f"| {c.car_id:02d} | {TEAM_ALGORITHMS[c.team_id]:<15} | {status:<11} | {c.laps}/{TARGET_LAPS}        | {c.checkpoints_passed_total:04d}/{total_sim_checkpoints:04d} | {int(c.max_speed_reached):>3} px/s | {c.survival_time_s:>9.2f}s | {c.cumulative_reward:>10.1f} | {g_str:<9} |")
    print(f"=====================================================================================================================\n")
    
def print_grand_final_report():
    print(f"\n==========================================================================================")
    print(f"GRAND SCIENTIFIC REPORT: METRICHE AGGREGATE FINALI ({generation_count} GENERAZIONI)")
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

    print("\n" + "="*60)
    print("RELAZIONE METRICHE DI AGGIORNAMENTO E STABILITÀ DEI GRADIENTI")
    print("="*60)
    
    for algorithm_name, history in global_gradient_registry.items():
        print(f"\nAlgoritmo: {algorithm_name}")
        if not history or sum(history) == 0:
            print("  Nessun dato di gradiente registrato o ottimizzazione non eseguita.")
            continue
            
        for gen_idx, grad_val in enumerate(history):
            print(f"  Generazione {gen_idx + 1}: Norma Media del Gradiente = {grad_val:.6f}")
            
        # Calcolo di un indicatore di stabilità basato sulla varianza del gradiente
        variance = np.var(history)
        print(f"  Varianza Storica della Norma: {variance:.8f}")
        if variance > 10.0:
            print("  Nota: Rilevata un'elevata fluttuazione dei gradienti. Valutare l'inserimento di Gradient Clipping.")

session_timer = 0
generation_count = 1
race_finished = False
winner_info = ""

best_lap_time_current_track = float('inf')
best_s1_time = float('inf')
best_s2_time = float('inf')
best_s3_time = float('inf')
best_lap_time_current_race = float('inf')
post_race_elapsed_ms = 0

while True:
    dt = clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if should_exit_early(state, event.key, pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

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
                    start_next_race()

            if state == "DRAWING":
                if event.key == pygame.K_UP:
                    track.brush_radius = min(80, track.brush_radius + 5)
                elif event.key == pygame.K_DOWN:
                    track.brush_radius = max(12, track.brush_radius - 5)

    if state == "POST_RACE":
        post_race_elapsed_ms += dt
        if should_auto_start_next_race(state, post_race_elapsed_ms):
            start_next_race()

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

        current_states = {
            car.car_id: car.get_observation(track)
            for car in cars
            if car.alive
        }
        progress_before = {
            car.car_id: car.race_progress(track)
            for car in cars
            if car.alive
        }
        actions = {}
        
        for car in cars:
            if not car.alive: continue
            car.survival_time_s += 1.0 / FPS
            agent = agents[car.car_id]
            algo = TEAM_ALGORITHMS[car.team_id]
            s = current_states[car.car_id]
            
            if algo == "HEURISTIC": actions[car.car_id] = agent.select_action(s)
            elif algo == "GENETIC": actions[car.car_id] = agent.forward(s)
            elif algo == "Q_LEARNING": actions[car.car_id] = agent.select_action(agent.discretize(s))
            elif algo == "DQN": actions[car.car_id] = agent.select_action(s)
            elif algo == "POLICY_GRADIENT": actions[car.car_id] = agent.select_action(s)

        individual_raw_rewards = {}

        for car in cars:
            if not car.alive:
                individual_raw_rewards[car.car_id] = -5.0
                continue
                
            r = 0.0
            car.update(actions[car.car_id], track, slipstream_flags.get(car.car_id, False))
            
            if car.alive and track.checkpoints:
                target_cp = track.checkpoints[car.next_checkpoint]
                c_dist = math.hypot(target_cp[0] - car.x, target_cp[1] - car.y)
                previous_cp = track.checkpoints[(car.next_checkpoint - 1) % len(track.checkpoints)]
                checkpoint_spacing = math.hypot(
                    target_cp[0] - previous_cp[0],
                    target_cp[1] - previous_cp[1],
                )
                checkpoint_radius = max(
                    2.0,
                    min(track.brush_radius * 0.30, checkpoint_spacing * 0.35),
                )
                if c_dist < checkpoint_radius:
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
                    
                    if car.next_checkpoint >= len(track.checkpoints):
                        car.next_checkpoint = 0
                        
                        s3_time = (session_timer - car.sector_timer_start) / 1000.0
                        car.sector_timer_start = session_timer
                        if s3_time < best_s3_time: best_s3_time = s3_time
                        
                        car.laps += 1
                        
                        lap_time_ms = session_timer - car.last_lap_timer
                        car.last_lap_timer = session_timer
                        lap_time_s = max(0.5, lap_time_ms / 1000.0)
                        best_lap_time_current_race = min(
                            best_lap_time_current_race,
                            lap_time_s,
                        )
                        
                        if lap_time_s < best_lap_time_current_track:
                            best_lap_time_current_track = lap_time_s
                        
                        if car.laps >= TARGET_LAPS:
                            race_finished = True
                            winner_info = f"Vettura {car.car_id} ({TEAM_ALGORITHMS[car.team_id]})"
                            car.termination_reason = "COMPLETED"
                            r += calculate_step_reward(0.0, 0, finished=True)
                            car.completion_bonus_total += FINISH_BONUS

            progress_delta = car.race_progress(track) - progress_before[car.car_id]
            progress_reward = PROGRESS_REWARD_SCALE * progress_delta
            time_penalty = TIME_PENALTY_PER_SECOND * dt / 1000.0
            r += calculate_step_reward(progress_delta, dt)
            car.progress_reward_total += progress_reward
            car.time_penalty_total += time_penalty
            car.penalties_accumulated += time_penalty

            if car.get_observation(track)[-1] < -0.2:
                if not hasattr(car, 'backwards_timer'): car.backwards_timer = 0.0
                car.backwards_timer += dt
                if car.backwards_timer >= 2500:
                    car.alive = False
                    car.dsq_triggered = True
                    car.termination_reason = "DSQ_BACK"
            else:
                car.backwards_timer = 0.0

            individual_raw_rewards[car.car_id] = r
            car.draw(screen)

        # Calcola le collisioni auto-auto che modificano lo stato 'alive' delle vetture
        check_car_to_car_collisions()

        current_cars_alive = {c.car_id: c.alive for c in cars}
        just_dead_cars = [
            car for car in cars
            if car.car_id in current_states and not car.alive and not car.crash_triggered
        ]
        for car in just_dead_cars:
            individual_raw_rewards[car.car_id] -= CRASH_PENALTY
            car.terminal_penalty_total += CRASH_PENALTY
            car.crash_triggered = True

        active_cars = sum(1 for car in cars if car.alive)
        session_terminal = (
            race_finished
            or active_cars == 0
            or session_timer >= SESSION_DURATION_MS
        )

        for team in teams:
            team_rewards = team.compute_cooperative_rewards(individual_raw_rewards, current_cars_alive)
            for car in team.cars:
                if car.car_id not in current_states: continue
                
                agent = agents[car.car_id]
                algo = TEAM_ALGORITHMS[car.team_id]
                s = current_states[car.car_id]
                s_prime = car.get_observation(track)
                a = actions[car.car_id]
                
                final_r = team_rewards[car.car_id]
                is_just_dead = car in just_dead_cars
                done = not car.alive or session_terminal

                if car.alive or is_just_dead:
                    car.cumulative_reward += final_r  

                    if algo == "Q_LEARNING":
                        agent.update(
                            agent.discretize(s),
                            a,
                            final_r,
                            agent.discretize(s_prime),
                            done=done,
                        )
                    elif algo == "DQN":
                        agent.store_transition(s, a, final_r, s_prime, done)
                        agent.train_step()
                    elif algo == "POLICY_GRADIENT":
                        agent.rewards.append(final_r)

        sorted_cars = sorted(cars, key=lambda car: car.race_progress(track), reverse=True)
        
        panel_x = WIDTH - 260
        pygame.draw.rect(screen, (20, 20, 20), (panel_x, 10, 250, 260))
        pygame.draw.rect(screen, (70, 70, 70), (panel_x, 10, 250, 260), 2)
        
        font_title = pygame.font.SysFont("Arial", 12, bold=True)
        screen.blit(font_title.render("CLASSIFICA LIVE / GIRI RIM.", True, (255, 255, 255)), (panel_x + 10, 16))
        
        font_lb = pygame.font.SysFont("Arial", 10)
        for rank, c in enumerate(sorted_cars[:10]):
            laps_remaining = max(0, TARGET_LAPS - c.laps)
            if getattr(c, 'dsq_triggered', False): status_text = "DSQ"
            elif not c.alive: status_text = c.termination_reason[:5]
            else: status_text = f"{int(c.speed_px)} px/s"
            if slipstream_flags.get(c.car_id, False) and c.alive: status_text += " +TOW"
                
            info_str = f"{rank+1}. ID {c.car_id:02d} ({TEAM_ALGORITHMS[c.team_id][:4]}): {laps_remaining} laps | {status_text}"
            screen.blit(font_lb.render(info_str, True, c.color if c.alive else (110, 110, 110)), (panel_x + 10, 42 + rank * 21))

        active_cars = sum(1 for car in cars if car.alive)
        font = pygame.font.SysFont("Arial", 16)
        
        track_record_str = f"{best_lap_time_current_track:.2f}s" if best_lap_time_current_track != float('inf') else "N/D"
        ui_text = font.render(f"Simulazione {generation_count}/{NUM_SIMULATIONS} | Attive: {active_cars}/20 | Tempo: {session_timer//1000}s", True, (255, 255, 0))
        record_text = font.render(f"Lap Record: {track_record_str} | S1: {best_s1_time:.2f}s | S2: {best_s2_time:.2f}s | S3: {best_s3_time:.2f}s", True, (0, 255, 255))
        screen.blit(ui_text, (20, 20))
        screen.blit(record_text, (20, 45))
        
        # --- LOGICA DI VALUTAZIONE CONCLUSIONE SESSISSIONE ---
        if race_finished or active_cars == 0 or session_timer >= SESSION_DURATION_MS:
            if race_finished:
                reason_str = f"VITTORIA AGENTE ({winner_info})"
                for car in cars:
                    if car.alive and car.laps < TARGET_LAPS:
                        car.termination_reason = "RACE_ENDED"
            elif active_cars == 0:
                reason_str = "TUTTE LE AUTO OUT (ALL CRASHED/DNF)"
            else:
                reason_str = "TIMEOUT SESSIONE (LIMITI DI TEMPO SUPERATI)"
                for car in cars:
                    if car.alive: car.termination_reason = "TIMEOUT"

            print_deep_simulation_report(generation_count, cars, reason_str)
            
            for car in cars:
                accuracy_aes = (car.checkpoints_passed_total / max(1, car.total_steps)) * 100.0
                global_simulation_stats.append({
                    'gen': generation_count,
                    'car_id': car.car_id,
                    'algo': TEAM_ALGORITHMS[car.team_id],
                    'aes': accuracy_aes,
                    'max_speed': car.max_speed_reached,
                    'cumulative_reward': car.cumulative_reward,
                    'overtakes': car.overtakes_performed,
                    'survival_time': car.survival_time_s
                })

            genetic_car_ids = [c.car_id for c in cars if TEAM_ALGORITHMS[c.team_id] == "GENETIC"]
            if genetic_car_ids:
                best_genetic_id = max(
                    genetic_car_ids,
                    key=lambda cid: (
                        cars[cid].race_progress(track),
                        cars[cid].cumulative_reward,
                    ),
                )
                best_net = agents[best_genetic_id]
                for cid in genetic_car_ids:
                    if cid != best_genetic_id:
                        agents[cid].W1 = best_net.W1.copy()
                        agents[cid].b1 = best_net.b1.copy()
                        agents[cid].W2 = best_net.W2.copy()
                        agents[cid].b2 = best_net.b2.copy()
                        agents[cid].mutate()

            # Contenitori temporanei per la generazione corrente
            current_gen_pg_grads = []
            current_gen_dqn_grads = []

            for car_id, agent in agents.items():
                algo_type = TEAM_ALGORITHMS[cars[car_id].team_id]
                
                if algo_type == "POLICY_GRADIENT":
                    agent.update_policy()
                    # Estrazione della norma se calcolata e salvata nell'istanza dell'agente
                    if hasattr(agent, 'last_grad_norm') and agent.last_grad_norm is not None:
                        current_gen_pg_grads.append(agent.last_grad_norm)
                        
                elif algo_type == "DQN":
                    # DQN aggiorna la rete frame by frame; estraiamo l'ultimo valore registrato
                    if hasattr(agent, 'last_grad_norm') and agent.last_grad_norm is not None:
                        current_gen_dqn_grads.append(agent.last_grad_norm)

            # Calcolo della media della generazione e inserimento nel registro globale
            if current_gen_pg_grads:
                global_gradient_registry["POLICY_GRADIENT"].append(np.mean(current_gen_pg_grads))
            else:
                global_gradient_registry["POLICY_GRADIENT"].append(0.0)

            if current_gen_dqn_grads:
                global_gradient_registry["DQN"].append(np.mean(current_gen_dqn_grads))
            else:
                global_gradient_registry["DQN"].append(0.0)

            race_settings = {
                "target_laps": TARGET_LAPS,
                "num_checkpoints": NUM_CHECKPOINTS,
                "fps": FPS,
                "sensor_angles": SENSOR_ANGLES,
                "max_ray_distance": MAX_RAY_DISTANCE,
                "observation_size": OBSERVATION_SIZE,
                "max_speed_px_s": MAX_SPEED_PX,
                "acceleration_px_s2": ACCELERATION,
                "deceleration_px_s2": DECELERATION,
                "q_learning_gamma": 0.999,
                "dqn_gamma": 0.995,
                "policy_gradient_gamma": 0.999,
                "progress_reward_scale": PROGRESS_REWARD_SCALE,
                "time_penalty_per_second": TIME_PENALTY_PER_SECOND,
                "crash_penalty": CRASH_PENALTY,
                "finish_bonus": FINISH_BONUS,
                "team_reward_share": TEAM_REWARD_SHARE,
            }
            race_logger.record_race(
                generation=generation_count,
                reason=reason_str,
                duration_ms=session_timer,
                track=track,
                cars=cars,
                algorithms=TEAM_ALGORITHMS,
                agents=agents,
                settings=race_settings,
                seed=RUN_SEED,
                best_lap_time=best_lap_time_current_race,
            )
                
            if generation_count == NUM_SIMULATIONS:
                print_grand_final_report()
                pygame.quit()
                sys.exit()
                
            generation_count += 1
            state = "POST_RACE"
            post_race_elapsed_ms = 0

    elif state == "POST_RACE":
        screen.blit(track.surface, (0, 0))
        pygame.draw.rect(screen, (15, 15, 15), (WIDTH//2 - 250, HEIGHT//2 - 70, 500, 140))
        pygame.draw.rect(screen, (255, 255, 255), (WIDTH//2 - 250, HEIGHT//2 - 70, 500, 140), 2)
        
        f_menu = pygame.font.SysFont("Arial", 18, bold=True)
        txt_gen = f_menu.render(f"Simulazione {generation_count - 1} COMPLETATA", True, (0, 255, 0))
        txt_opt1 = f_menu.render("Premi R per RIDISEGNARE un nuovo tracciato", True, (255, 255, 255))
        seconds_left = max(0.0, (POST_RACE_DELAY_MS - post_race_elapsed_ms) / 1000.0)
        txt_opt2 = f_menu.render(f"Prossima corsa automatica tra {seconds_left:.1f}s", True, (255, 255, 255))
        txt_opt3 = f_menu.render("SPAZIO: subito | ESC: termina", True, (255, 255, 255))
        
        screen.blit(txt_gen, (WIDTH//2 - 140, HEIGHT//2 - 45))
        screen.blit(txt_opt1, (WIDTH//2 - 210, HEIGHT//2 - 5))
        screen.blit(txt_opt2, (WIDTH//2 - 210, HEIGHT//2 + 20))
        screen.blit(txt_opt3, (WIDTH//2 - 210, HEIGHT//2 + 45))

    pygame.display.flip()


# Gli algoritmi sembrano commettere sempre gli stessi errori, ripetendoli in modo sistematico nel corso delle diverse simulazioni (a parità di tracciato).
# Fare in modo che il report di ogni simulazione, così come il report finale, vadano a comporre un file txt che raccolga tutte le metriche e le informazioni di interesse, in modo da poterle analizzare successivamente in maniera aggregata e comparativa. Questo permetterebbe di avere un log persistente delle performance degli algoritmi, utile per debug, analisi e miglioramento dei modelli. Il file .txt deve essere legato ad un timestamp e deve andare in una cartella dedicata, inclusa nel gitignore
