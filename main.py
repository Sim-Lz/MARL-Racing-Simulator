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
from simulation import IDX_S1, IDX_S2, RaceSimulation, create_agents, unique_agents
from race_logger import RaceLogger
from race_flow import should_auto_start_next_race, should_exit_early

pygame.init()


def initial_window_size():
    """Finestra che sta nel desktop (con margine per barra delle applicazioni e altre finestre)."""
    info = pygame.display.Info()
    scale = min(1.0, WINDOW_SCREEN_FRACTION * info.current_w / WIDTH, WINDOW_SCREEN_FRACTION * info.current_h / HEIGHT)
    return max(320, int(WIDTH * scale)), max(240, int(HEIGHT * scale))


# Si disegna su una superficie logica fissa (WIDTH x HEIGHT) che viene scalata nella finestra:
# la finestra è ridimensionabile e non copre lo schermo, mentre simulazione e coordinate restano invariate.
window = pygame.display.set_mode(initial_window_size(), pygame.RESIZABLE)
screen = pygame.Surface((WIDTH, HEIGHT))
pygame.display.set_caption("MARL Racing Simulator")
clock = pygame.time.Clock()
sim_speed_index = 0
show_racing_line = False


def viewport():
    window_w, window_h = window.get_size()
    scale = min(window_w / WIDTH, window_h / HEIGHT)
    width, height = max(1, int(WIDTH * scale)), max(1, int(HEIGHT * scale))
    return scale, (window_w - width) // 2, (window_h - height) // 2, width, height


def present():
    scale, offset_x, offset_y, width, height = viewport()
    window.fill((0, 0, 0))
    if abs(scale - 1.0) < 1e-3:
        window.blit(screen, (offset_x, offset_y))
    else:
        window.blit(pygame.transform.smoothscale(screen, (width, height)), (offset_x, offset_y))
    pygame.display.flip()


def logical_mouse_position():
    scale, offset_x, offset_y, _, _ = viewport()
    mouse_x, mouse_y = pygame.mouse.get_pos()
    return (int((mouse_x - offset_x) / scale), int((mouse_y - offset_y) / scale))

track = Track(WIDTH, HEIGHT)
state = "DRAWING"
simulation = None

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

# Un agente per team, condiviso tra le due auto (parameter sharing)
agents = create_agents(TEAM_ALGORITHMS)


def get_gradient_norm(agent):
    """Ultima norma L2 del gradiente (prima del clipping) per i modelli basati su backpropagation."""
    return float(getattr(agent, "last_grad_norm", 0.0) or 0.0)


def print_deep_simulation_report(gen_num, car_list, reason):
    total_sim_checkpoints = NUM_CHECKPOINTS * TARGET_LAPS

    print(f"\n=====================================================================================================================")
    print(f"REPORT SIMULAZIONE N° {gen_num} | MOTIVAZIONE CHIUSURA: {reason}")
    print(f"=====================================================================================================================")
    print(f"| ID | ALGORITMO       | STATO           | GIRI COMPL. | CHECKPOINTS | V-MAX     | T-SURVIVAL | REWARD TOT | GRAD. L2  |")
    print(f"---------------------------------------------------------------------------------------------------------------------")
    for c in sorted(car_list, key=lambda x: x.car_id):
        g_norm = get_gradient_norm(agents[c.car_id])
        g_str = f"{g_norm:.4f}" if g_norm > 0 else "N/A"
        status = "COMPLETED" if c.laps >= TARGET_LAPS else c.termination_reason
        print(f"| {c.car_id:02d} | {TEAM_ALGORITHMS[c.team_id]:<15} | {status:<15} | {c.laps}/{TARGET_LAPS}        | {c.checkpoints_passed_total:04d}/{total_sim_checkpoints:04d} | {int(c.max_speed_reached):>3} px/s | {c.survival_time_s:>9.2f}s | {c.cumulative_reward:>10.1f} | {g_str:<9} |")
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

        variance = np.var(history)
        print(f"  Varianza Storica della Norma: {variance:.8f}")
        if variance > 10.0:
            print("  Nota: Rilevata un'elevata fluttuazione dei gradienti. Valutare l'inserimento di Gradient Clipping.")


def race_settings():
    settings = {
        "target_laps": TARGET_LAPS,
        "num_checkpoints": NUM_CHECKPOINTS,
        "fps": FPS,
        "session_duration_ms": SESSION_DURATION_MS,
        "sensor_angles": SENSOR_ANGLES,
        "sensor_ranges": SENSOR_RANGES,
        "lookahead_distances_px": LOOKAHEAD_DISTANCES_PX,
        "observation_size": OBSERVATION_SIZE,
        "decision_interval_frames": {
            algo: DECISION_INTERVAL_FRAMES.get(algo, DEFAULT_DECISION_INTERVAL) for algo in set(TEAM_ALGORITHMS)
        },
        "max_speed_px_s": MAX_SPEED_PX,
        "acceleration_px_s2": ACCELERATION,
        "deceleration_px_s2": DECELERATION,
        "max_steer_angle": MAX_STEER_ANGLE,
        "progress_reward_scale": PROGRESS_REWARD_SCALE,
        "time_penalty_per_second": TIME_PENALTY_PER_SECOND,
        "termination_penalty": TERMINATION_PENALTY,
        "collision_penalty": COLLISION_PENALTY,
        "finish_bonus": FINISH_BONUS,
        "team_reward_share": TEAM_REWARD_SHARE,
        "lethal_car_collisions": LETHAL_CAR_COLLISIONS,
    }
    for agent, _ in unique_agents(agents):
        for key in ("gamma", "lr", "learning_rate"):
            if hasattr(agent, key):
                settings[f"{agent.algorithm.lower()}_{key}"] = getattr(agent, key)
    return settings


def finalize_race(sim):
    reason_str = sim.close_race()
    print_deep_simulation_report(generation_count, sim.cars, reason_str)

    for car in sim.cars:
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

    for algorithm_name in global_gradient_registry:
        norms = [get_gradient_norm(agent) for agent, _ in unique_agents(agents) if agent.algorithm == algorithm_name]
        global_gradient_registry[algorithm_name].append(float(np.mean(norms)) if norms else 0.0)

    race_logger.record_race(
        generation=generation_count,
        reason=reason_str,
        duration_ms=sim.session_ms,
        track=track,
        cars=sim.cars,
        algorithms=TEAM_ALGORITHMS,
        agents=agents,
        settings=race_settings(),
        seed=RUN_SEED,
        best_lap_time=sim.best_lap_time_race,
    )


def draw_gate(checkpoint_index, color, width):
    cp = track.checkpoints[checkpoint_index]
    index = track.index_at(checkpoint_index * track.checkpoint_spacing)
    nx, ny = track.normal[index]
    r = track.brush_radius
    pygame.draw.line(screen, color, (cp[0] + nx * r, cp[1] + ny * r), (cp[0] - nx * r, cp[1] - ny * r), width)


def draw_race(sim):
    screen.blit(track.surface, (0, 0))
    if track.checkpoints:
        draw_gate(0, (220, 53, 69), 5)
        for cp_idx in (IDX_S1, IDX_S2):
            draw_gate(cp_idx, (0, 255, 255), 4)

    if show_racing_line and track.racing_points is not None:
        pygame.draw.lines(screen, (255, 170, 0), True, [tuple(p) for p in track.racing_points[::4]], 1)

    for ghost in sim.ghosts.values():
        ghost.draw(screen)
    for car in sim.cars:
        car.draw(screen)

    sorted_cars = sorted(sim.cars, key=lambda car: car.race_progress(track), reverse=True)

    row_height = 15
    panel_x = WIDTH - 300
    panel_height = 34 + row_height * len(sorted_cars)
    pygame.draw.rect(screen, (20, 20, 20), (panel_x, 10, 290, panel_height))
    pygame.draw.rect(screen, (70, 70, 70), (panel_x, 10, 290, panel_height), 2)

    font_title = pygame.font.SysFont("Arial", 14, bold=True)
    screen.blit(font_title.render("CLASSIFICA LIVE / GIRI RIM.", True, (255, 255, 255)), (panel_x + 10, 16))

    font_lb = pygame.font.SysFont("Arial", 13)
    for rank, c in enumerate(sorted_cars):
        laps_remaining = max(0, TARGET_LAPS - c.laps)
        if c.termination_reason.startswith("DSQ"): status_text = "DSQ"
        elif not c.alive: status_text = c.termination_reason[:5]
        else: status_text = f"{int(c.speed_px)} px/s"
        if sim.slipstream_flags.get(c.car_id, False) and c.alive: status_text += " +TOW"

        info_str = f"{rank+1}. ID {c.car_id:02d} ({TEAM_ALGORITHMS[c.team_id][:4]}): {laps_remaining} laps | {status_text}"
        screen.blit(font_lb.render(info_str, True, c.color if c.alive else (110, 110, 110)), (panel_x + 10, 38 + rank * row_height))

    font = pygame.font.SysFont("Arial", 20)
    track_record_str = f"{sim.best_lap_time_track:.2f}s" if math.isfinite(sim.best_lap_time_track) else "N/D"
    s1, s2, s3 = sim.best_sector_times
    ui_text = font.render(f"Simulazione {generation_count}/{NUM_SIMULATIONS} | Attive: {sim.active_cars()}/{TOTAL_CARS} | Tempo: {int(sim.session_ms)//1000}s", True, (255, 255, 0))
    record_text = font.render(f"Lap Record: {track_record_str} | S1: {s1:.2f}s | S2: {s2:.2f}s | S3: {s3:.2f}s", True, (0, 255, 255))
    speed_text = font.render(
        f"Velocità x{SIM_SPEED_OPTIONS[sim_speed_index]} (tasti 1-{len(SIM_SPEED_OPTIONS)}) | L: linea ideale "
        f"({'ON' if show_racing_line else 'OFF'}, giro teorico {track.ideal_lap_time:.2f}s)",
        True, (200, 200, 200),
    )
    screen.blit(ui_text, (20, 20))
    screen.blit(record_text, (20, 48))
    screen.blit(speed_text, (20, 76))


def start_next_race():
    global state, post_race_elapsed_ms
    simulation.start_race()
    post_race_elapsed_ms = 0
    state = "RACING"


generation_count = 1
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

            speed_keys = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5][:len(SIM_SPEED_OPTIONS)]
            if event.key in speed_keys:
                sim_speed_index = speed_keys.index(event.key)
            elif event.key == pygame.K_l:
                show_racing_line = not show_racing_line

            if event.key == pygame.K_RETURN and state == "DRAWING" and len(track.drawing_points) > 20:
                if track.finalize():
                    simulation = RaceSimulation(track, agents, TEAM_ALGORITHMS)
                    state = "RACING"
                else:
                    print("Tracciato non valido: ridisegnalo.")
                    track = Track(WIDTH, HEIGHT)

            elif state == "POST_RACE":
                if event.key == pygame.K_r:
                    track = Track(WIDTH, HEIGHT)
                    simulation = None
                    state = "DRAWING"
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
        track.draw_step(screen, logical_mouse_position())

    elif state == "RACING":
        # Passo di simulazione fisso (SIM_DT_MS): i risultati non dipendono dal frame rate reale,
        # quindi accelerare la visualizzazione (più passi per frame) non cambia l'esito delle corse.
        race_over = False
        for _ in range(SIM_SPEED_OPTIONS[sim_speed_index]):
            race_over = simulation.step()
            if race_over:
                break
        draw_race(simulation)

        if race_over:
            finalize_race(simulation)

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

    present()


# Gli algoritmi sembrano commettere sempre gli stessi errori, ripetendoli in modo sistematico nel corso delle diverse simulazioni (a parità di tracciato).
# Fare in modo che il report di ogni simulazione, così come il report finale, vadano a comporre un file txt che raccolga tutte le metriche e le informazioni di interesse, in modo da poterle analizzare successivamente in maniera aggregata e comparativa. Questo permetterebbe di avere un log persistente delle performance degli algoritmi, utile per debug, analisi e miglioramento dei modelli. Il file .txt deve essere legato ad un timestamp e deve andare in una cartella dedicata, inclusa nel gitignore
