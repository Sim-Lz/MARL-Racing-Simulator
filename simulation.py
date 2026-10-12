# simulation.py
"""Logica di gara indipendente dal rendering: spawn, fisica, progresso, collisioni, reward e transizioni."""
import math
import random

import numpy as np

from car import Car
from config import *
from rl_models.dqn import DQNAgent
from rl_models.genetic import GeneticAgent
from rl_models.heuristic import HeuristicController
from rl_models.policy_gradient import PolicyGradientAgent
from rl_models.qlearning import TabularQLearner
from rl_models.rewards import calculate_step_reward
from team import Team

IDX_S1 = int(NUM_CHECKPOINTS * 0.33)
IDX_S2 = int(NUM_CHECKPOINTS * 0.67)
GRID_FRONT_GAP_PX = 10.0
GRID_ROW_SPACING_PX = 26.0
SAME_TRACK_SECTION_PX = 40.0
# Gli episodi fantasma hanno uno "slot" proprio (car_id + offset): possono girare in parallelo all'auto in
# gara senza mescolare i buffer degli agenti, che sono indicizzati per slot.
GHOST_SLOT_OFFSET = 100
CORNER_CURVATURE = 1.0 / 150.0   # sopra questa curvatura un tratto conta come curva nelle metriche di traiettoria

AGENT_FACTORIES = {
    "HEURISTIC": HeuristicController,
    "GENETIC": GeneticAgent,
    "Q_LEARNING": TabularQLearner,
    "DQN": DQNAgent,
    "POLICY_GRADIENT": PolicyGradientAgent,
}


def create_agents(algorithms=TEAM_ALGORITHMS):
    """Un agente per team, condiviso dalle sue auto (parameter sharing). Ritorna {car_id: agente}."""
    agents = {}
    for team_id, algorithm in enumerate(algorithms):
        agent = AGENT_FACTORIES[algorithm]()
        for index in range(CARS_PER_TEAM):
            agents[team_id * CARS_PER_TEAM + index] = agent
    return agents


def unique_agents(agents):
    seen = {}
    for car_id, agent in agents.items():
        seen.setdefault(id(agent), (agent, []))[1].append(car_id)
    return list(seen.values())


class RaceSimulation:
    def __init__(self, track, agents, algorithms=TEAM_ALGORITHMS):
        self.track = track
        self.agents = agents
        self.algorithms = algorithms
        self.total_cars = len(algorithms) * CARS_PER_TEAM
        self.cars = []
        self.teams = []
        # Record del tracciato (persistono tra le corse sullo stesso tracciato)
        self.best_lap_time_track = float("inf")
        self.best_sector_times = [float("inf")] * 3
        for agent, _ in unique_agents(self.agents):
            agent.begin_track(self)
        self.start_race()

    # ------------------------------------------------------------------ setup
    def start_race(self):
        self.session_ms = 0.0
        self.frame = 0
        self.race_finished = False
        self.winner_info = ""
        self.best_lap_time_race = float("inf")
        self.slipstream_flags = {}
        self._pending = {}
        self._collision_cooldown = {}
        self._previous_progress = {}
        self.ghosts = {}
        self._respawn_due = {}
        self._spawn_cars()
        for agent, car_ids in unique_agents(self.agents):
            agent.begin_race(car_ids)
            if TRAINING_RESPAWN and agent.continuous_training_ghost:
                for car_id in car_ids:
                    for copy in range(1, agent.training_ghosts_per_car + 1):
                        slot = car_id + copy * GHOST_SLOT_OFFSET
                        self._schedule_respawn(self.cars[car_id], index=None, delay=0, slot=slot)

    def _spawn_cars(self):
        track = self.track
        offset = track.brush_radius * 0.45
        side_by_side = (
            offset + CAR_HALF_WIDTH + 2.0 <= track.brush_radius
            and 2.0 * offset >= CAR_COLLISION_DISTANCE + 2.0
        )
        slots = list(range(self.total_cars))
        random.shuffle(slots)

        self.cars = [None] * self.total_cars
        for slot, car_id in enumerate(slots):
            row, col = (slot // 2, slot % 2) if side_by_side else (slot, 0)
            arc = track.length - (GRID_FRONT_GAP_PX + row * GRID_ROW_SPACING_PX)
            index = track.index_at(arc)
            lateral = (offset if col == 0 else -offset) if side_by_side else 0.0
            x = track.center[index, 0] + track.normal[index, 0] * lateral
            y = track.center[index, 1] + track.normal[index, 1] * lateral
            heading = math.degrees(math.atan2(track.tangent[index, 1], track.tangent[index, 0]))
            team_id = car_id // CARS_PER_TEAM
            car = Car(car_id, team_id, car_id % CARS_PER_TEAM, (x, y), heading, TEAM_COLORS[team_id])
            car.slot = car_id
            car.decision_interval = DECISION_INTERVAL_FRAMES.get(self.algorithms[team_id], DEFAULT_DECISION_INTERVAL)
            car.sector_timer_start = 0.0
            car.update_track_position(track, initial=True)
            car.cast_rays(track)
            self.cars[car_id] = car

        self.teams = [
            Team(team_id, algorithm, self.cars[team_id * CARS_PER_TEAM:(team_id + 1) * CARS_PER_TEAM])
            for team_id, algorithm in enumerate(self.algorithms)
        ]
        self._previous_progress = {car.car_id: car.progress_px for car in self.cars}

    # ------------------------------------------------------------------ percezione
    def _update_slipstream(self):
        flags = {car.car_id: False for car in self.cars if car.alive}
        for car_a in self.cars:
            if not car_a.alive:
                continue
            for car_b in self.cars:
                if not car_b.alive or car_b is car_a:
                    continue
                dx, dy = car_b.x - car_a.x, car_b.y - car_a.y
                dist = math.hypot(dx, dy)
                if 15.0 < dist < 120.0:
                    angle_diff = abs(math.degrees(math.atan2(dy, dx)) % 360 - car_a.angle)
                    if min(angle_diff, 360 - angle_diff) < 22.5:
                        flags[car_a.car_id] = True
                        break
        self.slipstream_flags = flags

    def _update_traffic(self):
        alive = [car for car in self.cars if car.alive]
        if len(alive) < 2:
            for car in alive:
                car.traffic_features = (0.0, 0.0, 0.0)
            return
        positions = np.array([(car.x, car.y) for car in alive])
        headings = np.radians([car.angle for car in alive])
        speeds = np.array([car.speed_px for car in alive])
        cone = math.radians(TRAFFIC_CONE_DEG)
        for i, car in enumerate(alive):
            delta = positions - positions[i]
            distance = np.hypot(delta[:, 0], delta[:, 1])
            bearing = (np.arctan2(delta[:, 1], delta[:, 0]) - headings[i] + np.pi) % (2.0 * np.pi) - np.pi
            candidates = (distance > 0.0) & (distance < TRAFFIC_RANGE_PX) & (np.abs(bearing) < cone)
            if not candidates.any():
                car.traffic_features = (0.0, 0.0, 0.0)
                continue
            nearest = int(np.flatnonzero(candidates)[np.argmin(distance[candidates])])
            closing = (speeds[i] - speeds[nearest] * math.cos(headings[nearest] - headings[i])) / MAX_SPEED_PX
            car.traffic_features = (
                float(1.0 - distance[nearest] / TRAFFIC_RANGE_PX),
                float(max(-1.0, min(1.0, closing))),
                float(math.sin(bearing[nearest])),
            )

    # ------------------------------------------------------------------ respawn di addestramento
    def _schedule_respawn(self, car, index, delay=RESPAWN_DELAY_FRAMES, slot=None):
        agent = self.agents[car.car_id]
        if not (TRAINING_RESPAWN and agent.uses_training_respawn):
            return
        if slot is None:
            # Un'auto in gara eliminata genera il primo slot fantasma; un fantasma riusa il proprio.
            slot = car.slot if car.is_ghost else car.car_id + GHOST_SLOT_OFFSET
        if slot in self.ghosts or slot in self._respawn_due:
            return
        self._respawn_due[slot] = (self.frame + delay, index)

    def _spawn_due_ghosts(self):
        track = self.track
        for slot, (due_frame, index) in list(self._respawn_due.items()):
            if self.frame < due_frame:
                continue
            del self._respawn_due[slot]
            owner = self.cars[slot % GHOST_SLOT_OFFSET]
            agent = self.agents[owner.car_id]
            if index is None:
                index = owner.track_index or 0
            else:
                index = (index - int(RESPAWN_BACKOFF_PX / track.step)) % len(track.center)
            agent.begin_episode(slot)
            index = agent.training_spawn_index(slot, index, track)
            heading = math.degrees(math.atan2(track.tangent[index, 1], track.tangent[index, 0]))
            ghost = Car(owner.car_id, owner.team_id, owner.index_in_team, track.center[index], heading, owner.color)
            ghost.slot = slot
            ghost.is_ghost = True
            ghost.decision_interval = owner.decision_interval
            ghost.update_track_position(track, initial=True)
            ghost.spawn_progress_px = ghost.progress_px
            ghost.cast_rays(track)
            self.ghosts[slot] = ghost
            owner.training_respawns += 1

    # ------------------------------------------------------------------ passo di simulazione
    def step(self):
        """Avanza di un frame. Ritorna True quando la corsa è terminata."""
        track = self.track
        self.session_ms += SIM_DT_MS
        self.frame += 1
        self._spawn_due_ghosts()
        self._update_slipstream()
        self._update_traffic()

        racing = [car for car in self.cars if car.alive]
        ghosts = [ghost for ghost in self.ghosts.values() if ghost.alive]
        active = racing + ghosts
        actions = {}
        for car in active:
            car.survival_time_s += 1.0 / FPS
            agent = self.agents[car.car_id]
            pending = self._pending.get(car.slot)
            if pending is None or pending["frames"] >= car.decision_interval:
                observation = car.get_observation(track)
                if pending is not None:
                    agent.observe(car.slot, pending["obs"], pending["action"], pending["reward"], observation, False, False)
                pending = {"obs": observation, "action": agent.act(car.slot, observation), "reward": 0.0, "frames": 0}
                self._pending[car.slot] = pending
                car.decisions += 1
            pending["frames"] += 1
            actions[car.slot] = pending["action"]

        progress_delta = {}
        finished = set()
        for car in active:
            slipstream = False if car.is_ghost else self.slipstream_flags.get(car.car_id, False)
            car.update(actions[car.slot], track, slipstream)
            progress_delta[car.slot] = car.update_track_position(track)
            if car.alive:
                if not car.is_ghost and self._register_progress(car):
                    finished.add(car.slot)
                else:
                    car.check_disqualifications(track)
            if not car.is_ghost:
                self._record_line_stats(car)

        # Episodi fantasma a durata limitata (es. valutazioni del genetico): chiusi come troncamento.
        expired = set()
        for ghost in ghosts:
            limit = self.agents[ghost.car_id].max_training_episode_s
            if ghost.alive and limit is not None and ghost.survival_time_s >= limit:
                expired.add(ghost.slot)

        collisions_at_fault = self._resolve_car_collisions(racing)
        self._count_overtakes()

        rewards = {}
        terms_by_car = {}
        for car in active:
            just_died = not car.alive
            if just_died and car.death_info is None and not car.is_ghost:
                car.death_info = self._death_info(car)
            terms = calculate_step_reward(
                progress_delta[car.slot] / track.checkpoint_spacing,
                SIM_DT_MS,
                terminated=just_died,
                collisions_at_fault=collisions_at_fault.get(car.car_id, 0) if not car.is_ghost else 0,
                finished=car.slot in finished,
            )
            if car.is_ghost:
                owner = self.cars[car.car_id]
                owner.training_reward_total += terms.total
                owner.training_progress_total += terms.progress / max(PROGRESS_REWARD_SCALE, 1e-9)
                car.individual_reward_total += terms.total
                self._pending[car.slot]["reward"] += terms.total
            else:
                terms_by_car[car.car_id] = terms
                rewards[car.car_id] = terms.total

        alive_now = {car.car_id: car.alive for car in self.cars}
        for team in self.teams:
            shared = team.compute_cooperative_rewards(rewards, alive_now)
            for car in team.cars:
                if car.car_id not in rewards:
                    continue
                terms = terms_by_car[car.car_id]
                car.progress_reward_total += terms.progress
                car.time_penalty_total += terms.time_penalty
                car.collision_penalty_total += terms.collision_penalty
                car.terminal_penalty_total += terms.termination_penalty
                car.completion_bonus_total += terms.finish_bonus
                car.individual_reward_total += terms.total
                car.cumulative_reward += shared[car.car_id]
                self._pending[car.slot]["reward"] += shared[car.car_id]

        for car in active:
            episode_over = not car.alive or car.slot in finished or car.slot in expired
            if not episode_over:
                continue
            terminated = not car.alive or car.slot in finished
            agent = self.agents[car.car_id]
            pending = self._pending.pop(car.slot)
            agent.observe(
                car.slot, pending["obs"], pending["action"], pending["reward"],
                car.get_observation(track), terminated, not terminated,
            )
            if car.slot in finished:
                car.alive = False
            if car.is_ghost:
                agent.end_episode(car.slot, {
                    "fitness": car.individual_reward_total,
                    "progress": (car.progress_px - car.spawn_progress_px) / track.checkpoint_spacing,
                    "terminated": terminated,
                })
                del self.ghosts[car.slot]
            if not self.race_finished and car.slot not in finished:
                self._schedule_respawn(car, car.track_index or 0)

        return self.is_over()

    def _record_line_stats(self, car):
        """Metriche di traiettoria: uso della carreggiata, lato interno in curva, velocità vs profilo ideale."""
        track = self.track
        index = car.track_index or 0
        half_width = max(1.0, float(track.brush_radius))
        stats = car.line_stats
        stats["frames"] += 1
        stats["abs_lateral"] += abs(car.lateral_offset) / half_width
        curvature = float(track.corner_curvature[index])
        if abs(curvature) > CORNER_CURVATURE:
            stats["corner_frames"] += 1
            stats["inside"] += math.copysign(1.0, curvature) * car.lateral_offset / half_width
        if track.racing_speed is not None:
            stats["speed_ratio"] += min(1.5, car.speed_px / max(1.0, float(track.racing_speed[index])))
            stats["line_error"] += abs(car.lateral_offset - float(track.racing_offset[index])) / half_width
        if car.traffic_features[0] > 0.5:
            stats["traffic_frames"] += 1

    def _register_progress(self, car):
        """Aggiorna checkpoint, settori e giri usando il massimo progresso raggiunto. True se ha finito."""
        track = self.track
        car.max_progress_px = max(car.max_progress_px, car.progress_px)
        passed = int(car.max_progress_px // track.checkpoint_spacing)
        finished = False
        while car.checkpoints_passed_total < passed:
            car.checkpoints_passed_total += 1
            index = car.checkpoints_passed_total % NUM_CHECKPOINTS
            if index in (IDX_S1, IDX_S2, 0):
                sector = {IDX_S1: 0, IDX_S2: 1, 0: 2}[index]
                sector_time = (self.session_ms - car.sector_timer_start) / 1000.0
                car.sector_timer_start = self.session_ms
                self.best_sector_times[sector] = min(self.best_sector_times[sector], sector_time)
            if index == 0:
                car.laps += 1
                lap_time = max(0.5, (self.session_ms - car.last_lap_timer) / 1000.0)
                car.last_lap_timer = self.session_ms
                car.lap_times.append(lap_time)
                self.best_lap_time_race = min(self.best_lap_time_race, lap_time)
                self.best_lap_time_track = min(self.best_lap_time_track, lap_time)
                if car.laps >= TARGET_LAPS and not self.race_finished:
                    self.race_finished = True
                    self.winner_info = f"Vettura {car.car_id} ({self.algorithms[car.team_id]})"
                    car.termination_reason = "COMPLETED"
                    finished = True
        car.next_checkpoint = (car.checkpoints_passed_total + 1) % NUM_CHECKPOINTS
        return finished

    def _resolve_car_collisions(self, active):
        at_fault = {}
        alive = [car for car in active if car.alive]
        length = self.track.length
        for i, car_a in enumerate(alive):
            for car_b in alive[i + 1:]:
                if math.hypot(car_a.x - car_b.x, car_a.y - car_b.y) >= CAR_COLLISION_DISTANCE:
                    continue
                gap = (car_a.progress_px - car_b.progress_px + length / 2.0) % length - length / 2.0
                if abs(gap) > SAME_TRACK_SECTION_PX:
                    continue  # tratti diversi di una pista che si incrocia
                pair = (car_a.car_id, car_b.car_id)
                if self.frame - self._collision_cooldown.get(pair, -COLLISION_COOLDOWN_FRAMES) < COLLISION_COOLDOWN_FRAMES:
                    continue
                self._collision_cooldown[pair] = self.frame
                behind, ahead = (car_a, car_b) if gap < 0 else (car_b, car_a)
                if LETHAL_CAR_COLLISIONS:
                    car_a.kill("COLLISION")
                    car_b.kill("COLLISION")
                else:
                    behind.speed_px *= COLLISION_SPEED_FACTOR_AT_FAULT
                    ahead.speed_px *= COLLISION_SPEED_FACTOR_VICTIM
                behind.collisions_caused += 1
                ahead.collisions_suffered += 1
                at_fault[behind.car_id] = at_fault.get(behind.car_id, 0) + 1
        return at_fault

    def _count_overtakes(self):
        alive = [car for car in self.cars if car.alive]
        for car_a in alive:
            before_a = self._previous_progress.get(car_a.car_id, car_a.progress_px)
            for car_b in alive:
                if car_b is car_a:
                    continue
                before_b = self._previous_progress.get(car_b.car_id, car_b.progress_px)
                if before_a < before_b and car_a.progress_px > car_b.progress_px \
                        and abs(car_a.progress_px - car_b.progress_px) < SAME_TRACK_SECTION_PX:
                    car_a.overtakes_performed += 1
        self._previous_progress = {car.car_id: car.progress_px for car in self.cars}

    def _death_info(self, car):
        track = self.track
        index = car.track_index if car.track_index is not None else 0
        window = (index + np.arange(-10, 11)) % len(track.center)
        info = {
            "time_seconds": round(self.session_ms / 1000.0, 3),
            "track_fraction": round(index / len(track.center), 4),
            "progress_checkpoints": round(car.race_progress(track), 3),
            "speed_px_s": round(car.speed_px, 2),
            "lateral_offset_px": round(car.lateral_offset, 2),
            "heading_error_deg": round(math.degrees(car.heading_error), 2),
            "local_curvature": round(float(np.mean(track.curvature[window])), 5),
            "sensors": [round(float(d), 1) for d in car.sensor_distances],
        }
        if track.racing_speed is not None:
            info["ideal_speed_px_s"] = round(float(track.racing_speed[index]), 2)
            info["ideal_offset_px"] = round(float(track.racing_offset[index]), 2)
        return info

    # ------------------------------------------------------------------ fine corsa
    def active_cars(self):
        return sum(1 for car in self.cars if car.alive)

    def is_over(self):
        return self.race_finished or self.active_cars() == 0 or self.session_ms >= SESSION_DURATION_MS

    def close_race(self):
        """Chiude le transizioni pendenti come troncamenti, notifica gli agenti e ritorna la motivazione."""
        if self.race_finished:
            reason = f"VITTORIA AGENTE ({self.winner_info})"
            truncated_reason = "RACE_ENDED"
        elif self.active_cars() == 0:
            reason = "TUTTE LE AUTO OUT (ALL CRASHED/DNF)"
            truncated_reason = None
        else:
            reason = "TIMEOUT SESSIONE (LIMITI DI TEMPO SUPERATI)"
            truncated_reason = "TIMEOUT"

        for car in self.cars + list(self.ghosts.values()):
            if car.alive:
                if not car.is_ghost:
                    car.termination_reason = truncated_reason or car.termination_reason
                pending = self._pending.pop(car.slot, None)
                if pending is not None:
                    self.agents[car.car_id].observe(
                        car.slot, pending["obs"], pending["action"], pending["reward"],
                        car.get_observation(self.track), False, True,
                    )
        self.ghosts = {}
        self._respawn_due = {}

        for agent, car_ids in unique_agents(self.agents):
            agent.end_race({
                car_id: {
                    "progress": self.cars[car_id].race_progress(self.track),
                    "fitness": self.cars[car_id].individual_reward_total,
                    "laps": self.cars[car_id].laps,
                    "termination_reason": self.cars[car_id].termination_reason,
                }
                for car_id in car_ids
            })
        return reason
