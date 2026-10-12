import json
import math
import os
import random
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import torch

import config
from car import Car, controls_to_action
from config import (
    COLLISION_PENALTY,
    FINISH_BONUS,
    NUM_ACTIONS,
    OBS_HEADING_COS,
    OBS_LATERAL,
    OBS_SPEED,
    OBSERVATION_SIZE,
    POST_RACE_DELAY_MS,
    TERMINATION_PENALTY,
)
from race_flow import should_auto_start_next_race, should_exit_early
from race_logger import RaceLogger
from rl_models.dqn import DQNAgent
from rl_models.genetic import GeneticAgent, GeneticNetwork
from rl_models.heuristic import HeuristicController
from rl_models.policy_gradient import PolicyGradientAgent
from rl_models.qlearning import TabularQLearner
from rl_models.rewards import calculate_step_reward
from simulation import GHOST_SLOT_OFFSET, RaceSimulation, create_agents
from team import Team
from track import Track, speed_profile


def make_track(radius=30, wobble=0.35):
    """Tracciato chiuso sintetico (ovale deformato) percorso in senso orario sullo schermo."""
    track = Track(1500, 1000)
    track.brush_radius = radius
    points = []
    for i in range(700):
        angle = 2.0 * math.pi * i / 700
        scale = 1.0 + wobble * math.sin(3.0 * angle)
        point = (int(750 + 420 * scale * math.cos(angle)), int(500 + 300 * scale * math.sin(angle)))
        if not points or math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) > 3:
            points.append(point)
    track.drawing_points = points
    for point in points:
        pygame.draw.circle(track.surface, (255, 255, 255), point, radius)
    assert track.finalize()
    return track


def car_on_centerline(track, arc, speed=0.0):
    index = track.index_at(arc)
    heading = math.degrees(math.atan2(track.tangent[index, 1], track.tangent[index, 0]))
    car = Car(0, 0, 0, track.center[index], heading, (255, 255, 255))
    car.speed_px = speed
    car.update_track_position(track, initial=True)
    car.cast_rays(track)
    return car


class TrackAndCarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = make_track()

    def test_checkpoints_are_equally_spaced_along_the_centerline(self):
        spacing = [
            math.dist(self.track.checkpoints[i], self.track.checkpoints[(i + 1) % len(self.track.checkpoints)])
            for i in range(len(self.track.checkpoints))
        ]
        self.assertEqual(len(self.track.checkpoints), config.NUM_CHECKPOINTS)
        self.assertLess(max(spacing) - min(spacing), 0.2 * np.mean(spacing))

    def test_observation_is_bounded_and_describes_pose_on_track(self):
        car = car_on_centerline(self.track, 500.0, speed=200.0)
        observation = car.get_observation(self.track)

        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertTrue(np.all(observation >= -1.0) and np.all(observation <= 1.0))
        self.assertAlmostEqual(float(observation[OBS_LATERAL]), 0.0, delta=0.1)
        self.assertGreater(float(observation[OBS_HEADING_COS]), 0.95)
        self.assertAlmostEqual(float(observation[OBS_SPEED]), 200.0 / (config.MAX_SPEED_PX * 1.15), places=5)

    def test_side_sensors_use_their_full_range_on_a_normal_track(self):
        car = car_on_centerline(self.track, 500.0)
        side = car.sensor_distances[0] / config.SENSOR_RANGES[0]
        # Muro a ~30 px dal centro, meno la semi-larghezza dell'auto: ~20% della portata del raggio laterale.
        self.assertGreater(side, 0.12)
        self.assertLess(side, 0.3)

    def test_progress_is_potential_based(self):
        car = car_on_centerline(self.track, 1000.0)
        start = car.progress_px
        for _ in range(30):
            car.x += math.cos(math.radians(car.angle)) * 3.0
            car.y += math.sin(math.radians(car.angle)) * 3.0
            car.update_track_position(self.track)
        moved = car.progress_px - start
        self.assertAlmostEqual(moved, 90.0, delta=6.0)
        for _ in range(30):
            car.x -= math.cos(math.radians(car.angle)) * 3.0
            car.y -= math.sin(math.radians(car.angle)) * 3.0
            car.update_track_position(self.track)
        self.assertAlmostEqual(car.progress_px, start, delta=3.0)

    def test_racing_line_stays_inside_the_track_and_cuts_corners(self):
        track = self.track
        for side in (-1.0, 1.0):   # la carrozzeria sulla linea ideale resta sempre in pista
            edge = track.racing_points + side * config.CAR_HALF_WIDTH * track.normal
            self.assertTrue(np.all(track.drivable(edge[:, 0], edge[:, 1])))
        corners = np.abs(track.corner_curvature) > 1.0 / 150.0
        inside = np.sign(track.corner_curvature[corners]) * track.racing_offset[corners]
        self.assertGreater(float(np.mean(inside)), 0.0)   # verso la corda in curva
        _, _, centerline_lap = speed_profile(track.center)
        self.assertLess(track.ideal_lap_time, centerline_lap)
        self.assertTrue(np.all(track.racing_speed <= config.MAX_SPEED_PX + 1e-6))
        self.assertTrue(math.isfinite(track.ideal_lap_time))

    def test_cars_cannot_rotate_in_place(self):
        car = car_on_centerline(self.track, 500.0, speed=0.0)
        heading = car.angle
        car.update(controls_to_action(1, 0), self.track)
        self.assertAlmostEqual(car.angle, heading % 360)

    def test_steering_bleeds_speed_gradually(self):
        car = car_on_centerline(self.track, 500.0, speed=config.MAX_SPEED_PX)
        car.update(controls_to_action(1, 1), self.track)
        self.assertGreater(car.speed_px, config.MAX_SPEED_PX * 0.95)


class RewardTests(unittest.TestCase):
    def test_reward_terms_are_balanced_and_progress_based(self):
        self.assertAlmostEqual(calculate_step_reward(0.25, 1000).total, 0.25 * config.PROGRESS_REWARD_SCALE - config.TIME_PENALTY_PER_SECOND)
        self.assertAlmostEqual(calculate_step_reward(-0.25, 0).total, -0.25 * config.PROGRESS_REWARD_SCALE)
        self.assertAlmostEqual(calculate_step_reward(0, 0, finished=True).total, FINISH_BONUS)
        self.assertAlmostEqual(calculate_step_reward(0, 0, terminated=True).total, -TERMINATION_PENALTY)
        self.assertAlmostEqual(calculate_step_reward(0, 0, collisions_at_fault=1).total, -COLLISION_PENALTY)

    def test_team_sharing_is_positive_only_and_not_double_counted(self):
        car_a = SimpleNamespace(car_id=0)
        car_b = SimpleNamespace(car_id=1)
        team = Team(0, "DQN", [car_a, car_b])
        original = config.TEAM_REWARD_SHARE
        try:
            import team as team_module
            team_module.TEAM_REWARD_SHARE = 0.05
            shared = team.compute_cooperative_rewards({0: 2.0, 1: 10.0}, {0: True, 1: True})
            isolated = team.compute_cooperative_rewards({0: 2.0, 1: 10.0}, {0: True, 1: False})
        finally:
            team_module.TEAM_REWARD_SHARE = original
        self.assertAlmostEqual(shared[0], 2.5)
        self.assertAlmostEqual(isolated[0], 2.0)


class AgentTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.track = make_track()
        self.observation = car_on_centerline(self.track, 500.0, speed=150.0).get_observation(self.track)

    def test_every_agent_returns_a_valid_action(self):
        for agent in (HeuristicController(), TabularQLearner(), DQNAgent(), PolicyGradientAgent(), GeneticAgent()):
            agent.begin_race([0])
            self.assertIn(agent.act(0, self.observation), range(NUM_ACTIONS))

    def test_heuristic_brakes_when_faster_than_the_ideal_profile(self):
        sim = RaceSimulation(self.track, create_agents(["HEURISTIC"]), ["HEURISTIC"])
        agent = sim.agents[0]
        car = sim.cars[0]
        slowest = int(np.argmin(self.track.racing_speed))
        car.x, car.y = self.track.racing_points[slowest]
        car.update_track_position(self.track, initial=True)
        car.speed_px = config.MAX_SPEED_PX
        self.assertEqual(agent.act(0, car.get_observation(self.track)) // 3, 2)

    def test_q_learning_bootstraps_truncations_but_not_terminations(self):
        agent = TabularQLearner()
        state, next_state = (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 0)
        agent.q_table[next_state] = np.full(NUM_ACTIONS, 10.0)

        agent.update(state, 4, 1.0, next_state, done=True)
        self.assertAlmostEqual(float(agent.q_table[state][4]), agent.lr * 1.0)

        agent.q_table[state][:] = 0.0
        agent.update(state, 4, 1.0, next_state, done=False)
        self.assertAlmostEqual(float(agent.q_table[state][4]), agent.lr * (1.0 + agent.gamma * 10.0))

    def test_q_learning_state_varies_with_lateral_position(self):
        agent = TabularQLearner()
        left = self.observation.copy()
        right = self.observation.copy()
        left[OBS_LATERAL], right[OBS_LATERAL] = -0.8, 0.8
        self.assertNotEqual(agent.discretize(left), agent.discretize(right))

    def test_dqn_builds_n_step_returns_and_trains_after_warmup(self):
        agent = DQNAgent()
        agent.minimum_replay_size = 8
        next_obs = self.observation + 0.01
        for step in range(4):
            agent.observe(0, self.observation, 1, 1.0, next_obs, False, False)
        agent.observe(0, self.observation, 1, 1.0, next_obs, True, False)

        n_step_return = sum(agent.gamma ** k for k in range(agent.n_step))
        self.assertAlmostEqual(agent.memory[0][2], n_step_return, places=5)
        self.assertEqual(agent.memory[-1][4], 1.0)
        for _ in range(10):
            agent.observe(1, self.observation, 2, 0.5, next_obs, False, False)
        self.assertGreater(agent.steps_counter, 0)
        self.assertGreater(agent.last_grad_norm, 0.0)

    def test_policy_gradient_updates_after_a_rollout(self):
        agent = PolicyGradientAgent()
        agent.rollout_size = 64
        for step in range(64):
            action = agent.act(step % 2, self.observation)
            agent.observe(step % 2, self.observation, action, 1.0 if action == 1 else 0.0, self.observation, step % 16 == 15, False)
        self.assertEqual(agent.updates, 1)
        self.assertEqual(agent.diagnostics()["pending_transitions"], 0)
        self.assertGreater(agent.last_grad_norm, 0.0)

    def test_genetic_agent_evolves_after_evaluating_its_population(self):
        agent = GeneticAgent(population_size=4, elite_count=1, seed=0)
        agent.begin_track(SimpleNamespace(track=self.track))
        agent.begin_race([0, 1])
        starts = set()
        for evaluation in range(4 * agent.benchmark_starts):
            slot = 100 + evaluation % 3
            agent.begin_episode(slot)
            starts.add(agent.training_spawn_index(slot, 123, self.track))
            individual = agent.assignments[slot][1]
            agent.end_episode(slot, {"fitness": float(individual), "progress": 0.0})
        self.assertEqual(starts, set(agent.start_indices))   # benchmark fisso
        self.assertEqual(agent.generation, 1)
        self.assertEqual(agent.best_fitness, 3.0 * agent.benchmark_starts)
        self.assertEqual(agent.fitness(0), 3.0 * agent.benchmark_starts)   # l'elitario conserva la fitness
        self.assertIsNone(agent.fitness(1))
        self.assertGreater(agent.last_mutation_norm, 0.0)

    def test_genetic_evaluation_is_deterministic(self):
        sim = RaceSimulation(self.track, create_agents(["GENETIC"]), ["GENETIC"])
        agent = sim.agents[0]
        results = []
        for _ in range(2):
            sim.start_race()
            agent.scores = [[None] * agent.benchmark_starts for _ in range(agent.population_size)]
            for _ in range(int(3 * config.FPS)):
                sim.step()
            ghost = sim.ghosts.get(GHOST_SLOT_OFFSET)
            results.append(None if ghost is None else (round(ghost.x, 6), round(ghost.y, 6)))
        self.assertEqual(results[0], results[1])

    def test_genetic_network_maps_two_outputs_to_nine_actions(self):
        network = GeneticNetwork(np.random.default_rng(0))
        self.assertIn(network.forward(self.observation), range(NUM_ACTIONS))


class SimulationTests(unittest.TestCase):
    def test_heuristic_team_drives_laps_without_eliminations(self):
        random.seed(1)
        track = make_track()
        sim = RaceSimulation(track, create_agents(["HEURISTIC"]), ["HEURISTIC"])
        for _ in range(int(30 * config.FPS)):
            if sim.step():
                break
        self.assertTrue(all(car.alive for car in sim.cars))
        self.assertTrue(all(car.laps >= 1 for car in sim.cars))

    def test_grid_fits_a_narrow_track(self):
        random.seed(2)
        track = make_track(radius=12)
        sim = RaceSimulation(track, create_agents(config.TEAM_ALGORITHMS), config.TEAM_ALGORITHMS)
        self.assertTrue(all(car.is_on_track(track) for car in sim.cars))

    def test_collisions_slow_cars_and_penalize_only_the_car_behind(self):
        track = make_track()
        sim = RaceSimulation(track, create_agents(["HEURISTIC"]), ["HEURISTIC"])
        ahead, behind = sim.cars
        for car, arc in ((ahead, 505.0), (behind, 500.0)):
            index = track.index_at(arc)
            car.x, car.y = track.center[index]
            car.speed_px = 200.0
            car.update_track_position(track, initial=True)
        at_fault = sim._resolve_car_collisions(sim.cars)

        self.assertEqual(at_fault, {behind.car_id: 1})
        self.assertTrue(ahead.alive and behind.alive)
        self.assertLess(behind.speed_px, ahead.speed_px)

    def test_eliminated_learners_respawn_as_ghosts_without_affecting_results(self):
        track = make_track()
        sim = RaceSimulation(track, create_agents(["DQN"]), ["DQN"])
        victim = sim.cars[0]
        victim.x, victim.y = -50.0, -50.0  # fuori pista: crash al primo frame
        for _ in range(config.RESPAWN_DELAY_FRAMES + 2):
            sim.step()

        self.assertEqual(victim.termination_reason, "CRASHED")
        self.assertIn(victim.car_id + GHOST_SLOT_OFFSET, sim.ghosts)
        self.assertEqual(victim.training_respawns, 1)
        self.assertEqual(sim.active_cars(), 1)

    def test_race_end_closes_pending_transitions_as_truncations(self):
        track = make_track()
        agents = create_agents(["Q_LEARNING"])
        sim = RaceSimulation(track, agents, ["Q_LEARNING"])
        for _ in range(10):
            sim.step()
        sim.close_race()
        self.assertEqual(sim._pending, {})
        self.assertTrue(all(car.termination_reason in ("TIMEOUT", "CRASHED") for car in sim.cars))


class RaceLoggerTests(unittest.TestCase):
    def test_race_log_records_track_geometry_and_per_car_results(self):
        random.seed(3)
        track = make_track()
        agents = create_agents(["HEURISTIC"])
        sim = RaceSimulation(track, agents, ["HEURISTIC"])
        for _ in range(60):
            sim.step()
        reason = sim.close_race()
        with tempfile.TemporaryDirectory() as directory:
            logger = RaceLogger(directory)
            for generation in (1, 2):
                logger.record_race(
                    generation=generation,
                    reason=reason,
                    duration_ms=sim.session_ms,
                    track=track,
                    cars=sim.cars,
                    algorithms=["HEURISTIC"],
                    agents=agents,
                    settings={"target_laps": 1},
                    seed=7,
                    best_lap_time=float("inf") if generation == 1 else 12.0,
                )
            saved_tracks = [json.loads(line) for line in (logger.run_directory / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
            saved_races = [json.loads(line) for line in (logger.run_directory / "races.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(saved_tracks), 1)
        self.assertEqual(len(saved_races), 2)
        self.assertIn("curvature", saved_tracks[0])
        self.assertEqual(len(saved_tracks[0]["curvature_by_checkpoint"]), config.NUM_CHECKPOINTS)
        self.assertEqual(saved_races[0]["cars"][0]["algorithm"], "HEURISTIC")
        self.assertEqual(saved_races[0]["track_id"], saved_tracks[0]["track_id"])
        self.assertIsNone(saved_races[0]["best_lap_seconds"])
        self.assertEqual(saved_races[1]["best_lap_seconds"], 12.0)


class RaceFlowTests(unittest.TestCase):
    def test_escape_only_exits_between_races(self):
        self.assertTrue(should_exit_early("POST_RACE", 27, 27))
        self.assertFalse(should_exit_early("RACING", 27, 27))
        self.assertFalse(should_exit_early("DRAWING", 27, 27))

    def test_next_race_starts_automatically_after_intermission(self):
        self.assertFalse(should_auto_start_next_race("POST_RACE", POST_RACE_DELAY_MS - 1))
        self.assertTrue(should_auto_start_next_race("POST_RACE", POST_RACE_DELAY_MS))
        self.assertFalse(should_auto_start_next_race("RACING", POST_RACE_DELAY_MS))


if __name__ == "__main__":
    unittest.main()
