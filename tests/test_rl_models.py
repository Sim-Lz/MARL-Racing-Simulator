import json
import math
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from car import Car
from config import FINISH_BONUS, OBSERVATION_SIZE
from race_logger import RaceLogger
from race_flow import should_auto_start_next_race, should_exit_early
from config import POST_RACE_DELAY_MS
from rl_models.dqn import DQNAgent
from rl_models.genetic import GeneticNetwork
from rl_models.heuristic import HeuristicController
from rl_models.policy_gradient import PolicyGradientAgent
from rl_models.qlearning import TabularQLearner
from rl_models.rewards import calculate_step_reward
from team import Team


class ReinforcementModelTests(unittest.TestCase):
    def test_observation_contains_normalized_sensors_speed_and_target_bearing(self):
        car = Car(0, 0, 0, (0, 0), 0, (255, 255, 255))
        car.sensor_distances[:] = [0, 100, 200, 300, 400, 200, 100]
        car.speed_px = 360
        track = SimpleNamespace(
            checkpoints=[(0, 10)],
            get_pixel_safety=lambda *_: True,
        )

        observation = car.get_observation(track)

        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        np.testing.assert_allclose(observation[:7], [0, 0.25, 0.5, 0.75, 1, 0.5, 0.25])
        self.assertAlmostEqual(float(observation[7]), 360 / (360 * 1.15))
        self.assertAlmostEqual(float(observation[8]), 1.0)
        self.assertAlmostEqual(float(observation[9]), 0.0, places=6)

    def test_race_progress_is_continuous_across_a_checkpoint(self):
        track = SimpleNamespace(
            checkpoints=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )
        car = Car(0, 0, 0, (99, 0), 0, (255, 255, 255))
        car.next_checkpoint = 1
        before = car.race_progress(track)

        car.x, car.y = 100, 0
        car.next_checkpoint = 2
        after = car.race_progress(track)

        self.assertLess(abs(after - before), 0.03)

    def test_footprint_check_does_not_treat_a_nearby_side_wall_as_a_collision(self):
        track = SimpleNamespace(
            get_pixel_safety=lambda x, y: 40 <= x <= 60 and 45 <= y <= 55,
        )
        car = Car(0, 0, 0, (50, 50), 0, (255, 255, 255))

        car.cast_rays(track)

        self.assertLess(float(car.sensor_distances[0]), 8.0)
        self.assertTrue(car.is_on_track(track))

    def test_reward_follows_progress_not_speed_or_control_choice(self):
        self.assertAlmostEqual(calculate_step_reward(0.25, 1000), 1.95)
        self.assertAlmostEqual(calculate_step_reward(-0.25, 0), -2.0)
        self.assertAlmostEqual(calculate_step_reward(0, 0, finished=True), FINISH_BONUS)

    def test_heuristic_steers_toward_target_and_brakes_for_stopping_distance(self):
        agent = HeuristicController()
        clear_road = np.array([1.0] * 7 + [0.0, 0.5, math.sqrt(0.75)])
        self.assertEqual(agent.select_action(clear_road), 2)

        urgent_braking = clear_road.copy()
        urgent_braking[3] = 0.1
        urgent_braking[7] = 0.9
        self.assertEqual(agent.select_action(urgent_braking) // 3, 2)

    def test_q_learning_uses_terminal_target_without_bootstrapping(self):
        agent = TabularQLearner()
        state = (0, 0, 0, 0, 0)
        terminal_state = (3, 3, 3, 3, 4)
        agent.q_table[terminal_state] = np.full(9, 100.0)

        agent.update(state, 4, 1.0, terminal_state, done=True)

        self.assertAlmostEqual(float(agent.q_table[state][4]), agent.lr)
        self.assertGreater(agent.epsilon, agent.min_epsilon)

    def test_dqn_trains_with_a_small_replay_buffer(self):
        agent = DQNAgent()
        state = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        next_state = np.full(OBSERVATION_SIZE, 0.1, dtype=np.float32)
        for _ in range(8):
            agent.store_transition(state, 1, 1.0, next_state, True)

        for _ in range(4):
            agent.train_step()

        self.assertEqual(agent.steps_counter, 1)
        self.assertGreater(agent.last_grad_norm, 0.0)

    def test_policy_gradient_updates_and_clears_an_aligned_episode(self):
        agent = PolicyGradientAgent()
        state = np.full(OBSERVATION_SIZE, 0.2, dtype=np.float32)
        for reward in (1.0, 0.0, -1.0):
            agent.select_action(state)
            agent.rewards.append(reward)

        agent.update_policy()

        self.assertEqual(agent.saved_log_probs, [])
        self.assertEqual(agent.rewards, [])
        self.assertEqual(agent.saved_entropies, [])
        self.assertGreater(agent.last_grad_norm, 0.0)

    def test_policy_gradient_rejects_misaligned_episode_data(self):
        agent = PolicyGradientAgent()
        agent.select_action(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "not aligned"):
            agent.update_policy()

    def test_genetic_model_accepts_the_shared_observation_shape(self):
        agent = GeneticNetwork()
        state = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        self.assertIn(agent.forward(state), range(9))
        agent.mutate()
        self.assertGreaterEqual(agent.last_mutation_norm, 0.0)

    def test_team_sharing_is_small_positive_only_and_not_double_counted(self):
        car_a = SimpleNamespace(car_id=0)
        car_b = SimpleNamespace(car_id=1)
        team = Team(0, "DQN", [car_a, car_b])

        shared = team.compute_cooperative_rewards({0: 2.0, 1: 10.0}, {0: True, 1: True})
        isolated = team.compute_cooperative_rewards({0: 2.0, 1: 10.0}, {0: True, 1: False})

        self.assertAlmostEqual(shared[0], 2.5)
        self.assertAlmostEqual(isolated[0], 2.0)


class RaceLoggerTests(unittest.TestCase):
    def test_race_log_records_track_geometry_and_per_car_results(self):
        car = SimpleNamespace(
            car_id=0,
            team_id=0,
            laps=1,
            checkpoints_passed_total=4,
            termination_reason="COMPLETED",
            total_steps=12,
            survival_time_s=3.0,
            distance_traveled=200.0,
            max_speed_reached=100.0,
            cumulative_reward=20.0,
            progress_reward_total=18.0,
            time_penalty_total=0.1,
            terminal_penalty_total=0.0,
            completion_bonus_total=2.0,
            overtakes_performed=0,
        )
        track = SimpleNamespace(
            width=100,
            height=100,
            brush_radius=10,
            drawing_points=[(0, 0), (10, 0), (10, 10)],
            checkpoints=[(0, 0), (10, 0), (10, 10)],
        )
        with tempfile.TemporaryDirectory() as directory:
            logger = RaceLogger(directory)
            logger.record_race(
                generation=1,
                reason="test",
                duration_ms=3000,
                track=track,
                cars=[car],
                algorithms=["HEURISTIC"],
                agents={0: HeuristicController()},
                settings={"target_laps": 1},
                seed=7,
                best_lap_time=float("inf"),
            )
            logger.record_race(
                generation=2,
                reason="test",
                duration_ms=3000,
                track=track,
                cars=[car],
                algorithms=["HEURISTIC"],
                agents={0: HeuristicController()},
                settings={"target_laps": 1},
                seed=7,
                best_lap_time=12.0,
            )

            track_file = logger.run_directory / "tracks.jsonl"
            race_file = logger.run_directory / "races.jsonl"
            saved_tracks = [json.loads(line) for line in track_file.read_text(encoding="utf-8").splitlines()]
            saved_races = [json.loads(line) for line in race_file.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(saved_tracks), 1)
        self.assertEqual(len(saved_races), 2)
        self.assertEqual(saved_tracks[0]["centerline"], [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
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
