import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path


class RaceLogger:
    def __init__(self, root="race_logs"):
        self.started_at = datetime.now(timezone.utc)
        self.run_id = f"{self.started_at:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        self.run_directory = Path(root) / self.run_id
        self._written_tracks = set()

    def record_race(
        self,
        generation,
        reason,
        duration_ms,
        track,
        cars,
        algorithms,
        agents,
        settings,
        seed,
        best_lap_time,
    ):
        centerline = [[float(x), float(y)] for x, y in track.drawing_points]
        checkpoints = [[float(x), float(y)] for x, y in track.checkpoints]
        track_identity = {
            "width": track.width,
            "height": track.height,
            "brush_radius": track.brush_radius,
            "centerline": centerline,
            "checkpoints": checkpoints,
        }
        track_id = hashlib.sha256(
            json.dumps(track_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        self.run_directory.mkdir(parents=True, exist_ok=True)
        if track_id not in self._written_tracks:
            self._append_json_line(
                self.run_directory / "tracks.jsonl",
                {"track_id": track_id, **track_identity, **self._track_features(checkpoints)},
            )
            self._written_tracks.add(track_id)

        car_records = []
        for car in sorted(cars, key=lambda item: item.car_id):
            agent = agents[car.car_id]
            car_records.append(
                {
                    "car_id": car.car_id,
                    "team_id": car.team_id,
                    "algorithm": algorithms[car.team_id],
                    "laps": car.laps,
                    "checkpoints_passed": car.checkpoints_passed_total,
                    "completed": car.laps >= settings["target_laps"],
                    "termination_reason": car.termination_reason,
                    "steps": car.total_steps,
                    "survival_seconds": car.survival_time_s,
                    "distance_traveled": car.distance_traveled,
                    "max_speed_px_s": car.max_speed_reached,
                    "cumulative_reward": car.cumulative_reward,
                    "progress_reward": car.progress_reward_total,
                    "time_penalty": car.time_penalty_total,
                    "terminal_penalty": car.terminal_penalty_total,
                    "completion_bonus": car.completion_bonus_total,
                    "overtakes": car.overtakes_performed,
                    "agent_state": self._agent_state(agent),
                }
            )

        race_record = {
            "run_id": self.run_id,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
            "track_id": track_id,
            "reason": reason,
            "duration_seconds": duration_ms / 1000.0,
            "best_lap_seconds": best_lap_time if math.isfinite(best_lap_time) else None,
            "settings": settings,
            "cars": car_records,
        }
        self._append_json_line(self.run_directory / "races.jsonl", race_record)

    @staticmethod
    def _append_json_line(path, record):
        with path.open("a", encoding="utf-8", newline="\n") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    @staticmethod
    def _track_features(checkpoints):
        if len(checkpoints) < 2:
            return {"checkpoint_spacing": {}, "turn_angle_degrees": {}}

        headings = []
        spacings = []
        for index, point in enumerate(checkpoints):
            next_point = checkpoints[(index + 1) % len(checkpoints)]
            dx = next_point[0] - point[0]
            dy = next_point[1] - point[1]
            spacings.append(math.hypot(dx, dy))
            headings.append(math.atan2(dy, dx))

        turn_angles = []
        for index, heading in enumerate(headings):
            delta = (headings[(index + 1) % len(headings)] - heading + math.pi) % (
                2.0 * math.pi
            ) - math.pi
            turn_angles.append(abs(math.degrees(delta)))

        return {
            "checkpoint_spacing": {
                "minimum": min(spacings),
                "mean": sum(spacings) / len(spacings),
                "maximum": max(spacings),
            },
            "turn_angle_degrees": {
                "mean_absolute": sum(turn_angles) / len(turn_angles),
                "maximum_absolute": max(turn_angles),
            },
        }

    @staticmethod
    def _agent_state(agent):
        state = {}
        if hasattr(agent, "epsilon"):
            state["epsilon"] = float(agent.epsilon)
        if hasattr(agent, "memory"):
            state["replay_size"] = len(agent.memory)
            state["optimizer_steps"] = int(agent.steps_counter)
            state["training_calls"] = int(agent.training_calls)
        if hasattr(agent, "q_table"):
            state["q_table_states"] = len(agent.q_table)
        if hasattr(agent, "saved_log_probs"):
            state["pending_policy_actions"] = len(agent.saved_log_probs)
        if hasattr(agent, "last_grad_norm"):
            state["last_grad_norm"] = float(agent.last_grad_norm)
        if hasattr(agent, "last_td_error"):
            state["last_td_error"] = float(agent.last_td_error)
        if hasattr(agent, "last_mutation_norm"):
            state["last_mutation_norm"] = float(agent.last_mutation_norm)
        for attribute in (
            "lr",
            "learning_rate",
            "gamma",
            "epsilon_decay",
            "min_epsilon",
            "minimum_replay_size",
            "batch_size",
            "training_interval",
            "target_update_interval",
            "entropy_coefficient",
            "max_grad_norm",
            "mutation_rate",
            "mutation_fraction",
        ):
            if hasattr(agent, attribute):
                state[attribute] = getattr(agent, attribute)
        model = getattr(agent, "model", getattr(agent, "policy", None))
        if "last_grad_norm" not in state and model is not None and hasattr(model, "last_grad_norm"):
            state["last_grad_norm"] = float(model.last_grad_norm)
        return state
