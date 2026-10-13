import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SHARP_CORNER_CURVATURE = 1.0 / 60.0   # raggio < 60 px


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class RaceLogger:
    """Scrive in race_logs/<run_id>/ un record JSONL per tracciato (tracks.jsonl) e uno per corsa (races.jsonl)."""

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
        track_records=None,
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
                {"track_id": track_id, **track_identity, **self._track_features(track, checkpoints)},
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
                    "race_progress": getattr(car, "progress_px", 0.0) / max(1e-9, self._spacing(track)),
                    "completed": car.laps >= settings["target_laps"],
                    "termination_reason": car.termination_reason,
                    "death": getattr(car, "death_info", None),
                    "steps": car.total_steps,
                    "decisions": getattr(car, "decisions", None),
                    "survival_seconds": car.survival_time_s,
                    "distance_traveled": car.distance_traveled,
                    "max_speed_px_s": car.max_speed_reached,
                    "average_speed_px_s": car.average_speed() if hasattr(car, "average_speed") else None,
                    "lap_times": list(getattr(car, "lap_times", [])),
                    "cumulative_reward": car.cumulative_reward,
                    "individual_reward": getattr(car, "individual_reward_total", None),
                    "progress_reward": car.progress_reward_total,
                    "time_penalty": car.time_penalty_total,
                    "terminal_penalty": car.terminal_penalty_total,
                    "collision_penalty": getattr(car, "collision_penalty_total", 0.0),
                    "completion_bonus": car.completion_bonus_total,
                    "personal_best_bonus": getattr(car, "personal_best_bonus_total", 0.0),
                    "personal_bests_beaten": getattr(car, "personal_bests_beaten", 0),
                    "collisions_caused": getattr(car, "collisions_caused", 0),
                    "collisions_suffered": getattr(car, "collisions_suffered", 0),
                    "overtakes": car.overtakes_performed,
                    "slipstream_frames": getattr(car, "slipstream_frames", 0),
                    "line": self._line_summary(car),
                    "training_respawns": getattr(car, "training_respawns", 0),
                    "training_progress": getattr(car, "training_progress_total", 0.0),
                    "training_reward": getattr(car, "training_reward_total", 0.0),
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
            "track_records": track_records,
            "settings": settings,
            "cars": car_records,
        }
        self._append_json_line(self.run_directory / "races.jsonl", race_record)

    @staticmethod
    def _line_summary(car):
        """Uso della carreggiata: valori normalizzati sulla semi-larghezza della pista.
        inside_in_corners > 0 = l'auto sta all'interno delle curve (tende alla corda), < 0 = all'esterno.
        speed_vs_ideal = velocità media / velocità del profilo ideale nello stesso punto."""
        stats = getattr(car, "line_stats", None)
        if not stats or not stats["frames"]:
            return None
        frames = stats["frames"]
        return {
            "mean_abs_lateral": stats["abs_lateral"] / frames,
            "inside_in_corners": stats["inside"] / stats["corner_frames"] if stats["corner_frames"] else None,
            "speed_vs_ideal": stats["speed_ratio"] / frames,
            "mean_racing_line_error": stats["line_error"] / frames,
            "traffic_fraction": stats["traffic_frames"] / frames,
        }

    @staticmethod
    def _spacing(track):
        spacing = getattr(track, "checkpoint_spacing", None)
        if isinstance(spacing, (int, float)) and spacing > 0:
            return spacing
        return 1.0

    @staticmethod
    def _append_json_line(path, record):
        with path.open("a", encoding="utf-8", newline="\n") as log_file:
            log_file.write(json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")

    @staticmethod
    def _track_features(track, checkpoints):
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

        features = {
            "checkpoint_spacing": {
                "minimum": min(spacings),
                "mean": sum(spacings) / len(spacings),
                "maximum": max(spacings),
            },
            "turn_angle_degrees": {
                "mean_absolute": sum(turn_angles) / len(turn_angles),
                "maximum_absolute": max(turn_angles),
            },
            "checkpoint_turn_angles_degrees": turn_angles,
        }

        curvature = getattr(track, "curvature", None)
        if curvature is not None and len(curvature) > 0:
            # Curvatura media su ~20 px per attenuare il rumore del disegno a mano.
            kernel = np.ones(11) / 11.0
            smooth = np.convolve(np.concatenate([curvature[-5:], curvature, curvature[:5]]), kernel, mode="valid")
            absolute = np.abs(smooth)
            sharp = absolute > SHARP_CORNER_CURVATURE
            corners = int(np.sum(sharp & ~np.roll(sharp, 1)))
            features.update({
                "length_px": float(track.length),
                "track_width_px": float(track.brush_radius * 2),
                "curvature": {
                    "mean_absolute": float(absolute.mean()),
                    "p90_absolute": float(np.percentile(absolute, 90)),
                    "maximum_absolute": float(absolute.max()),
                    "minimum_radius_px": float(1.0 / max(absolute.max(), 1e-9)),
                    "sharp_fraction": float(sharp.mean()),
                    "sharp_corner_count": corners,
                    "net_turning_degrees": float(math.degrees(np.sum(curvature) * track.step)),
                },
                "ideal_lap_seconds": float(getattr(track, "ideal_lap_time", float("nan"))),
                "ideal_speed_px_s": {
                    "mean": float(np.mean(track.racing_speed)),
                    "minimum": float(np.min(track.racing_speed)),
                } if getattr(track, "racing_speed", None) is not None else None,
                "racing_offset_by_checkpoint": [
                    float(chunk.mean()) for chunk in np.array_split(track.racing_offset, len(checkpoints))
                ] if getattr(track, "racing_offset", None) is not None else None,
                # Profilo di curvatura per checkpoint (segno: + = destra) per correlare le uscite di pista.
                "curvature_by_checkpoint": [
                    float(chunk.mean()) for chunk in np.array_split(smooth, len(checkpoints))
                ],
            })
        return features

    @staticmethod
    def _agent_state(agent):
        if hasattr(agent, "diagnostics"):
            return agent.diagnostics()
        return {}
