"""Analisi aggregata dei log di gara (race_logs/<run_id>/races.jsonl + tracks.jsonl).

Uso:
    python analyze_logs.py                 # ultimo run
    python analyze_logs.py race_logs/<id>  # run specifico
    python analyze_logs.py --all           # tutti i run
    python analyze_logs.py --blocks 5      # dimensione dei blocchi per il trend di apprendimento
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_runs(paths):
    races, tracks = [], {}
    for run in paths:
        races_file = run / "races.jsonl"
        tracks_file = run / "tracks.jsonl"
        if tracks_file.exists():
            for line in tracks_file.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                tracks[record["track_id"]] = record
        if races_file.exists():
            races.extend(json.loads(line) for line in races_file.read_text(encoding="utf-8").splitlines())
    return races, tracks


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.fmean(values) if values else float("nan")


def car_progress(car, track):
    if car.get("race_progress") is not None:
        return car["race_progress"]
    return car.get("checkpoints_passed", 0)


def print_algorithm_summary(races, tracks):
    print("\n=== RIEPILOGO PER ALGORITMO (auto in gara, esclusi i respawn di addestramento) ===")
    header = f"{'ALGORITMO':<16}{'AUTO':>6}{'COMPL%':>8}{'PROGR.':>9}{'GIRI':>7}{'SURV s':>8}{'V-MED':>8}{'COLL.':>7}{'TRAIN/RESP':>12}"
    print(header)
    print("-" * len(header))
    by_algo = defaultdict(list)
    for race in races:
        for car in race["cars"]:
            by_algo[car["algorithm"]].append((race, car))
    for algo, entries in sorted(by_algo.items()):
        cars = [car for _, car in entries]
        completed = sum(1 for car in cars if car.get("completed"))
        respawns = sum(car.get("training_respawns", 0) for car in cars)
        training = sum(car.get("training_progress", 0.0) for car in cars)
        print(
            f"{algo:<16}{len(cars):>6}{100.0 * completed / len(cars):>7.1f}%"
            f"{mean([car_progress(car, tracks.get(race['track_id'])) for race, car in entries]):>9.1f}"
            f"{mean([car.get('laps') for car in cars]):>7.2f}"
            f"{mean([car.get('survival_seconds') for car in cars]):>8.1f}"
            f"{mean([car.get('average_speed_px_s') for car in cars]):>8.0f}"
            f"{mean([car.get('collisions_caused') for car in cars]):>7.2f}"
            f"{(training / respawns if respawns else float('nan')):>12.1f}"
        )

    print("\n=== MOTIVI DI USCITA (%) ===")
    for algo, entries in sorted(by_algo.items()):
        reasons = Counter(car["termination_reason"] for _, car in entries)
        total = sum(reasons.values())
        print(f"{algo:<16}" + "  ".join(f"{reason}: {100.0 * count / total:.0f}%" for reason, count in reasons.most_common()))


def print_line_analysis(races, tracks):
    print("\n=== TRAIETTORIA E VELOCITÀ (auto in gara; valori normalizzati sulla semi-larghezza) ===")
    print("  |lat| = distanza media dalla mezzeria | interno = posizione verso la corda in curva (>0 interno)")
    print("  v/ideale = velocità rispetto al profilo limite della linea ideale | traffico = quota di tempo con un'auto vicina davanti")
    header = f"{'ALGORITMO':<16}{'|lat|':>8}{'interno':>9}{'err.linea':>11}{'v/ideale':>10}{'traffico':>10}{'sorpassi':>10}"
    print(header)
    by_algo = defaultdict(list)
    for race in races:
        for car in race["cars"]:
            by_algo[car["algorithm"]].append(car)
    for algo, cars in sorted(by_algo.items()):
        lines = [car.get("line") for car in cars if car.get("line")]
        if not lines:
            continue
        print(
            f"{algo:<16}{mean([l['mean_abs_lateral'] for l in lines]):>8.2f}"
            f"{mean([l['inside_in_corners'] for l in lines]):>9.2f}"
            f"{mean([l['mean_racing_line_error'] for l in lines]):>11.2f}"
            f"{mean([l['speed_vs_ideal'] for l in lines]):>10.2f}"
            f"{mean([l['traffic_fraction'] for l in lines]):>10.2f}"
            f"{mean([car.get('overtakes') for car in cars]):>10.1f}"
        )

    print("\n=== GIRO MIGLIORE PER TRACCIATO vs GIRO TEORICO DELLA LINEA IDEALE ===")
    for track_id, track in tracks.items():
        ideal = track.get("ideal_lap_seconds")
        laps = defaultdict(list)
        for race in races:
            if race["track_id"] != track_id:
                continue
            for car in race["cars"]:
                laps[car["algorithm"]].extend(car.get("lap_times", [])[1:])   # esclude il giro di lancio
        text = "  ".join(f"{algo}: {min(values):.2f}s" for algo, values in sorted(laps.items()) if values)
        print(f"  {track_id[:8]} teorico {ideal if ideal else float('nan'):.2f}s | {text}")


def print_learning_trend(races, block):
    print(f"\n=== TREND DI APPRENDIMENTO (progresso medio in checkpoint per blocchi di {block} corse) ===")
    races = sorted(races, key=lambda r: (r.get("run_id", ""), r.get("generation", 0)))
    algorithms = sorted({car["algorithm"] for race in races for car in race["cars"]})
    blocks = [races[i:i + block] for i in range(0, len(races), block)]
    print(f"{'ALGORITMO':<16}" + "".join(f"{f'B{i + 1}':>8}" for i in range(len(blocks))))
    for algo in algorithms:
        row = []
        for chunk in blocks:
            row.append(mean([car_progress(car, None) for race in chunk for car in race["cars"] if car["algorithm"] == algo]))
        print(f"{algo:<16}" + "".join(f"{value:>8.1f}" for value in row))


def print_death_analysis(races, tracks):
    print("\n=== DOVE ESCONO DI PISTA (per tracciato, solo uscite con posizione registrata) ===")
    by_track = defaultdict(list)
    for race in races:
        for car in race["cars"]:
            if car.get("death"):
                by_track[race["track_id"]].append((car["algorithm"], car["termination_reason"], car["death"]))
    for track_id, deaths in by_track.items():
        track = tracks.get(track_id, {})
        curvature = track.get("curvature_by_checkpoint") or []
        count = len(curvature) or 50
        print(f"\nTracciato {track_id[:8]}  lunghezza {track.get('length_px', float('nan')):.0f} px, "
              f"larghezza {track.get('track_width_px', float('nan')):.0f} px, "
              f"curve strette {track.get('curvature', {}).get('sharp_corner_count', 'n/d')}")
        histogram = Counter(int(d["track_fraction"] * count) % count for _, _, d in deaths)
        print("  Checkpoint con più uscite (indice: uscite, curvatura media, velocità media all'uscita):")
        for index, hits in histogram.most_common(8):
            speeds = [d["speed_px_s"] for _, _, d in deaths if int(d["track_fraction"] * count) % count == index]
            radius = 1.0 / abs(curvature[index]) if curvature and abs(curvature[index]) > 1e-6 else float("inf")
            print(f"    cp {index:>2}: {hits:>4}  raggio ~{radius:>6.0f} px  v {mean(speeds):>5.0f} px/s")
        per_algo = defaultdict(list)
        for algo, _, death in deaths:
            per_algo[algo].append(death)
        print("  Per algoritmo: velocità media all'uscita | offset laterale medio | errore di direzione medio")
        for algo, items in sorted(per_algo.items()):
            print(f"    {algo:<16} {mean([d['speed_px_s'] for d in items]):>6.0f} px/s | "
                  f"{mean([d['lateral_offset_px'] for d in items]):>6.1f} px | "
                  f"{mean([d['heading_error_deg'] for d in items]):>6.1f} deg")


def print_model_state(races):
    print("\n=== STATO DEI MODELLI NELL'ULTIMA CORSA ===")
    last = max(races, key=lambda r: (r.get("run_id", ""), r.get("generation", 0)))
    seen = set()
    for car in last["cars"]:
        key = (car["algorithm"], car["team_id"])
        if key in seen:
            continue
        seen.add(key)
        state = {k: v for k, v in (car.get("agent_state") or {}).items() if k != "assignments"}
        print(f"  team {car['team_id']} {car['algorithm']:<16} {state}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", help="cartella del run (default: ultimo)")
    parser.add_argument("--root", default="race_logs")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--blocks", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    if args.run:
        runs = [Path(args.run)]
    else:
        runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
        if not args.all:
            runs = runs[-1:]
    races, tracks = load_runs(runs)
    if not races:
        raise SystemExit("Nessuna corsa trovata.")
    print(f"Run analizzati: {', '.join(r.name for r in runs)} | corse: {len(races)} | tracciati: {len(tracks)}")
    print_algorithm_summary(races, tracks)
    print_line_analysis(races, tracks)
    print_learning_trend(races, args.blocks)
    print_death_analysis(races, tracks)
    print_model_state(races)


if __name__ == "__main__":
    main()
