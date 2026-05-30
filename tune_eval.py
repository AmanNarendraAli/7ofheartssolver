from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from full_game_eval import unique_run_output_dir
from seven_hearts import (
    DEFAULT_MONTE_CARLO_AGENT_NAME,
    DEFAULT_WEIGHTS,
    FullGameAgent,
    StrategyWeights,
    evaluate_duplicate_deal_seat_rotation,
)


REPORT_DIR = Path("tuning_reports")
TARGET_BASELINE = "Heuristic"

WEIGHT_RANGES: dict[str, tuple[float, float]] = {
    "self_unlock": (1.0, 6.0),
    "opponent_unlock_risk": (0.5, 5.0),
    "future_chain_unseen_penalty": (0.05, 1.0),
    "future_chain_own_credit": (0.1, 2.0),
    "tail_base_credit": (0.5, 5.0),
    "tail_distance_penalty": (0.02, 0.4),
    "gate_card_credit": (0.0, 3.0),
    "opponent_runway_penalty": (0.0, 0.25),
    "race_pressure_credit": (0.0, 1.0),
    "response_penalty": (0.0, 1.0),
}


def parse_int_choices(text: str) -> tuple[int, ...]:
    choices = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not choices:
        raise argparse.ArgumentTypeError("expected at least one integer choice")
    if any(choice <= 0 for choice in choices):
        raise argparse.ArgumentTypeError("choices must be positive integers")
    return choices


def random_weights(rng: random.Random) -> StrategyWeights:
    values = asdict(DEFAULT_WEIGHTS)
    for name, (low, high) in WEIGHT_RANGES.items():
        values[name] = rng.uniform(low, high)
    return StrategyWeights(**values)


def candidate_weights(count: int, rng: random.Random) -> tuple[StrategyWeights, ...]:
    if count <= 0:
        return ()
    return (DEFAULT_WEIGHTS,) + tuple(random_weights(rng) for _ in range(count - 1))


def candidate_agents(
    mode: str,
    weights: StrategyWeights,
    samples_per_move: int,
    rollout_max_turns: int,
) -> tuple[FullGameAgent, ...]:
    if mode == "heuristic":
        candidate = FullGameAgent(DEFAULT_MONTE_CARLO_AGENT_NAME, "heuristic", weights=weights)
    elif mode == "monte-carlo":
        candidate = FullGameAgent(
            DEFAULT_MONTE_CARLO_AGENT_NAME,
            "information_limited_monte_carlo",
            weights=weights,
            samples_per_move=samples_per_move,
            rollout_max_turns=rollout_max_turns,
        )
    else:
        raise ValueError(f"unknown tuning mode: {mode}")
    return (
        candidate,
        FullGameAgent("Random", "random"),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent(TARGET_BASELINE, "heuristic"),
    )


def paired_advantage_against(evaluation: Any, baseline: str) -> Any:
    for advantage in evaluation.paired_card_advantages:
        if advantage.baseline_agent == baseline:
            return advantage
    raise ValueError(f"missing paired advantage against {baseline}")


def agent_summary(evaluation: Any, agent_name: str) -> Any:
    for summary in evaluation.agent_summaries:
        if summary.agent_name == agent_name:
            return summary
    raise ValueError(f"missing summary for {agent_name}")


def format_float(value: float) -> str:
    return f"{value:.6f}"


def write_candidate_reports(rows: Sequence[dict[str, Any]], run_dir: Path) -> tuple[Path, Path]:
    summary_path = run_dir / "candidate_summary.npz"
    parameters_path = run_dir / "candidate_parameters.npz"
    summary_fields = [
        "rank",
        "candidate_id",
        "mode",
        "score",
        "target_baseline",
        "average_advantage",
        "standard_error",
        "ci95_low",
        "ci95_high",
        "candidate_average_cards_left",
        "candidate_cards_left_standard_error",
        "candidate_win_rate",
        "candidate_rank",
        "sampled_mc_decisions",
        "samples_per_move",
        "rollout_max_turns",
        "games",
        "timeout_rate",
    ]
    summary_string_fields = {"candidate_id", "mode", "target_baseline"}
    summary_int_fields = {
        "rank",
        "sampled_mc_decisions",
        "samples_per_move",
        "rollout_max_turns",
        "games",
    }

    ranked = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    summary_columns: dict[str, np.ndarray] = {}
    for field in summary_fields:
        values = list(range(1, len(ranked) + 1)) if field == "rank" else [row[field] for row in ranked]
        if field in summary_string_fields:
            summary_columns[field] = np.array(values, dtype=str)
        elif field in summary_int_fields:
            summary_columns[field] = np.array(values, dtype=np.int64)
        else:
            summary_columns[field] = np.array(values, dtype=np.float64)
    # Load with: data = np.load("candidate_summary.npz", allow_pickle=True); data["score"]
    np.savez_compressed(summary_path, **summary_columns)

    parameter_candidate_ids: list[str] = []
    parameter_names: list[str] = []
    parameter_values: list[float] = []
    for row in rows:
        weights: StrategyWeights = row["_weights"]
        for name, value in asdict(weights).items():
            parameter_candidate_ids.append(str(row["candidate_id"]))
            parameter_names.append(f"weights.{name}")
            parameter_values.append(float(format_float(value)))
        parameter_candidate_ids.append(str(row["candidate_id"]))
        parameter_names.append("samples_per_move")
        parameter_values.append(float(row["samples_per_move"]))
        parameter_candidate_ids.append(str(row["candidate_id"]))
        parameter_names.append("rollout_max_turns")
        parameter_values.append(float(row["rollout_max_turns"]))
    # Load with: params = np.load("candidate_parameters.npz", allow_pickle=True); params["parameter"]
    np.savez_compressed(
        parameters_path,
        candidate_id=np.array(parameter_candidate_ids, dtype=str),
        parameter=np.array(parameter_names, dtype=str),
        value=np.array(parameter_values, dtype=np.float64),
    )

    return summary_path, parameters_path


def write_metadata(args: argparse.Namespace, run_dir: Path, elapsed_seconds: float) -> Path:
    path = run_dir / "run_metadata.json"
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": args.mode,
        "candidates": args.candidates,
        "deals": args.deals,
        "cards_per_suit": args.cards_per_suit,
        "max_turns": args.max_turns,
        "seed": args.seed,
        "workers": args.workers,
        "target_baseline": TARGET_BASELINE,
        "weight_ranges": WEIGHT_RANGES,
        "mc_samples_choices": list(args.mc_samples_choices),
        "mc_rollout_turn_choices": list(args.mc_rollout_turn_choices),
        "elapsed_seconds": elapsed_seconds,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def run_tuning(args: argparse.Namespace) -> tuple[Path, tuple[Path, ...]]:
    rng = random.Random(args.seed)
    weights_by_candidate = candidate_weights(args.candidates, rng)
    run_dir = unique_run_output_dir(args.output_dir)
    rows: list[dict[str, Any]] = []
    started = perf_counter()

    for index, weights in enumerate(weights_by_candidate):
        candidate_id = f"c{index:04d}"
        if args.mode == "heuristic" or index == 0:
            samples_per_move = args.samples_per_move
            rollout_max_turns = args.rollout_max_turns
        else:
            samples_per_move = rng.choice(args.mc_samples_choices)
            rollout_max_turns = rng.choice(args.mc_rollout_turn_choices)
        agents = candidate_agents(args.mode, weights, samples_per_move, rollout_max_turns)
        evaluation = evaluate_duplicate_deal_seat_rotation(
            agents,
            deals=args.deals,
            rng=random.Random(args.eval_seed),
            max_turns=args.max_turns,
            primary_agent_name=DEFAULT_MONTE_CARLO_AGENT_NAME,
            cards_per_suit=args.cards_per_suit,
            workers=args.workers,
        )
        advantage = paired_advantage_against(evaluation, TARGET_BASELINE)
        summary = agent_summary(evaluation, DEFAULT_MONTE_CARLO_AGENT_NAME)
        row = {
            "candidate_id": candidate_id,
            "mode": args.mode,
            "score": format_float(advantage.average_advantage),
            "target_baseline": TARGET_BASELINE,
            "average_advantage": format_float(advantage.average_advantage),
            "standard_error": format_float(advantage.standard_error),
            "ci95_low": format_float(advantage.ci95_low),
            "ci95_high": format_float(advantage.ci95_high),
            "candidate_average_cards_left": format_float(summary.average_cards_left),
            "candidate_cards_left_standard_error": format_float(summary.cards_left_standard_error),
            "candidate_win_rate": format_float(summary.win_rate),
            "candidate_rank": format_float(summary.average_rank),
            "sampled_mc_decisions": summary.sampled_mc_decisions,
            "samples_per_move": samples_per_move,
            "rollout_max_turns": rollout_max_turns,
            "games": evaluation.games,
            "timeout_rate": format_float(evaluation.timeout_rate),
            "_weights": weights,
        }
        rows.append(row)
        if args.progress_every > 0 and (
            index == 0 or index + 1 == len(weights_by_candidate) or (index + 1) % args.progress_every == 0
        ):
            print(
                f"[{index + 1}/{len(weights_by_candidate)}] {candidate_id} "
                f"advantage={advantage.average_advantage:+.4f} "
                f"ci95=[{advantage.ci95_low:+.4f},{advantage.ci95_high:+.4f}]",
                flush=True,
            )

    elapsed_seconds = perf_counter() - started
    paths = (*write_candidate_reports(rows, run_dir), write_metadata(args, run_dir, elapsed_seconds))
    return run_dir, paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-search tuning for full-game 7 of Hearts agents.")
    parser.add_argument("--mode", choices=("heuristic", "monte-carlo"), default="heuristic")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--deals", type=int, default=100)
    parser.add_argument("--cards-per-suit", type=int, default=13)
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23, help="seed for candidate generation")
    parser.add_argument("--eval-seed", type=int, default=7, help="seed for shared evaluation deals")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--samples-per-move", type=int, default=16)
    parser.add_argument("--rollout-max-turns", type=int, default=80)
    parser.add_argument("--mc-samples-choices", type=parse_int_choices, default=parse_int_choices("4,8,16,32,64"))
    parser.add_argument("--mc-rollout-turn-choices", type=parse_int_choices, default=parse_int_choices("40,80,120,160"))
    args = parser.parse_args()

    run_dir, paths = run_tuning(args)
    print(f"\nTuning run written to {run_dir}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
