from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from seven_hearts import (
    DuplicateDealEvaluation,
    FullGameAgent,
    FullGameResult,
    evaluate_duplicate_deal_seat_rotation,
)


REPORT_DIR = Path("full_game_reports")


def default_agents(samples_per_move: int, rollout_max_turns: int, fast: bool = False) -> tuple[FullGameAgent, ...]:
    ours = (
        FullGameAgent("Ours", "heuristic")
        if fast
        else FullGameAgent(
            "Ours",
            "information_limited_monte_carlo",
            samples_per_move=samples_per_move,
            rollout_max_turns=rollout_max_turns,
        )
    )
    return (
        ours,
        FullGameAgent("Random", "random"),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent("Heuristic", "heuristic"),
    )


def make_progress_reporter(
    every: int,
) -> Callable[[int, int, int, int, FullGameResult], None] | None:
    if every <= 0:
        return None

    started = perf_counter()

    def report(
        completed_games: int,
        total_games: int,
        deal_index: int,
        rotation_index: int,
        result: FullGameResult,
    ) -> None:
        if completed_games != 1 and completed_games != total_games and completed_games % every != 0:
            return
        elapsed = perf_counter() - started
        rate = completed_games / elapsed if elapsed > 0 else 0.0
        remaining = (total_games - completed_games) / rate if rate > 0 else 0.0
        winner = result.agent_names_by_seat[result.winner] if result.winner is not None else "none"
        print(
            f"[{completed_games}/{total_games}] deal={deal_index} rotation={rotation_index} "
            f"winner={winner} turns={result.turns_played} timeout={result.timed_out} "
            f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
            flush=True,
        )

    return report


def unique_run_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"run_{timestamp}_pid{os.getpid()}"
    for suffix in range(1, 10_000):
        name = stem if suffix == 1 else f"{stem}_{suffix}"
        run_dir = base_dir / name
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create unique run directory under {base_dir}")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def agent_metadata(agent: FullGameAgent) -> dict[str, Any]:
    uses_monte_carlo = agent.policy == "information_limited_monte_carlo"
    uses_heuristic_weights = agent.policy in {
        "heuristic",
        "information_limited_monte_carlo",
        "oracle_greedy",
    }
    metadata: dict[str, Any] = {
        "name": agent.name,
        "policy": agent.policy,
        "uses_monte_carlo": uses_monte_carlo,
        "uses_heuristic_weights": uses_heuristic_weights,
    }
    if uses_monte_carlo:
        metadata["monte_carlo"] = {
            "samples_per_move": agent.samples_per_move,
            "rollout_max_turns": agent.rollout_max_turns,
            "rollout_policy": agent.rollout_policy,
            "rationality": agent.rationality,
            "score_weights": {
                "monte_carlo_win_rate_weight": agent.weights.monte_carlo_win_rate_weight,
                "monte_carlo_timeout_penalty": agent.weights.monte_carlo_timeout_penalty,
            },
        }
    if uses_heuristic_weights:
        metadata["heuristic_weights"] = heuristic_weight_metadata(agent.weights)
    return metadata


def heuristic_weight_metadata(weights: Any) -> dict[str, Any]:
    return {
        name: value
        for name, value in asdict(weights).items()
        if not name.startswith("monte_carlo_")
    }


def active_agent_parameter_rows(agent: FullGameAgent) -> list[list[Any]]:
    metadata = agent_metadata(agent)
    rows: list[list[Any]] = [
        [agent.name, agent.policy, "policy", agent.policy],
    ]
    if metadata["uses_monte_carlo"]:
        monte_carlo = metadata["monte_carlo"]
        rows.extend(
            [
                [agent.name, agent.policy, "monte_carlo.samples_per_move", monte_carlo["samples_per_move"]],
                [agent.name, agent.policy, "monte_carlo.rollout_max_turns", monte_carlo["rollout_max_turns"]],
                [agent.name, agent.policy, "monte_carlo.rollout_policy", monte_carlo["rollout_policy"]],
                [agent.name, agent.policy, "monte_carlo.rationality", monte_carlo["rationality"]],
                [
                    agent.name,
                    agent.policy,
                    "monte_carlo.score_weights.monte_carlo_win_rate_weight",
                    monte_carlo["score_weights"]["monte_carlo_win_rate_weight"],
                ],
                [
                    agent.name,
                    agent.policy,
                    "monte_carlo.score_weights.monte_carlo_timeout_penalty",
                    monte_carlo["score_weights"]["monte_carlo_timeout_penalty"],
                ],
            ]
        )
    if metadata["uses_heuristic_weights"]:
        rows.extend(
            [agent.name, agent.policy, f"heuristic_weights.{name}", value]
            for name, value in heuristic_weight_metadata(agent.weights).items()
        )
    return rows


def write_agent_parameters_csv(agents: tuple[FullGameAgent, ...], output_dir: Path) -> Path:
    path = output_dir / "agent_parameters.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["agent", "policy", "parameter", "value"])
        for agent in agents:
            writer.writerows(active_agent_parameter_rows(agent))
    return path


def build_global_parameter_metadata(args: argparse.Namespace, output_dir: Path, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "output_dir": str(output_dir),
        "eval_args": json_safe(vars(args)),
        "duplicate_deal": {
            "primary_agent_name": "Ours",
            "rotations_per_deal": 4,
        },
        "implementation_defaults": {
            "information_limited_policy_cache_max_entries": 50_000,
            "rollout_transposition_cache_max_entries": 50_000,
        },
        "elapsed_seconds": elapsed_seconds,
    }


def write_global_parameters_csv(metadata: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "run_parameters.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["parameter", "value"])
        writer.writerows(flatten_metadata("", metadata))
    return path


def build_run_metadata(
    args: argparse.Namespace,
    agents: tuple[FullGameAgent, ...],
    output_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    metadata = build_global_parameter_metadata(args, output_dir, elapsed_seconds)
    metadata["agents"] = [agent_metadata(agent) for agent in agents]
    return metadata


def flatten_metadata(prefix: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_metadata(next_prefix, item))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            rows.extend(flatten_metadata(next_prefix, item))
        return rows
    return [(prefix, str(value))]


def write_metadata_reports(
    metadata: dict[str, Any],
    agents: tuple[FullGameAgent, ...],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    json_path = output_dir / "run_metadata.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    global_metadata = {key: value for key, value in metadata.items() if key != "agents"}
    global_csv_path = write_global_parameters_csv(global_metadata, output_dir)
    agent_csv_path = write_agent_parameters_csv(agents, output_dir)
    return json_path, global_csv_path, agent_csv_path


def write_summary_csv(evaluation: DuplicateDealEvaluation, output_dir: Path) -> tuple[Path, Path, Path]:
    agents_path = output_dir / "agent_summary.csv"
    paired_path = output_dir / "paired_card_advantage.csv"
    games_path = output_dir / "games.csv"

    with agents_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["agent", "seats_played", "win_rate", "average_cards_left", "average_rank"])
        for summary in evaluation.agent_summaries:
            writer.writerow(
                [
                    summary.agent_name,
                    summary.seats_played,
                    f"{summary.win_rate:.6f}",
                    f"{summary.average_cards_left:.6f}",
                    f"{summary.average_rank:.6f}",
                ]
            )

    with paired_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["primary_agent", "baseline_agent", "comparisons", "average_advantage", "standard_error"])
        for advantage in evaluation.paired_card_advantages:
            writer.writerow(
                [
                    advantage.primary_agent,
                    advantage.baseline_agent,
                    advantage.comparisons,
                    f"{advantage.average_advantage:.6f}",
                    f"{advantage.standard_error:.6f}",
                ]
            )

    with games_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "deal_index",
                "rotation_index",
                "seed",
                "seat_0_agent",
                "seat_1_agent",
                "seat_2_agent",
                "seat_3_agent",
                "winner_seat",
                "winner_agent",
                "finish_order",
                "seat_0_rank",
                "seat_1_rank",
                "seat_2_rank",
                "seat_3_rank",
                "seat_0_cards_left",
                "seat_1_cards_left",
                "seat_2_cards_left",
                "seat_3_cards_left",
                "turns_played",
                "timed_out",
            ]
        )
        for game_result in evaluation.game_results:
            result = game_result.result
            winner_agent = (
                result.agent_names_by_seat[result.winner] if result.winner is not None else ""
            )
            writer.writerow(
                [
                    game_result.deal_index,
                    game_result.rotation_index,
                    game_result.seed,
                    *result.agent_names_by_seat,
                    "" if result.winner is None else result.winner,
                    winner_agent,
                    " ".join(str(player) for player in result.finish_order),
                    *result.ranks_by_seat,
                    *result.final_hand_counts,
                    result.turns_played,
                    int(result.timed_out),
                ]
            )

    return agents_path, paired_path, games_path


def print_summary(evaluation: DuplicateDealEvaluation, paths: tuple[Path, ...], elapsed_seconds: float) -> None:
    print(
        f"Full-game duplicate-deal evaluation: {evaluation.deals} deals, "
        f"{evaluation.rotations_per_deal} rotations/deal, {evaluation.games} games"
    )
    print(f"average turns: {evaluation.average_turns:.1f}, timeouts: {evaluation.timeout_rate:.1%}")
    print(f"total elapsed: {elapsed_seconds:.1f}s")
    print("\nAgent summaries:")
    for summary in evaluation.agent_summaries:
        print(
            f"  {summary.agent_name}: cards_left={summary.average_cards_left:.3f}, "
            f"rank={summary.average_rank:.3f}, win={summary.win_rate:.1%}, seats={summary.seats_played}"
        )

    print("\nPaired card advantage:")
    for advantage in evaluation.paired_card_advantages:
        print(
            f"  {advantage.primary_agent} vs {advantage.baseline_agent}: "
            f"{advantage.average_advantage:+.3f} +/- {advantage.standard_error:.3f} "
            f"over {advantage.comparisons} comparisons"
        )

    print("\nWrote reports:")
    for path in paths:
        print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run duplicate-deal, seat-rotated full-game evaluation.")
    parser.add_argument("--deals", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--samples-per-move", type=int, default=2)
    parser.add_argument("--rollout-max-turns", type=int, default=40)
    parser.add_argument(
        "--cards-per-suit",
        type=int,
        default=13,
        help="use a centered reduced deck; 5 means ranks 5 through 9 in every suit",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use the heuristic policy for Ours instead of belief-state Monte Carlo",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print progress every N completed games; use 0 to silence progress",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes for independent duplicate-deal rotation games",
    )
    args = parser.parse_args()

    started = perf_counter()
    agents = default_agents(args.samples_per_move, args.rollout_max_turns, fast=args.fast)
    evaluation = evaluate_duplicate_deal_seat_rotation(
        agents,
        deals=args.deals,
        rng=random.Random(args.seed),
        max_turns=args.max_turns,
        primary_agent_name="Ours",
        progress_callback=make_progress_reporter(args.progress_every),
        cards_per_suit=args.cards_per_suit,
        workers=args.workers,
    )
    elapsed_seconds = perf_counter() - started
    run_dir = unique_run_output_dir(args.output_dir)
    metadata = build_run_metadata(args, agents, run_dir, elapsed_seconds)
    paths = (*write_summary_csv(evaluation, run_dir), *write_metadata_reports(metadata, agents, run_dir))
    print_summary(evaluation, paths, elapsed_seconds)


if __name__ == "__main__":
    main()
