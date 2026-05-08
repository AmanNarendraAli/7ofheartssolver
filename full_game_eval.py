from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from time import perf_counter
from typing import Callable

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


def write_summary_csv(evaluation: DuplicateDealEvaluation, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    evaluation = evaluate_duplicate_deal_seat_rotation(
        default_agents(args.samples_per_move, args.rollout_max_turns, fast=args.fast),
        deals=args.deals,
        rng=random.Random(args.seed),
        max_turns=args.max_turns,
        primary_agent_name="Ours",
        progress_callback=make_progress_reporter(args.progress_every),
        cards_per_suit=args.cards_per_suit,
        workers=args.workers,
    )
    paths = write_summary_csv(evaluation, args.output_dir)
    print_summary(evaluation, paths, perf_counter() - started)


if __name__ == "__main__":
    main()
