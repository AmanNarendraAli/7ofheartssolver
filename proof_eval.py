from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seven_hearts import (
    Card,
    GameState,
    MonteCarloMoveScore,
    PlayerKnowledge,
    Suit,
    SuitRun,
    build_opponent_model,
    evaluate_move_monte_carlo,
    labels,
    recommend_move_exact_imperfect_information,
    score_move,
)


REPORT_DIR = Path("proof_reports")


@dataclass(frozen=True)
class ProofEvaluationScenario:
    name: str
    state: GameState
    knowledge: PlayerKnowledge


@dataclass(frozen=True)
class MoveEvaluation:
    card: Card
    exact_ev: float
    exact_ev_vector: tuple[float, float, float, float]
    heuristic_score: float
    heuristic_components: dict[str, float]
    monte_carlo_score: float | None = None
    monte_carlo_win_rate: float | None = None
    monte_carlo_standard_error: float | None = None


@dataclass(frozen=True)
class EngineEvaluation:
    scenario: ProofEvaluationScenario
    move_evaluations: tuple[MoveEvaluation, ...]
    exact_choice: Card
    heuristic_choice: Card
    monte_carlo_choice: Card | None
    exact_best_ev: float
    heuristic_regret: float
    monte_carlo_regret: float | None
    hidden_deals: int
    exhaustive: bool
    continuation_model: str


def cards(text: str) -> frozenset[Card]:
    return frozenset(Card.parse(part) for part in text.split())


def evaluation_scenarios() -> tuple[ProofEvaluationScenario, ...]:
    reduced_state = GameState(
        table={suit: SuitRun() for suit in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(3, 2, 1, 1),
        current_player=0,
    )
    reduced_knowledge = PlayerKnowledge(
        player=0,
        hand=cards("5H 6H 8H"),
        deck=cards("5H 6H 7H 8H 9H 7C AC AD"),
    )

    multi_suit_table = {
        suit: SuitRun()
        for suit in Suit
    } | {
        Suit.HEARTS: SuitRun(low=6, high=8),
        Suit.CLUBS: SuitRun(low=7, high=7),
    }
    multi_suit_state = GameState(
        table=multi_suit_table,
        hand_counts=(4, 2, 2, 2),
        current_player=0,
    )
    multi_suit_knowledge = PlayerKnowledge(
        player=0,
        hand=cards("5H 9H 6C 8C"),
        deck=cards("4H 5H 6H 7H 8H 9H 10H 6C 7C 8C 7D 7S AC AD"),
    )

    return (
        ProofEvaluationScenario("reduced_belief_ev", reduced_state, reduced_knowledge),
        ProofEvaluationScenario("multi_suit_belief_ev", multi_suit_state, multi_suit_knowledge),
    )


def evaluate_scenario(
    scenario: ProofEvaluationScenario,
    samples_per_move: int = 80,
    max_turns: int = 200,
    seed: int = 7,
    include_monte_carlo: bool = True,
) -> EngineEvaluation:
    exact = recommend_move_exact_imperfect_information(scenario.state, scenario.knowledge)
    if exact is None or exact.chosen_move is None:
        raise ValueError(f"scenario {scenario.name} has no exact move to evaluate")

    exact_by_card = {score.card: score.expected_value for score in exact.move_scores}
    legal = tuple(score.card for score in exact.move_scores)
    model = build_opponent_model(scenario.state, scenario.knowledge)
    scored_moves = {
        card: score_move(scenario.state, scenario.knowledge, card, model)
        for card in legal
    }
    heuristic_scores = {card: scored.score for card, scored in scored_moves.items()}
    heuristic_choice = max(legal, key=lambda card: heuristic_scores[card])

    monte_carlo_scores: dict[Card, MonteCarloMoveScore] = {}
    if include_monte_carlo:
        rng = random.Random(seed)
        monte_carlo_scores = {
            card: evaluate_move_monte_carlo(
                scenario.state,
                scenario.knowledge,
                card,
                samples=samples_per_move,
                max_turns=max_turns,
                rng=rng,
            )
            for card in legal
        }
        monte_carlo_choice = max(legal, key=lambda card: monte_carlo_scores[card].score)
    else:
        monte_carlo_choice = None

    move_evaluations = tuple(
        MoveEvaluation(
            card=card,
            exact_ev=exact_by_card[card],
            exact_ev_vector=next(score.expected_value_vector for score in exact.move_scores if score.card == card),
            heuristic_score=heuristic_scores[card],
            heuristic_components=dict(scored_moves[card].components),
            monte_carlo_score=monte_carlo_scores[card].score if card in monte_carlo_scores else None,
            monte_carlo_win_rate=monte_carlo_scores[card].win_rate if card in monte_carlo_scores else None,
            monte_carlo_standard_error=(
                monte_carlo_scores[card].win_rate_standard_error if card in monte_carlo_scores else None
            ),
        )
        for card in legal
    )
    best_ev = max(exact_by_card.values())
    monte_carlo_regret = None
    if monte_carlo_choice is not None:
        monte_carlo_regret = best_ev - exact_by_card[monte_carlo_choice]

    return EngineEvaluation(
        scenario=scenario,
        move_evaluations=move_evaluations,
        exact_choice=exact.chosen_move,
        heuristic_choice=heuristic_choice,
        monte_carlo_choice=monte_carlo_choice,
        exact_best_ev=best_ev,
        heuristic_regret=best_ev - exact_by_card[heuristic_choice],
        monte_carlo_regret=monte_carlo_regret,
        hidden_deals=exact.hidden_deals,
        exhaustive=exact.exhaustive,
        continuation_model=exact.continuation_model,
    )


def render_reports(evaluations: tuple[EngineEvaluation, ...], output_dir: Path = REPORT_DIR) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [write_summary_csv(evaluations, output_dir), write_move_csv(evaluations, output_dir)]
    paths.append(plot_regret_summary(evaluations, output_dir))
    for evaluation in evaluations:
        paths.append(plot_scenario_dashboard(evaluation, output_dir))
    return tuple(paths)


def write_summary_csv(evaluations: tuple[EngineEvaluation, ...], output_dir: Path) -> Path:
    path = output_dir / "engine_quality_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "scenario",
                "hidden_deals",
                "exhaustive",
                "exact_choice",
                "heuristic_choice",
                "monte_carlo_choice",
                "heuristic_regret",
                "monte_carlo_regret",
            ]
        )
        for evaluation in evaluations:
            writer.writerow(
                [
                    evaluation.scenario.name,
                    evaluation.hidden_deals,
                    evaluation.exhaustive,
                    evaluation.exact_choice.label(),
                    evaluation.heuristic_choice.label(),
                    evaluation.monte_carlo_choice.label() if evaluation.monte_carlo_choice else "",
                    f"{evaluation.heuristic_regret:.6f}",
                    f"{evaluation.monte_carlo_regret:.6f}" if evaluation.monte_carlo_regret is not None else "",
                ]
            )
    return path


def write_move_csv(evaluations: tuple[EngineEvaluation, ...], output_dir: Path) -> Path:
    path = output_dir / "engine_quality_moves.csv"
    component_names = sorted(
        {
            name
            for evaluation in evaluations
            for move in evaluation.move_evaluations
            for name in move.heuristic_components
        }
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "scenario",
                "move",
                "is_exact_choice",
                "is_heuristic_choice",
                "is_monte_carlo_choice",
                "exact_ev",
                "exact_ev_p0",
                "exact_ev_p1",
                "exact_ev_p2",
                "exact_ev_p3",
                "heuristic_score",
                "monte_carlo_score",
                "monte_carlo_win_rate",
                "monte_carlo_standard_error",
            ]
            + [f"heuristic_{name}" for name in component_names]
        )
        for evaluation in evaluations:
            for move in evaluation.move_evaluations:
                writer.writerow(
                    [
                        evaluation.scenario.name,
                        move.card.label(),
                        move.card == evaluation.exact_choice,
                        move.card == evaluation.heuristic_choice,
                        move.card == evaluation.monte_carlo_choice,
                        f"{move.exact_ev:.6f}",
                        *(f"{value:.6f}" for value in move.exact_ev_vector),
                        f"{move.heuristic_score:.6f}",
                        f"{move.monte_carlo_score:.6f}" if move.monte_carlo_score is not None else "",
                        f"{move.monte_carlo_win_rate:.6f}" if move.monte_carlo_win_rate is not None else "",
                        (
                            f"{move.monte_carlo_standard_error:.6f}"
                            if move.monte_carlo_standard_error is not None
                            else ""
                        ),
                        *(f"{move.heuristic_components.get(name, 0.0):.6f}" for name in component_names),
                    ]
                )
    return path


def plot_regret_summary(evaluations: tuple[EngineEvaluation, ...], output_dir: Path) -> Path:
    names = [evaluation.scenario.name.replace("_", "\n") for evaluation in evaluations]
    heuristic = [evaluation.heuristic_regret for evaluation in evaluations]
    monte_carlo = [evaluation.monte_carlo_regret or 0.0 for evaluation in evaluations]
    x_positions = range(len(evaluations))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.bar([x - width / 2 for x in x_positions], heuristic, width, label="Heuristic", color="#d95f02")
    ax.bar([x + width / 2 for x in x_positions], monte_carlo, width, label="Monte Carlo", color="#1b9e77")
    ax.set_title("Engine Regret Against Exact Hidden-Deal Oracle", fontsize=15, weight="bold")
    ax.set_ylabel("Exact EV Regret")
    ax.set_xticks(list(x_positions), names)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for index, evaluation in enumerate(evaluations):
        ax.text(
            index,
            max(heuristic[index], monte_carlo[index]) + 0.015,
            f"exact {evaluation.exact_choice.label()}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    path = output_dir / "engine_regret_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scenario_dashboard(evaluation: EngineEvaluation, output_dir: Path) -> Path:
    moves = [move.card.label() for move in evaluation.move_evaluations]
    exact_values = [move.exact_ev for move in evaluation.move_evaluations]
    heuristic_scores = [move.heuristic_score for move in evaluation.move_evaluations]
    mc_rates = [move.monte_carlo_win_rate or 0.0 for move in evaluation.move_evaluations]
    mc_errors = [move.monte_carlo_standard_error or 0.0 for move in evaluation.move_evaluations]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        f"{evaluation.scenario.name.replace('_', ' ').title()}",
        fontsize=16,
        weight="bold",
    )

    plot_metric_bars(
        axes[0][0],
        moves,
        exact_values,
        "Exact EV by Move",
        "EV",
        evaluation.exact_choice,
        [move.card for move in evaluation.move_evaluations],
        "#4c78a8",
    )
    plot_metric_bars(
        axes[0][1],
        moves,
        heuristic_scores,
        "Heuristic Scores",
        "score",
        evaluation.heuristic_choice,
        [move.card for move in evaluation.move_evaluations],
        "#f58518",
    )
    plot_metric_bars(
        axes[1][0],
        moves,
        mc_rates,
        "Monte Carlo Win Rate (score choice marked)",
        "win rate",
        evaluation.monte_carlo_choice,
        [move.card for move in evaluation.move_evaluations],
        "#54a24b",
        yerr=mc_errors,
    )

    axes[1][1].axis("off")
    summary = [
        f"hidden deals: {evaluation.hidden_deals}",
        f"exhaustive exact proof: {evaluation.exhaustive}",
        "continuation:",
        "full-info rational",
        "after materialization",
        f"exact choice: {evaluation.exact_choice.label()}",
        f"heuristic choice: {evaluation.heuristic_choice.label()}",
        f"heuristic regret: {evaluation.heuristic_regret:.3f}",
    ]
    if evaluation.monte_carlo_choice is not None and evaluation.monte_carlo_regret is not None:
        summary.extend(
            [
                f"monte carlo choice: {evaluation.monte_carlo_choice.label()}",
                f"monte carlo regret: {evaluation.monte_carlo_regret:.3f}",
            ]
        )
    axes[1][1].text(
        0.02,
        0.98,
        "\n".join(summary),
        va="top",
        ha="left",
        fontsize=13,
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f7f7", "edgecolor": "#dddddd"},
    )

    path = output_dir / f"{evaluation.scenario.name}_dashboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_metric_bars(
    ax: plt.Axes,
    labels_: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    chosen_card: Card | None,
    cards_: list[Card],
    color: str,
    yerr: list[float] | None = None,
) -> None:
    colors = ["#222222" if card == chosen_card else color for card in cards_]
    ax.bar(labels_, values, yerr=yerr, color=colors, alpha=0.88, capsize=4)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="x", rotation=0)

    low = min(values + [0.0])
    high = max(values + [0.0])
    span = high - low or 1.0
    padding = span * 0.18
    ax.set_ylim(low - padding, high + padding)

    for index, value in enumerate(values):
        offset = span * 0.04
        if value >= 0:
            ax.text(index, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(index, value - offset, f"{value:.3f}", ha="center", va="top", fontsize=9)


def print_console_summary(evaluations: tuple[EngineEvaluation, ...], paths: tuple[Path, ...]) -> None:
    print("Engine quality against exact hidden-deal full-information-continuation oracle")
    for evaluation in evaluations:
        mc_choice = evaluation.monte_carlo_choice.label() if evaluation.monte_carlo_choice else "n/a"
        mc_regret = f"{evaluation.monte_carlo_regret:.3f}" if evaluation.monte_carlo_regret is not None else "n/a"
        print(
            f"{evaluation.scenario.name}: exact={evaluation.exact_choice.label()}, "
            f"heuristic={evaluation.heuristic_choice.label()} regret={evaluation.heuristic_regret:.3f}, "
            f"monte_carlo={mc_choice} regret={mc_regret}, hidden_deals={evaluation.hidden_deals}"
        )
        print(f"  continuation: {evaluation.continuation_model}")
        print(f"  legal moves: {labels(move.card for move in evaluation.move_evaluations)}")
    print("\nWrote visual reports:")
    for path in paths:
        print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual engine-quality report against exact proof positions.")
    parser.add_argument("--samples-per-move", type=int, default=80)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--skip-monte-carlo", action="store_true")
    args = parser.parse_args()

    evaluations = tuple(
        evaluate_scenario(
            scenario,
            samples_per_move=args.samples_per_move,
            max_turns=args.max_turns,
            seed=args.seed,
            include_monte_carlo=not args.skip_monte_carlo,
        )
        for scenario in evaluation_scenarios()
    )
    paths = render_reports(evaluations, args.output_dir)
    print_console_summary(evaluations, paths)


if __name__ == "__main__":
    main()
