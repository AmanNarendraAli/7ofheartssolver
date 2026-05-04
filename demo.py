import random
from dataclasses import dataclass

from seven_hearts import (
    Card,
    GameState,
    PlayerKnowledge,
    StrategyCandidate,
    StrategyWeights,
    Suit,
    SuitRun,
    evaluate_move_monte_carlo,
    estimate_complete_game_metrics,
    estimate_strategy_self_play,
    labels,
    recommend_move,
    score_move,
)


SAMPLES_PER_MOVE = 80
COMPLETE_GAME_SAMPLES = 400
SELF_PLAY_TUNING_GAMES = 400
MAX_TURNS = 200
COMPLETE_GAME_MAX_TURNS = 300
SEED = 7


@dataclass(frozen=True)
class Scenario:
    name: str
    state: GameState
    knowledge: PlayerKnowledge


def cards(text: str) -> frozenset[Card]:
    return frozenset(Card.parse(part) for part in text.split())


def main() -> None:
    for scenario in scenarios():
        run_scenario(scenario)
    run_complete_game_estimate()
    run_strategy_tuning_estimate()


def run_scenario(scenario: Scenario) -> None:
    state = scenario.state
    knowledge = scenario.knowledge
    legal = state.legal_moves(knowledge.hand)

    print(f"\n=== {scenario.name} ===")
    print(f"Legal moves: {labels(legal)}")

    for card in legal:
        result = score_move(state, knowledge, card)
        print(f"\n{result.card}: {result.score:.1f}")
        for reason in result.reasons:
            print(f"  - {reason}")

    best = recommend_move(state, knowledge)
    if best is None:
        print("\nHeuristic recommendation: pass")
    else:
        print(f"\nHeuristic recommendation: play {best.card} ({best.score:.1f})")

    rng = random.Random(SEED)
    print("\nOracle Monte Carlo:")
    mc_results = []
    for card in legal:
        result = evaluate_move_monte_carlo(
            state,
            knowledge,
            card,
            samples=SAMPLES_PER_MOVE,
            max_turns=MAX_TURNS,
            rng=rng,
        )
        mc_results.append(result)
        print(
            f"  {result.card}: win {result.win_rate:.1%} +/- {result.win_rate_standard_error:.1%}, "
            f"margin {result.average_finish_margin:.1f}, turns {result.average_turns:.1f}, "
            f"timeouts {result.timeout_rate:.1%}, score {result.score:.1f}, samples {result.samples}"
        )

    if not mc_results:
        print("Oracle recommendation: pass")
    else:
        mc_best = max(mc_results, key=lambda result: result.score)
        print(f"Oracle recommendation: play {mc_best.card} ({mc_best.win_rate:.1%} win rate)")


def run_complete_game_estimate() -> None:
    estimate = estimate_complete_game_metrics(
        COMPLETE_GAME_SAMPLES,
        random.Random(SEED),
        max_turns=COMPLETE_GAME_MAX_TURNS,
    )

    print("\n=== Complete Random-Deal Oracle Self-Play ===")
    print(
        f"games {estimate.games}, turns {estimate.average_turns:.1f}, "
        f"timeouts {estimate.timeout_rate:.1%}"
    )
    for player in range(4):
        print(
            f"  P{player}: win {estimate.win_rates[player]:.1%} "
            f"+/- {estimate.win_rate_standard_errors[player]:.1%}, "
            f"margin {estimate.average_finish_margins[player]:.1f}"
        )


def run_strategy_tuning_estimate() -> None:
    candidates = (
        StrategyCandidate("tuned baseline"),
        StrategyCandidate(
            "legacy baseline",
            StrategyWeights(tail_base_credit=2.0, tail_distance_penalty=0.25),
        ),
        StrategyCandidate(
            "previous tuned",
            StrategyWeights(tail_base_credit=2.4, tail_distance_penalty=0.18),
        ),
        StrategyCandidate(
            "cautious unlocks",
            StrategyWeights(opponent_unlock_risk=2.5, opponent_runway_penalty=0.12),
        ),
        StrategyCandidate(
            "gate tempo",
            StrategyWeights(gate_card_credit=1.4, self_unlock=3.3, response_penalty=0.45),
        ),
    )
    estimates = estimate_strategy_self_play(
        candidates,
        games=SELF_PLAY_TUNING_GAMES,
        rng=random.Random(SEED),
        max_turns=COMPLETE_GAME_MAX_TURNS,
    )

    print("\n=== Shared-Deal Strategy Tuning Probe ===")
    print(f"candidate controls P0 against baseline opponents over {SELF_PLAY_TUNING_GAMES} deals")
    for estimate in estimates:
        print(
            f"  {estimate.candidate.name}: win {estimate.win_rate:.1%} "
            f"+/- {estimate.win_rate_standard_error:.1%}, "
            f"margin {estimate.average_finish_margin:.1f}, turns {estimate.average_turns:.1f}, "
            f"timeouts {estimate.timeout_rate:.1%}, score {estimate.score:.1f}"
        )


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            name="Demo Baseline",
            state=GameState(
                table={
                    Suit.CLUBS: SuitRun(),
                    Suit.DIAMONDS: SuitRun(),
                    Suit.HEARTS: SuitRun(low=6, high=8),
                    Suit.SPADES: SuitRun(),
                },
                hand_counts=(10, 13, 13, 13),
                current_player=0,
            ),
            knowledge=PlayerKnowledge(
                player=0,
                hand=cards("7S AS 5H 9H 7D KC 3C 10D 2S 8S"),
            ),
        ),
        Scenario(
            name="Bare Seven Versus Heart Progress",
            state=GameState(
                table={
                    Suit.CLUBS: SuitRun(),
                    Suit.DIAMONDS: SuitRun(),
                    Suit.HEARTS: SuitRun(low=6, high=8),
                    Suit.SPADES: SuitRun(),
                },
                current_player=0,
            ),
            knowledge=PlayerKnowledge(
                player=0,
                hand=cards("7S AS 5H 9H KC 3C 10D 2S"),
            ),
        ),
        Scenario(
            name="Controlled Seven With Gates",
            state=GameState(
                table={
                    Suit.CLUBS: SuitRun(),
                    Suit.DIAMONDS: SuitRun(),
                    Suit.HEARTS: SuitRun(low=6, high=8),
                    Suit.SPADES: SuitRun(),
                },
                current_player=0,
            ),
            knowledge=PlayerKnowledge(
                player=0,
                hand=cards("7S 6S 8S AS 5H 9H KC 3C"),
            ),
        ),
        Scenario(
            name="Tail Runway Pressure",
            state=GameState(
                table={
                    Suit.CLUBS: SuitRun(),
                    Suit.DIAMONDS: SuitRun(),
                    Suit.HEARTS: SuitRun(low=7, high=7),
                    Suit.SPADES: SuitRun(),
                },
                current_player=0,
            ),
            knowledge=PlayerKnowledge(
                player=0,
                hand=cards("7S AS KS 6H 8H 3C JD"),
            ),
        ),
    )


if __name__ == "__main__":
    main()
