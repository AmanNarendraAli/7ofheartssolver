import argparse

from seven_hearts import (
    Card,
    FullInformationState,
    GameState,
    Suit,
    SuitRun,
    benchmark_full_information_search,
)


FAST_BENCHMARKS = {
    "single_suit_2_cards_each",
    "multi_suit_3_cards_each",
    "multi_suit_4_cards_each",
}
HARD_BENCHMARKS = {
    "four_suit_5_cards_each",
    "chain_pressure_8_cards_each",
}


def cards(text: str) -> frozenset[Card]:
    return frozenset(Card.parse(part) for part in text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact full-information proof benchmarks.")
    parser.add_argument(
        "--include-hard",
        action="store_true",
        help="include thousand-state benchmark positions that measure deeper exact-search scaling",
    )
    args = parser.parse_args()

    for name, state in benchmark_positions(include_hard=args.include_hard):
        result = benchmark_full_information_search(name, state)
        chosen = result.chosen_move.label() if result.chosen_move else "pass"
        print(
            f"{result.name}: chosen={chosen}, value={result.value}, "
            f"states={result.states_evaluated}, cache_hits={result.cache_hits}, "
            f"terminals={result.terminal_states}, deadlocks={result.deadlock_states}, "
            f"elapsed={result.elapsed_seconds:.6f}s, states_per_second={result.states_per_second:.1f}"
        )


def benchmark_positions(include_hard: bool = False) -> tuple[tuple[str, FullInformationState], ...]:
    positions = [
        (
            "single_suit_2_cards_each",
            FullInformationState.from_hands(
                {
                    0: cards("6H 8H"),
                    1: cards("5H 9H"),
                    2: cards("4H 10H"),
                    3: cards("3H JH"),
                },
                GameState(
                    table={suit: SuitRun() for suit in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
                    current_player=0,
                ),
            ),
        ),
        (
            "multi_suit_3_cards_each",
            FullInformationState.from_hands(
                {
                    0: cards("6H 8H 6C"),
                    1: cards("5H 9H 8C"),
                    2: cards("7D AC AD"),
                    3: cards("7S AS KD"),
                },
                GameState(
                    table={
                        suit: SuitRun()
                        for suit in Suit
                    } | {
                        Suit.HEARTS: SuitRun(low=7, high=7),
                        Suit.CLUBS: SuitRun(low=7, high=7),
                    },
                    current_player=0,
                ),
            ),
        ),
        (
            "multi_suit_4_cards_each",
            FullInformationState.from_hands(
                {
                    0: cards("5H 6H 8H 6C"),
                    1: cards("4H 9H 8C AC"),
                    2: cards("3H 10H 7D AD"),
                    3: cards("2H JH 7S AS"),
                },
                GameState(
                    table={
                        suit: SuitRun()
                        for suit in Suit
                    } | {
                        Suit.HEARTS: SuitRun(low=7, high=7),
                        Suit.CLUBS: SuitRun(low=7, high=7),
                    },
                    current_player=0,
                ),
            ),
        ),
    ]

    if include_hard:
        positions.extend(hard_benchmark_positions())

    return tuple(positions)


def hard_benchmark_positions() -> tuple[tuple[str, FullInformationState], ...]:
    all_suits_open = {
        Suit.CLUBS: SuitRun(low=7, high=7),
        Suit.DIAMONDS: SuitRun(low=7, high=7),
        Suit.HEARTS: SuitRun(low=7, high=7),
        Suit.SPADES: SuitRun(low=7, high=7),
    }

    return (
        (
            "four_suit_5_cards_each",
            FullInformationState.from_hands(
                {
                    0: cards("6H 8H 6C 8C 6D"),
                    1: cards("5H 9H 5C 9C 8D"),
                    2: cards("4H 10H 4C 10C 7S"),
                    3: cards("3H JH 3C 8S 9D"),
                },
                GameState(
                    table={suit: SuitRun() for suit in Suit} | {
                        Suit.HEARTS: SuitRun(low=7, high=7),
                        Suit.CLUBS: SuitRun(low=7, high=7),
                        Suit.DIAMONDS: SuitRun(low=7, high=7),
                    },
                    current_player=0,
                ),
            ),
        ),
        (
            "chain_pressure_8_cards_each",
            FullInformationState.from_hands(
                {
                    0: cards("4H 5H 6H 8H 9H 6C 8C 6D"),
                    1: cards("3H 10H 5C 9C 5D 8D 6S 8S"),
                    2: cards("2H JH 4C 10C 4D 9D 5S 9S"),
                    3: cards("AH QH 3C JC 3D 10D 4S 10S"),
                },
                GameState(
                    table={suit: SuitRun() for suit in Suit} | all_suits_open,
                    current_player=0,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    main()
