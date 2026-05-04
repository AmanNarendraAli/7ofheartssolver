from seven_hearts import (
    Card,
    FullInformationState,
    GameState,
    PlayerKnowledge,
    Suit,
    SuitRun,
    format_exact_imperfect_information_certificate,
    format_exact_solver_certificate,
    recommend_move_exact_imperfect_information,
    solve_full_information,
    solve_full_information_against_policy,
    lowest_legal_card_policy,
)


def cards(text: str) -> frozenset[Card]:
    return frozenset(Card.parse(part) for part in text.split())


def main() -> None:
    run_full_information_rational_certificate()
    run_full_information_fixed_policy_certificate()
    run_imperfect_information_certificate()


def run_full_information_rational_certificate() -> None:
    state = FullInformationState.from_hands(
        {
            0: cards("6H 7S"),
            1: cards("5H"),
            2: cards("AC"),
            3: cards("AD"),
        },
        GameState(
            table={suit: SuitRun() for suit in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )
    result = solve_full_information(state)

    print("\n=== Rational Full-Information Proof ===")
    print(format_exact_solver_certificate(state, result))


def run_full_information_fixed_policy_certificate() -> None:
    state = FullInformationState.from_hands(
        {
            0: cards("6H 8H"),
            1: cards("5H"),
            2: cards("9H"),
            3: cards("7S"),
        },
        GameState(
            table={suit: SuitRun() for suit in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )
    result = solve_full_information_against_policy(
        state,
        lowest_legal_card_policy,
        policy_name="lowest_legal_card_fixed_policy",
    )

    print("\n=== Fixed-Policy Full-Information Proof ===")
    print(format_exact_solver_certificate(state, result))


def run_imperfect_information_certificate() -> None:
    state = GameState(
        table={suit: SuitRun() for suit in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 48, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(player=0, hand=cards("6H"))

    result = recommend_move_exact_imperfect_information(state, knowledge)
    assert result is not None

    print("\n=== Exact Hidden-Deal EV Proof ===")
    print(format_exact_imperfect_information_certificate(state, knowledge, result))


if __name__ == "__main__":
    main()
