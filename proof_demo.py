from seven_hearts import (
    FullInformationState,
    GameState,
    PlayerKnowledge,
    Suit,
    SuitRun,
    format_exact_imperfect_information_certificate,
    format_exact_solver_certificate,
    lowest_legal_card_policy,
    parse_cards,
    recommend_move_exact_imperfect_information,
    solve_full_information,
    solve_full_information_against_policy,
)


def main() -> None:
    run_full_information_rational_certificate()
    run_full_information_fixed_policy_certificate()
    run_imperfect_information_certificate()
    run_reduced_belief_imperfect_information_certificate()
    run_multi_suit_imperfect_information_certificate()


def run_full_information_rational_certificate() -> None:
    state = FullInformationState.from_hands(
        {
            0: parse_cards("6H 7S"),
            1: parse_cards("5H"),
            2: parse_cards("AC"),
            3: parse_cards("AD"),
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
            0: parse_cards("6H 8H"),
            1: parse_cards("5H"),
            2: parse_cards("9H"),
            3: parse_cards("7S"),
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
    knowledge = PlayerKnowledge(player=0, hand=parse_cards("6H"))

    result = recommend_move_exact_imperfect_information(state, knowledge)
    assert result is not None

    print("\n=== Exact Hidden-Deal EV Proof With Full-Information Continuation ===")
    print(format_exact_imperfect_information_certificate(state, knowledge, result))


def run_reduced_belief_imperfect_information_certificate() -> None:
    deck = parse_cards("5H 6H 7H 8H 9H 7C AC AD")
    state = GameState(
        table={suit: SuitRun() for suit in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(3, 2, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        player=0,
        hand=parse_cards("5H 6H 8H"),
        deck=deck,
    )

    result = recommend_move_exact_imperfect_information(state, knowledge)
    assert result is not None

    print("\n=== Reduced-Belief Exact EV Proof With Full-Information Continuation ===")
    print(format_exact_imperfect_information_certificate(state, knowledge, result))


def run_multi_suit_imperfect_information_certificate() -> None:
    table = {
        suit: SuitRun()
        for suit in Suit
    } | {
        Suit.HEARTS: SuitRun(low=6, high=8),
        Suit.CLUBS: SuitRun(low=7, high=7),
    }
    deck = parse_cards("4H 5H 6H 7H 8H 9H 10H 6C 7C 8C 7D 7S AC AD")
    state = GameState(
        table=table,
        hand_counts=(4, 2, 2, 2),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        player=0,
        hand=parse_cards("5H 9H 6C 8C"),
        deck=deck,
    )

    result = recommend_move_exact_imperfect_information(state, knowledge)
    assert result is not None

    print("\n=== Multi-Suit Exact EV Proof With Full-Information Continuation ===")
    print(format_exact_imperfect_information_certificate(state, knowledge, result))


if __name__ == "__main__":
    main()
