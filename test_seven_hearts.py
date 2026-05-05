import math
import random

from seven_hearts import (
    Card,
    FullInformationState,
    GameState,
    MoveEvent,
    PlayerKnowledge,
    StrategyCandidate,
    StrategyWeights,
    Suit,
    SuitRun,
    apply_known_play,
    build_opponent_model,
    deal_random_hands,
    endgame_urgency,
    enumerate_hidden_deals,
    estimate_complete_game_metrics,
    estimate_strategy_self_play,
    evaluate_move_monte_carlo,
    evaluate_move_exact_imperfect_information,
    format_exact_imperfect_information_certificate,
    format_exact_solver_certificate,
    future_chain_impact,
    full_deck,
    hand_count_tuple,
    highest_legal_card_policy,
    initial_state_for_hands,
    lowest_legal_card_policy,
    recommend_move,
    recommend_move_exact_imperfect_information,
    recommend_move_monte_carlo,
    rollout_oracle,
    sample_hidden_deal,
    sample_hidden_deals,
    score_oracle_move,
    score_move,
    choose_rational_move,
    choose_expected_value_vector_move,
    solve_full_information,
    solve_full_information_against_policy,
    simulate_complete_game,
    time_to_playable,
)
from proof_benchmark import FAST_BENCHMARKS, HARD_BENCHMARKS, benchmark_positions, hard_benchmark_positions
from proof_eval import evaluate_scenario, evaluation_scenarios


def hand(text: str) -> frozenset[Card]:
    return frozenset(Card.parse(part) for part in text.split())


def test_closed_suit_only_allows_seven() -> None:
    state = GameState()

    assert state.public_legal_cards() == {Card(Suit.HEARTS, 7)}


def test_after_opening_move_closed_suits_allow_their_sevens() -> None:
    state = GameState().after_play(0, Card(Suit.HEARTS, 7))

    assert state.public_legal_cards() == {
        Card(Suit.CLUBS, 7),
        Card(Suit.DIAMONDS, 7),
        Card(Suit.HEARTS, 6),
        Card(Suit.HEARTS, 8),
        Card(Suit.SPADES, 7),
    }


def test_open_suit_allows_adjacent_cards() -> None:
    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=5, high=8)})

    legal_hearts = {card for card in state.public_legal_cards() if card.suit == Suit.HEARTS}

    assert legal_hearts == {Card(Suit.HEARTS, 4), Card(Suit.HEARTS, 9)}


def test_player_legal_moves_are_intersection_of_hand_and_table_legal_cards() -> None:
    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=6, high=7)})

    assert set(state.legal_moves(hand("5H 8H 7S AS"))) == {
        Card(Suit.HEARTS, 5),
        Card(Suit.HEARTS, 8),
        Card(Suit.SPADES, 7),
    }


def test_pass_removes_then_legal_cards_from_that_opponent_possibilities() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)), MoveEvent(1, None)),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 5H 9H 10H JH QH KH 7C 7D 7S"))

    model = build_opponent_model(state, knowledge)

    assert Card(Suit.HEARTS, 6) not in model.possible_cards[1]
    assert Card(Suit.HEARTS, 8) not in model.possible_cards[1]
    assert Card(Suit.HEARTS, 6) in model.possible_cards[2]


def test_opponent_play_removes_card_from_other_opponents_possibilities() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=6, high=7)},
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)), MoveEvent(1, Card(Suit.HEARTS, 6))),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 5H 9H 10H JH QH KH 7C 7D 7S"))

    model = build_opponent_model(state, knowledge)

    assert Card(Suit.HEARTS, 6) not in model.possible_cards[2]
    assert Card(Suit.HEARTS, 6) not in model.possible_cards[3]


def test_multi_pass_removes_cards_legal_at_each_pass_moment() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=6, high=7)},
        history=(
            MoveEvent(0, Card(Suit.HEARTS, 7)),
            MoveEvent(1, None),
            MoveEvent(2, Card(Suit.HEARTS, 6)),
            MoveEvent(1, None),
        ),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 9H 10H JH QH KH 7C 7D 7S 8S"))

    model = build_opponent_model(state, knowledge)

    assert Card(Suit.HEARTS, 6) not in model.possible_cards[1]
    assert Card(Suit.HEARTS, 8) not in model.possible_cards[1]
    assert Card(Suit.HEARTS, 5) not in model.possible_cards[1]


def test_recommend_move_returns_none_when_forced_to_pass() -> None:
    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)})
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 9H 10H"))

    assert recommend_move(state, knowledge) is None


def test_recommend_move_works_without_hand_counts() -> None:
    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)})
    knowledge = PlayerKnowledge(player=0, hand=hand("6H 8H AS KS"))

    result = recommend_move(state, knowledge)

    assert result is not None
    assert result.card in {Card(Suit.HEARTS, 6), Card(Suit.HEARTS, 8)}


def test_known_player_count_is_decremented_but_unknown_counts_remain_unknown() -> None:
    state = GameState(hand_counts=(4, None, 7, None))

    next_state = state.after_play(0, Card(Suit.HEARTS, 7))

    assert next_state.hand_counts == (3, None, 7, None)


def test_validate_turn_rejects_passing_with_legal_moves() -> None:
    state = GameState().after_play(0, Card(Suit.HEARTS, 7))

    try:
        state.after_play(1, None, hand=hand("6H AS"))
    except ValueError as error:
        assert "cannot pass" in str(error)
    else:
        raise AssertionError("expected illegal pass to raise ValueError")


def test_validate_turn_rejects_non_opening_card_on_empty_table() -> None:
    state = GameState()

    try:
        state.after_play(0, Card(Suit.SPADES, 7), hand=hand("7S"))
    except ValueError as error:
        assert "cannot play" in str(error)
    else:
        raise AssertionError("expected illegal first move to raise ValueError")


def test_endgame_urgency_has_gradient_for_known_counts() -> None:
    calm = GameState(hand_counts=(8, 7, 9, 10))
    close_race = GameState(hand_counts=(4, 1, 9, 10))

    assert endgame_urgency(calm, 0) == 0.0
    assert endgame_urgency(close_race, 0) > endgame_urgency(calm, 0)


def test_endgame_urgency_is_not_an_active_score_component() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(3, 2, 8, 9),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("6H 8H AS KS"))

    result = score_move(state, knowledge, Card(Suit.HEARTS, 6))

    assert "endgame_urgency" not in result.components


def test_earlier_side_card_releases_more_future_chain_than_later_side_card() -> None:
    state = GameState(
        table={
            Suit.CLUBS: SuitRun(low=7, high=7),
            Suit.DIAMONDS: SuitRun(),
            Suit.HEARTS: SuitRun(low=7, high=9),
            Suit.SPADES: SuitRun(),
        }
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("8C 10H AS KD"))

    assert future_chain_impact(state, knowledge, Card(Suit.CLUBS, 8)) < future_chain_impact(
        state, knowledge, Card(Suit.HEARTS, 10)
    )


def test_opening_seven_is_better_when_it_controls_both_immediate_gates() -> None:
    state = GameState().after_play(0, Card(Suit.HEARTS, 7))
    bare_seven = PlayerKnowledge(player=0, hand=hand("7S AH 2H 3H"))
    controlled_seven = PlayerKnowledge(player=0, hand=hand("7S 6S 8S AH"))

    bare_score = score_move(state, bare_seven, Card(Suit.SPADES, 7)).score
    controlled_score = score_move(state, controlled_seven, Card(Suit.SPADES, 7)).score

    assert controlled_score > bare_score


def test_opening_seven_gets_some_credit_for_distant_tail_card() -> None:
    state = GameState().after_play(0, Card(Suit.HEARTS, 7))
    bare_seven = PlayerKnowledge(player=0, hand=hand("7S AH 2H 3H"))
    ace_pressure = PlayerKnowledge(player=0, hand=hand("7S AS AH 2H"))

    bare_impact = future_chain_impact(state, bare_seven, Card(Suit.SPADES, 7))
    ace_impact = future_chain_impact(state, ace_pressure, Card(Suit.SPADES, 7))

    assert ace_impact > bare_impact


def test_tail_bonus_uses_tail_distance_not_average_side_distance() -> None:
    state = GameState().after_play(0, Card(Suit.HEARTS, 7))
    with_near_card = PlayerKnowledge(player=0, hand=hand("7S 8S KS AH"))
    without_near_card = PlayerKnowledge(player=0, hand=hand("7S KS AH 2H"))

    with_near_impact = future_chain_impact(state, with_near_card, Card(Suit.SPADES, 7))
    without_near_impact = future_chain_impact(state, without_near_card, Card(Suit.SPADES, 7))

    assert with_near_impact - without_near_impact < 2.3


def test_future_chain_impact_handles_ace_and_king_edges() -> None:
    state = GameState(
        table={
            Suit.CLUBS: SuitRun(low=2, high=7),
            Suit.DIAMONDS: SuitRun(low=7, high=12),
            Suit.HEARTS: SuitRun(low=7, high=7),
            Suit.SPADES: SuitRun(low=7, high=7),
        }
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AC KD 6H 8S"))

    assert future_chain_impact(state, knowledge, Card(Suit.CLUBS, 1)) == 0.0
    assert future_chain_impact(state, knowledge, Card(Suit.DIAMONDS, 13)) == 0.0


def test_time_to_playable_counts_required_chain_steps() -> None:
    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.SPADES: SuitRun(low=7, high=7)})

    assert time_to_playable(state, Card(Suit.SPADES, 1)) == 6
    assert time_to_playable(state, Card(Suit.SPADES, 8)) == 1


def test_holder_probabilities_sum_to_one_for_possible_unseen_cards() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)), MoveEvent(1, None)),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 5H 9H 10H JH QH KH 7C 7D 7S"))
    model = build_opponent_model(state, knowledge)

    for card in knowledge.unseen_cards(state):
        possible_holders = [player for player, cards in model.possible_cards.items() if card in cards]
        if possible_holders:
            total = sum(model.holder_probability(player, card) for player in model.possible_cards)
            assert abs(total - 1.0) < 0.000001


def test_known_zero_count_removes_player_from_exact_holder_marginals() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(13, 0, None, None),
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)),),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 5H 9H 10H JH QH KH 7C 7D 7S"))

    model = build_opponent_model(state, knowledge)

    assert model.inference_mode == "exact_count_dp"
    for card in knowledge.unseen_cards(state):
        assert model.holder_probability(1, card) == 0.0


def test_exact_holder_marginals_obey_pass_constraints_and_known_counts() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(13, 0, 0, None),
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)), MoveEvent(1, None)),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 5H 9H 10H JH QH KH 7C 7D 7S"))

    model = build_opponent_model(state, knowledge)

    assert model.holder_probability(1, Card(Suit.HEARTS, 6)) == 0.0
    assert model.holder_probability(2, Card(Suit.HEARTS, 6)) == 0.0
    assert model.holder_probability(3, Card(Suit.HEARTS, 6)) == 1.0


def test_sample_hidden_deal_assigns_every_unseen_card_once() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(13, 1, 0, None),
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)), MoveEvent(1, None)),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 5H 9H 10H JH QH KH 7C 7D 7S"))

    deal = sample_hidden_deal(state, knowledge, random.Random(1))

    assert deal is not None
    dealt_cards = set().union(*deal.hands.values())
    assert dealt_cards == knowledge.unseen_cards(state)
    assert sum(len(cards) for cards in deal.hands.values()) == len(dealt_cards)


def test_sample_hidden_deal_respects_known_counts_and_pass_constraints() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(13, 1, 0, None),
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)), MoveEvent(1, None)),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 5H 9H 10H JH QH KH 7C 7D 7S"))

    deal = sample_hidden_deal(state, knowledge, random.Random(2))

    assert deal is not None
    assert len(deal.hand(1)) == 1
    assert len(deal.hand(2)) == 0
    assert Card(Suit.HEARTS, 6) not in deal.hand(1)
    assert Card(Suit.HEARTS, 8) not in deal.hand(1)


def test_sample_hidden_deal_returns_none_for_impossible_counts() -> None:
    state = GameState(hand_counts=(13, 52, None, None))
    knowledge = PlayerKnowledge(player=0, hand=hand("AH"))

    assert sample_hidden_deal(state, knowledge, random.Random(3)) is None


def test_sample_hidden_deals_returns_requested_number_when_possible() -> None:
    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)})
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 5H 9H 10H JH QH KH 7C 7D 7S"))

    deals = sample_hidden_deals(state, knowledge, 4, random.Random(4))

    assert len(deals) == 4


def test_oracle_rollout_detects_first_player_out() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        current_player=0,
    )
    hands = {
        0: {Card(Suit.HEARTS, 6)},
        1: {Card(Suit.CLUBS, 1)},
        2: {Card(Suit.DIAMONDS, 1)},
        3: {Card(Suit.SPADES, 1)},
    }

    result = rollout_oracle(state, hands, max_turns=10)

    assert result.winner == 0
    assert result.final_hand_counts[0] == 0


def test_oracle_rollout_passes_when_player_has_no_legal_move() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        current_player=1,
    )
    hands = {
        0: {Card(Suit.HEARTS, 6)},
        1: {Card(Suit.CLUBS, 1)},
        2: {Card(Suit.DIAMONDS, 1)},
        3: {Card(Suit.SPADES, 1)},
    }

    result = rollout_oracle(state, hands, max_turns=1)

    assert result.winner is None
    assert result.final_hand_counts == (1, 1, 1, 1)


def test_evaluate_move_monte_carlo_scores_legal_move() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, None, None, None),
        current_player=0,
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("6H 8H"))

    result = evaluate_move_monte_carlo(state, knowledge, Card(Suit.HEARTS, 6), samples=3, rng=random.Random(5))

    assert result.card == Card(Suit.HEARTS, 6)
    assert result.samples == 3
    assert 0.0 <= result.win_rate <= 1.0
    assert result.average_turns >= 0.0
    assert result.timeout_rate == 0.0
    assert result.win_rate_standard_error >= 0.0


def test_evaluate_move_monte_carlo_reports_timeouts() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        current_player=0,
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("6H 8H"))

    result = evaluate_move_monte_carlo(
        state,
        knowledge,
        Card(Suit.HEARTS, 6),
        samples=2,
        max_turns=0,
        rng=random.Random(6),
    )

    assert result.timeout_rate == 1.0
    assert result.average_turns == 0.0


def test_recommend_move_monte_carlo_returns_legal_move() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        current_player=0,
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("6H 8H"))

    result = recommend_move_monte_carlo(state, knowledge, samples_per_move=2, rng=random.Random(6))

    assert result is not None
    assert result.card in {Card(Suit.HEARTS, 6), Card(Suit.HEARTS, 8)}


def test_oracle_move_score_penalizes_strong_next_response() -> None:
    state = GameState(
        table={
            Suit.CLUBS: SuitRun(),
            Suit.DIAMONDS: SuitRun(),
            Suit.HEARTS: SuitRun(low=7, high=7),
            Suit.SPADES: SuitRun(),
        },
        current_player=0,
    )
    quiet_hands = {
        0: {Card(Suit.HEARTS, 6)},
        1: {Card(Suit.CLUBS, 1)},
        2: {Card(Suit.DIAMONDS, 1)},
        3: {Card(Suit.SPADES, 1)},
    }
    response_hands = {
        0: {Card(Suit.HEARTS, 6)},
        1: {Card(Suit.HEARTS, 8), Card(Suit.HEARTS, 9)},
        2: {Card(Suit.DIAMONDS, 1)},
        3: {Card(Suit.SPADES, 1)},
    }

    quiet_score = score_oracle_move(state, quiet_hands, 0, Card(Suit.HEARTS, 6))
    response_score = score_oracle_move(state, response_hands, 0, Card(Suit.HEARTS, 6))

    assert response_score < quiet_score


def test_deal_random_hands_deals_full_deck_once() -> None:
    hands = deal_random_hands(random.Random(7))

    all_cards = set().union(*hands.values())

    assert len(hands) == 4
    assert all(len(hand_cards) == 13 for hand_cards in hands.values())
    assert len(all_cards) == 52


def test_deal_random_hands_is_seed_reproducible() -> None:
    hands = deal_random_hands(random.Random(7))

    assert hands[0] == hand("AC 9C JC QC 2D 4D 5D 7D 6H KH AS 2S QS")


def test_initial_state_starts_with_seven_hearts_holder() -> None:
    hands = {
        0: set(hand("AC 2C 3C 4C 5C 6C 8C 9C 10C JC QC KC AD")),
        1: set(hand("7H 2D 3D 4D 5D 6D 7D 8D 9D 10D JD QD KD")),
        2: set(hand("AH 2H 3H 4H 5H 6H 8H 9H 10H JH QH KH AS")),
        3: set(hand("2S 3S 4S 5S 6S 7S 8S 9S 10S JS QS KS 7C")),
    }

    state = initial_state_for_hands(hands)

    assert state.current_player == 1
    assert state.public_legal_cards() == {Card(Suit.HEARTS, 7)}


def test_simulate_complete_game_returns_result() -> None:
    result = simulate_complete_game(random.Random(8), max_turns=300)

    assert result.winner in {0, 1, 2, 3}
    assert sum(result.final_hand_counts) < 52
    assert not result.timed_out


def test_estimate_complete_game_metrics_reports_rates() -> None:
    estimate = estimate_complete_game_metrics(4, random.Random(9), max_turns=300)

    assert estimate.games == 4
    assert abs(sum(estimate.win_rates) - 1.0) < 0.000001
    assert all(rate >= 0.0 for rate in estimate.win_rates)
    assert all(error >= 0.0 for error in estimate.win_rate_standard_errors)
    assert estimate.average_turns > 0.0


def test_opponent_model_reports_impossible_known_counts() -> None:
    model = build_opponent_model(
        GameState(hand_counts=(13, 52, None, None)),
        PlayerKnowledge(player=0, hand=hand("AH")),
    )

    assert model.consistency_errors()


def test_score_components_are_immutable() -> None:
    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)})
    knowledge = PlayerKnowledge(player=0, hand=hand("6H 8H AS KS"))

    result = score_move(state, knowledge, Card(Suit.HEARTS, 6))

    try:
        result.components["extra"] = 1.0
    except TypeError:
        pass
    else:
        raise AssertionError("expected score components to be immutable")


def test_strategy_weights_change_heuristic_score_components() -> None:
    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)})
    knowledge = PlayerKnowledge(player=0, hand=hand("6H 5H 8H"))
    card = Card(Suit.HEARTS, 6)

    baseline = score_move(state, knowledge, card)
    tuned = score_move(state, knowledge, card, weights=StrategyWeights(self_unlock=5.0))

    assert tuned.components["self_unlock"] > baseline.components["self_unlock"]
    assert tuned.score > baseline.score


def test_strategy_self_play_compares_candidates_on_shared_deals() -> None:
    candidates = (
        StrategyCandidate("baseline"),
        StrategyCandidate("cautious", StrategyWeights(opponent_unlock_risk=3.0)),
    )

    estimates = estimate_strategy_self_play(candidates, games=3, rng=random.Random(10), max_turns=300)

    assert {estimate.candidate.name for estimate in estimates} == {"baseline", "cautious"}
    assert all(estimate.games == 3 for estimate in estimates)
    assert all(0.0 <= estimate.win_rate <= 1.0 for estimate in estimates)
    assert all(estimate.win_rate_standard_error >= 0.0 for estimate in estimates)


def test_rollout_oracle_can_use_seat_specific_weights() -> None:
    hands = deal_random_hands(random.Random(11))
    state = initial_state_for_hands(hands)
    result = rollout_oracle(
        state,
        hands,
        max_turns=300,
        weights={0: StrategyWeights(self_unlock=4.0)},
    )

    assert result.winner in {0, 1, 2, 3}
    assert not result.timed_out


def test_full_information_state_enforces_card_conservation_for_declared_deck() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("6H"),
            1: hand("8H"),
            2: hand("7S"),
            3: hand("7D"),
        },
        GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)}),
    )

    state.assert_card_conservation(hand("6H 7H 8H 7S 7D"))


def test_full_information_exact_solver_picks_immediate_win() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("6H"),
            1: hand("8H"),
            2: hand("7S"),
            3: hand("7D"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    result = solve_full_information(state)

    assert result.chosen_move == Card(Suit.HEARTS, 6)
    assert result.value == (1.0, 0.0, 0.0, 0.0)
    assert result.deadlock_states == 0


def test_full_information_exact_solver_follows_forced_pass_chain() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("AC"),
            1: hand("AD"),
            2: hand("6H"),
            3: hand("AS"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    result = solve_full_information(state)

    assert result.chosen_move is None
    assert result.value == (0.0, 0.0, 1.0, 0.0)
    assert result.deadlock_states == 0


def test_full_information_solver_reports_deadlock_state() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("AC"),
            1: hand("AD"),
            2: hand("AS"),
            3: hand("2C"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
        consecutive_passes=4,
    )

    result = solve_full_information(state)

    assert result.value == (0.0, 0.0, 0.0, 0.0)
    assert result.deadlock_states == 1


def test_full_information_root_forced_pass_has_no_chosen_move() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("AC"),
            1: hand("AD"),
            2: hand("6H"),
            3: hand("AS"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    result = solve_full_information(state)

    assert state.legal_moves() == ()
    assert result.chosen_move is None
    assert result.best_moves == ()
    assert result.value == solve_full_information(state.after_action(None)).value


def test_forced_pass_chain_canonicalization_preserves_value() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("AC"),
            1: hand("AD"),
            2: hand("AS"),
            3: hand("6H"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    first_pass = state.after_action(None)
    second_pass = first_pass.after_action(None)
    third_pass = second_pass.after_action(None)

    assert solve_full_information(state).value == solve_full_information(first_pass).value
    assert solve_full_information(state).value == solve_full_information(second_pass).value
    assert solve_full_information(state).value == solve_full_information(third_pass).value


def test_full_information_exact_solver_avoids_opening_next_player_win() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("6H 7S"),
            1: hand("5H"),
            2: hand("AC"),
            3: hand("AD"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    result = solve_full_information(state)

    assert set(result.move_values) == {Card(Suit.HEARTS, 6), Card(Suit.SPADES, 7)}
    assert result.move_values[Card(Suit.HEARTS, 6)] == (0.0, 1.0, 0.0, 0.0)
    assert result.chosen_move == Card(Suit.SPADES, 7)
    assert result.value[0] == 1.0


def test_full_information_exact_solver_uses_deterministic_tie_breaker() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("6H 8H"),
            1: hand("AC"),
            2: hand("AD"),
            3: hand("AS"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    result = solve_full_information(state)

    assert result.best_moves == (Card(Suit.HEARTS, 6), Card(Suit.HEARTS, 8))
    assert result.chosen_move == Card(Suit.HEARTS, 6)


def brute_force_full_information_value(state: FullInformationState) -> tuple[float, float, float, float]:
    if state.winner is not None:
        values = [0.0, 0.0, 0.0, 0.0]
        values[state.winner] = 1.0
        return tuple(values)  # type: ignore[return-value]

    legal = state.legal_moves()
    if not legal:
        if state.consecutive_passes >= 4:
            return (0.0, 0.0, 0.0, 0.0)
        return brute_force_full_information_value(state.after_action(None))

    move_values = {card: brute_force_full_information_value(state.after_action(card)) for card in legal}
    chosen = choose_rational_move(state.current_player, move_values)
    return move_values[chosen]


def random_reduced_full_information_state(rng: random.Random) -> tuple[FullInformationState, frozenset[Card]]:
    deck = tuple(hand("5H 6H 7H 8H 9H 7C 7D 7S"))
    shuffled = list(deck)
    rng.shuffle(shuffled)
    hands = {player: set(shuffled[player * 2 : (player + 1) * 2]) for player in range(4)}
    current_player = next(player for player, cards in hands.items() if Card(Suit.HEARTS, 7) in cards)
    state = FullInformationState.from_hands(
        hands,
        GameState(current_player=current_player),
    )

    for _ in range(rng.randrange(5)):
        if state.winner is not None:
            break
        legal = state.legal_moves()
        state = state.after_action(rng.choice(legal) if legal else None)

    return state, frozenset(deck)


def test_full_information_memoized_solver_matches_tiny_brute_force() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("6H 8H"),
            1: hand("5H"),
            2: hand("9H"),
            3: hand("7S"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    result = solve_full_information(state)

    assert result.value == brute_force_full_information_value(state)
    assert result.states_evaluated > 0
    assert result.terminal_states > 0


def test_full_information_memoized_solver_matches_random_tiny_brute_force() -> None:
    rng = random.Random(12)

    for _ in range(40):
        state, deck = random_reduced_full_information_state(rng)
        state.assert_card_conservation(deck)

        result = solve_full_information(state)

        assert result.value == brute_force_full_information_value(state)
        assert result.chosen_move in state.legal_moves() or result.chosen_move is None
        assert result.deadlock_states == 0


def test_random_full_game_walk_preserves_conservation_and_table_invariants() -> None:
    rng = random.Random(13)

    for _ in range(8):
        hands = deal_random_hands(rng)
        state = initial_state_for_hands(hands)
        seen_hand_counts = hand_count_tuple(hands)

        for _ in range(80):
            full_state = FullInformationState.from_hands(hands, state)
            full_state.assert_card_conservation()
            assert state.hand_counts == seen_hand_counts
            assert set().union(*hands.values()).isdisjoint(state.played_cards())

            for suit, run in state.table.items():
                if run.is_open:
                    assert run.low is not None and run.high is not None
                    assert run.low <= 7 <= run.high
                    assert all(Card(suit, rank) in state.played_cards() for rank in range(run.low, run.high + 1))

            if full_state.winner is not None:
                break

            player = state.current_player
            legal = state.legal_moves(hands[player])
            card = rng.choice(legal) if legal else None
            previous_count = len(hands[player])
            state, hands = apply_known_play(state, hands, player, card)
            seen_hand_counts = hand_count_tuple(hands)

            if card is None:
                assert len(hands[player]) == previous_count
            else:
                assert len(hands[player]) == previous_count - 1
                assert card in state.played_cards()


def test_exact_solver_certificate_snapshot_for_rational_proof_position() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("6H 7S"),
            1: hand("5H"),
            2: hand("AC"),
            3: hand("AD"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    certificate = format_exact_solver_certificate(state, solve_full_information(state))

    assert certificate == "\n".join(
        [
            "Exact full-information certificate",
            "policy: full_information_rational_first_out",
            "current_player: P0",
            "hand_counts: (2, 1, 1, 1)",
            "legal_moves: [6H, 7S]",
            "value: (P0=1.000, P1=0.000, P2=0.000, P3=0.000)",
            "best_moves: [7S]",
            "chosen_move: 7S",
            "exhaustive: True",
            "search: states=6, cache_hits=2, terminals=2, deadlocks=0",
            "tie_break: maximize acting player's first-out value; then minimize next player's value; then minimize strongest opponent value; then deterministic card order",
            "move_values:",
            "    6H: (P0=0.000, P1=1.000, P2=0.000, P3=0.000)",
            "  * 7S: (P0=1.000, P1=0.000, P2=0.000, P3=0.000)",
        ]
    )


def test_exact_solver_certificate_snapshot_for_fixed_policy_position() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("6H 8H"),
            1: hand("AC"),
            2: hand("9H"),
            3: hand("AD"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    certificate = format_exact_solver_certificate(state, solve_full_information_against_policy(state, highest_legal_card_policy))

    assert certificate == "\n".join(
        [
            "Exact full-information certificate",
            "policy: fixed_policy",
            "current_player: P0",
            "hand_counts: (2, 1, 1, 1)",
            "legal_moves: [6H, 8H]",
            "value: (P0=1.000, P1=0.000, P2=0.000, P3=0.000)",
            "best_moves: [6H]",
            "chosen_move: 6H",
            "exhaustive: True",
            "search: states=6, cache_hits=0, terminals=2, deadlocks=0",
            "tie_break: root chooses best move against fixed future policy; tied root moves use deterministic card order",
            "move_values:",
            "  * 6H: (P0=1.000, P1=0.000, P2=0.000, P3=0.000)",
            "    8H: (P0=0.000, P1=0.000, P2=1.000, P3=0.000)",
        ]
    )


def test_enumerate_hidden_deals_exhausts_count_consistent_small_belief_set() -> None:
    table_state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(49, 1, 1, 0),
        current_player=0,
    )
    hidden = hand("5H 9H")
    own_cards = frozenset(full_deck() - table_state.played_cards() - set(hidden))
    knowledge = PlayerKnowledge(player=0, hand=own_cards)

    enumeration = enumerate_hidden_deals(table_state, knowledge)

    assert enumeration.exhaustive
    assert enumeration.deal_count == 2
    assert {deal.hand(1) for deal in enumeration.deals} == {hand("5H"), hand("9H")}
    assert all(len(deal.hand(2)) == 1 for deal in enumeration.deals)
    assert all(len(deal.hand(3)) == 0 for deal in enumeration.deals)


def test_enumerate_hidden_deals_respects_pass_constraints_with_known_counts() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 1, 2, 2),
        current_player=0,
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)), MoveEvent(1, None)),
    )
    knowledge = PlayerKnowledge(
        player=0,
        hand=hand("5H"),
        deck=hand("5H 7H 6H 8H AC AD AS"),
    )

    enumeration = enumerate_hidden_deals(state, knowledge)

    assert enumeration.exhaustive
    assert enumeration.deal_count == 18
    assert all(Card(Suit.HEARTS, 6) not in deal.hand(1) for deal in enumeration.deals)
    assert all(Card(Suit.HEARTS, 8) not in deal.hand(1) for deal in enumeration.deals)
    assert all(len(deal.hand(1)) == 1 for deal in enumeration.deals)
    assert all(len(deal.hand(2)) == 2 for deal in enumeration.deals)
    assert all(len(deal.hand(3)) == 2 for deal in enumeration.deals)


def test_enumerate_hidden_deals_respects_played_card_ownership_history() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=6, high=7)},
        hand_counts=(1, 0, 1, 1),
        current_player=0,
        history=(
            MoveEvent(0, Card(Suit.HEARTS, 7)),
            MoveEvent(1, Card(Suit.HEARTS, 6)),
        ),
    )
    knowledge = PlayerKnowledge(
        player=0,
        hand=hand("5H"),
        deck=hand("5H 6H 7H AC AD"),
    )

    enumeration = enumerate_hidden_deals(state, knowledge)

    assert enumeration.exhaustive
    assert enumeration.deal_count == 2
    assert all(Card(Suit.HEARTS, 6) not in deal.hand(2) for deal in enumeration.deals)
    assert all(Card(Suit.HEARTS, 6) not in deal.hand(3) for deal in enumeration.deals)


def test_enumerate_hidden_deals_returns_zero_when_pass_constraints_conflict_with_counts() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 1, 1, 0),
        current_player=0,
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)), MoveEvent(1, None)),
    )
    knowledge = PlayerKnowledge(
        player=0,
        hand=hand("5H"),
        deck=hand("5H 7H 6H 8H"),
    )

    enumeration = enumerate_hidden_deals(state, knowledge)

    assert enumeration.exhaustive
    assert enumeration.deal_count == 0
    assert enumeration.deals == ()


def test_exact_imperfect_information_move_value_is_exact_for_immediate_win() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 48, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("6H"))

    score = evaluate_move_exact_imperfect_information(state, knowledge, Card(Suit.HEARTS, 6))

    assert score.exhaustive
    assert score.hidden_deals == 2450
    assert score.expected_value == 1.0
    assert score.value_standard_error == 0.0


def test_exact_imperfect_information_certificate_snapshot_for_reduced_belief_position() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(3, 2, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        player=0,
        hand=hand("5H 6H 8H"),
        deck=hand("5H 6H 7H 8H 9H 7C AC AD"),
    )

    result = recommend_move_exact_imperfect_information(state, knowledge)
    assert result is not None
    certificate = format_exact_imperfect_information_certificate(state, knowledge, result)

    assert certificate == "\n".join(
        [
            "Exact imperfect-information certificate",
            "policy: exact_hidden_deal_expectation_with_full_information_rational_continuation",
            "continuation_model: opponents play full-information rationally from each materialized hidden deal",
            "solver_player: P0",
            "current_player: P0",
            "solver_hand: [5H, 6H, 8H]",
            "hand_counts: (3, 2, 1, 1)",
            "legal_moves: [6H, 8H]",
            "hidden_deals: 12",
            "exhaustive: True",
            "best_moves: [6H]",
            "chosen_move: 6H",
            "search: states=112, cache_hits=8, terminals=24, deadlocks=0",
            "move_expected_values:",
            "  * 6H: EV=0.500, vector=(P0=0.500, P1=0.000, P2=0.250, P3=0.250), outcomes=(P0=6, P1=0, P2=3, P3=3, neutral=0), deals=12, exhaustive=True, se=0.000",
            "    8H: EV=0.000, vector=(P0=0.000, P1=0.167, P2=0.500, P3=0.333), outcomes=(P0=0, P1=2, P2=6, P3=4, neutral=0), deals=12, exhaustive=True, se=0.000",
        ]
    )


def test_exact_imperfect_information_certificate_snapshot_for_multi_suit_belief_position() -> None:
    table = {
        s: SuitRun()
        for s in Suit
    } | {
        Suit.HEARTS: SuitRun(low=6, high=8),
        Suit.CLUBS: SuitRun(low=7, high=7),
    }
    state = GameState(
        table=table,
        hand_counts=(4, 2, 2, 2),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        player=0,
        hand=hand("5H 9H 6C 8C"),
        deck=hand("4H 5H 6H 7H 8H 9H 10H 6C 7C 8C 7D 7S AC AD"),
    )

    result = recommend_move_exact_imperfect_information(state, knowledge)
    assert result is not None
    certificate = format_exact_imperfect_information_certificate(state, knowledge, result)

    assert certificate == "\n".join(
        [
            "Exact imperfect-information certificate",
            "policy: exact_hidden_deal_expectation_with_full_information_rational_continuation",
            "continuation_model: opponents play full-information rationally from each materialized hidden deal",
            "solver_player: P0",
            "current_player: P0",
            "solver_hand: [6C, 8C, 5H, 9H]",
            "hand_counts: (4, 2, 2, 2)",
            "legal_moves: [6C, 8C, 5H, 9H]",
            "hidden_deals: 90",
            "exhaustive: True",
            "best_moves: [6C, 8C]",
            "chosen_move: 6C",
            "search: states=3072, cache_hits=948, terminals=216, deadlocks=0",
            "move_expected_values:",
            "  * 6C: EV=0.667, vector=(P0=0.667, P1=0.067, P2=0.133, P3=0.133), outcomes=(P0=60, P1=6, P2=12, P3=12, neutral=0), deals=90, exhaustive=True, se=0.000",
            "    8C: EV=0.667, vector=(P0=0.667, P1=0.067, P2=0.133, P3=0.133), outcomes=(P0=60, P1=6, P2=12, P3=12, neutral=0), deals=90, exhaustive=True, se=0.000",
            "    5H: EV=0.400, vector=(P0=0.400, P1=0.200, P2=0.200, P3=0.200), outcomes=(P0=36, P1=18, P2=18, P3=18, neutral=0), deals=90, exhaustive=True, se=0.000",
            "    9H: EV=0.400, vector=(P0=0.400, P1=0.200, P2=0.200, P3=0.200), outcomes=(P0=36, P1=18, P2=18, P3=18, neutral=0), deals=90, exhaustive=True, se=0.000",
        ]
    )


def test_full_information_from_hands_derives_trailing_public_pass_count() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        current_player=3,
        history=(
            MoveEvent(0, Card(Suit.HEARTS, 7)),
            MoveEvent(1, None),
            MoveEvent(2, None),
        ),
    )

    full_state = FullInformationState.from_hands(
        {
            0: hand("6H"),
            1: hand("AC"),
            2: hand("AD"),
            3: hand("AS"),
        },
        state,
    )

    assert full_state.consecutive_passes == 2


def test_exact_imperfect_information_reuses_same_hidden_deal_set_for_all_candidate_moves() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(3, 2, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        player=0,
        hand=hand("5H 6H 8H"),
        deck=hand("5H 6H 7H 8H 9H 7C AC AD"),
    )

    result = recommend_move_exact_imperfect_information(state, knowledge, max_deals=5)

    assert result is not None
    assert not result.exhaustive
    assert {score.hidden_deals for score in result.move_scores} == {5}
    assert all(not score.exhaustive for score in result.move_scores)
    assert all(math.isnan(score.value_standard_error) for score in result.move_scores)


def test_exact_imperfect_information_uses_vector_tie_breaker() -> None:
    move = choose_expected_value_vector_move(
        0,
        {
            Card(Suit.CLUBS, 6): (0.5, 0.5, 0.0, 0.0),
            Card(Suit.CLUBS, 8): (0.5, 0.0, 0.5, 0.0),
        },
    )

    assert move == Card(Suit.CLUBS, 8)


def test_recommend_move_exact_imperfect_information_reports_non_exhaustive_limit() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 48, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("6H"))

    result = recommend_move_exact_imperfect_information(state, knowledge, max_deals=2)

    assert result is not None
    assert not result.exhaustive
    assert result.hidden_deals == 2
    assert result.chosen_move in state.legal_moves(knowledge.hand)
    assert all(math.isnan(score.value_standard_error) for score in result.move_scores)


def test_full_information_fixed_policy_solver_chooses_best_root_response() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("6H 8H"),
            1: hand("AC"),
            2: hand("9H"),
            3: hand("AD"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    result = solve_full_information_against_policy(state, highest_legal_card_policy)

    assert result.policy_name == "fixed_policy"
    assert result.chosen_move == Card(Suit.HEARTS, 6)
    assert result.value == (1.0, 0.0, 0.0, 0.0)


def test_exact_solver_certificate_includes_move_values_and_stats() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("6H"),
            1: hand("8H"),
            2: hand("7S"),
            3: hand("7D"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )
    result = solve_full_information(state)

    certificate = format_exact_solver_certificate(state, result)

    assert "Exact full-information certificate" in certificate
    assert "chosen_move: 6H" in certificate
    assert "move_values:" in certificate
    assert "states=" in certificate


def test_exact_imperfect_information_certificate_reports_exhaustive_status() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 48, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("6H"))

    result = recommend_move_exact_imperfect_information(state, knowledge, max_deals=2)
    assert result is not None
    certificate = format_exact_imperfect_information_certificate(state, knowledge, result)

    assert "Exact imperfect-information certificate" in certificate
    assert "hidden_deals: 2" in certificate
    assert "exhaustive: False" in certificate
    assert "uncertainty=not available for deterministic truncation" in certificate
    assert "se=" not in certificate
    assert "move_expected_values:" in certificate


def test_lowest_legal_card_policy_rejects_forced_pass_state() -> None:
    state = FullInformationState.from_hands(
        {
            0: hand("AC"),
            1: hand("AD"),
            2: hand("6H"),
            3: hand("AS"),
        },
        GameState(
            table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
            current_player=0,
        ),
    )

    try:
        lowest_legal_card_policy(state)
    except ValueError as error:
        assert "no legal moves" in str(error)
    else:
        raise AssertionError("expected fixed policy helper to reject forced-pass state")


def test_proof_benchmark_tiers_include_harder_positions() -> None:
    fast_positions = benchmark_positions()
    all_positions = benchmark_positions(include_hard=True)

    assert {name for name, _ in fast_positions} == FAST_BENCHMARKS
    assert {name for name, _ in all_positions} == FAST_BENCHMARKS | HARD_BENCHMARKS


def test_hard_proof_benchmark_reaches_thousand_state_scale() -> None:
    state = dict(hard_benchmark_positions())["four_suit_5_cards_each"]

    result = solve_full_information(state)

    assert result.states_evaluated >= 1000
    assert result.deadlock_states == 0


def test_proof_eval_reports_engine_regret_against_exact_ev() -> None:
    scenario = dict((scenario.name, scenario) for scenario in evaluation_scenarios())["multi_suit_belief_ev"]

    evaluation = evaluate_scenario(scenario, include_monte_carlo=False)

    assert evaluation.exhaustive
    assert evaluation.hidden_deals == 90
    assert evaluation.exact_choice in evaluation.scenario.state.legal_moves(evaluation.scenario.knowledge.hand)
    assert evaluation.heuristic_choice in evaluation.scenario.state.legal_moves(evaluation.scenario.knowledge.hand)
    assert evaluation.heuristic_regret >= 0.0
    assert {move.card for move in evaluation.move_evaluations} == set(
        evaluation.scenario.state.legal_moves(evaluation.scenario.knowledge.hand)
    )
