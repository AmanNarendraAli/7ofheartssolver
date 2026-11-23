import itertools
import math
import random

from seven_hearts import (
    BoundedInformationLimitedPolicyCache,
    BoundedRolloutTranspositionCache,
    Card,
    EXACT_LAYOUT_VOTE_POLICY,
    FirstOutSolver,
    FullInformationState,
    GameState,
    FullGameAgent,
    FullGameResult,
    PAIRED_METRICS,
    MoveEvent,
    PlayerKnowledge,
    StrategyCandidate,
    StrategyWeights,
    Suit,
    SuitRun,
    apply_known_play,
    build_opponent_model,
    card_bit,
    card_from_index,
    card_index,
    card_set_mask,
    deal_random_hands,
    enumerate_hidden_deals,
    evaluate_move_exact_information_limited_policy,
    evaluate_move_perfect_information_rollout_ev,
    estimate_complete_game_metrics,
    estimate_strategy_self_play,
    evaluate_duplicate_deal_seat_rotation,
    evaluate_move_monte_carlo,
    evaluate_move_exact_imperfect_information,
    format_exact_imperfect_information_certificate,
    format_exact_solver_certificate,
    future_chain_impact,
    first_empty_player,
    first_empty_player_mask,
    full_deck,
    hand_count_tuple,
    hand_masks_from_hands,
    highest_legal_card_policy,
    initial_state_for_hands,
    lowest_legal_card_policy,
    mask_after_play,
    mask_hand_count_tuple,
    mask_to_cards,
    recommend_move,
    recommend_move_exact_imperfect_information,
    recommend_move_exact_layout_vote,
    recommend_move_exact_information_limited_policy,
    recommend_move_information_limited_monte_carlo,
    recommend_move_monte_carlo,
    reduced_deck,
    rollout_transposition_cache_key,
    rollout_information_limited,
    rollout_oracle,
    sample_hidden_deal,
    sample_hidden_deals,
    score_oracle_move,
    score_move,
    choose_rational_move,
    solve_full_information,
    solve_full_information_against_policy,
    simulate_full_game_to_completion,
    summarize_duplicate_deal_results,
    table_suit_codes,
    time_to_playable,
)
from best_response import BestResponseSolver, evaluate_assumed_table, hero_policy, opponent_policy, position_posterior
from proof_benchmark import hard_benchmark_positions
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


def test_card_bitmask_round_trip_and_legal_move_mask() -> None:
    cards = hand("AC 7H KS")
    mask = card_set_mask(cards)

    assert set(mask_to_cards(mask)) == set(cards)
    for card in cards:
        assert card_from_index(card_index(card)) == card
        assert mask & card_bit(card)

    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)})
    legal_mask = card_set_mask(state.legal_moves(hand("6H 9H 7S")))

    assert legal_mask == card_bit(Card(Suit.HEARTS, 6)) | card_bit(Card(Suit.SPADES, 7))


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


def test_opponent_model_rejects_history_that_does_not_match_table() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=6, high=7)},
        history=(MoveEvent(0, Card(Suit.HEARTS, 7)),),
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 5H 9H 10H JH QH KH 7C 7D 7S"))

    try:
        build_opponent_model(state, knowledge)
    except ValueError as error:
        assert "history" in str(error)
    else:
        raise AssertionError("expected mismatched public history to raise ValueError")


def test_recommend_move_returns_none_when_forced_to_pass() -> None:
    state = GameState(table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)})
    knowledge = PlayerKnowledge(player=0, hand=hand("AH 2H 3H 4H 9H 10H"))

    assert recommend_move(state, knowledge) is None




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

    assert result.samples == 3
    assert result.timeout_rate == 0.0


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




def test_recommend_move_monte_carlo_skips_forced_move_sampling() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        current_player=0,
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("6H AS"))

    result = recommend_move_monte_carlo(state, knowledge, samples_per_move=20, rng=random.Random(6))

    assert result is not None
    assert result.card == Card(Suit.HEARTS, 6)
    assert result.samples == 0
    assert result.policy_name == "oracle_greedy_full_information_forced_move"


def test_recommend_move_monte_carlo_shortcuts_immediate_win() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 2, 3, 4),
        current_player=0,
    )
    knowledge = PlayerKnowledge(player=0, hand=hand("6H"))

    result = recommend_move_monte_carlo(state, knowledge, samples_per_move=20, rng=random.Random(6))

    assert result is not None
    assert result.card == Card(Suit.HEARTS, 6)
    assert result.samples == 0
    assert result.win_rate == 1.0
    assert result.average_finish_margin == 2.0
    assert result.policy_name == "oracle_greedy_full_information_immediate_win"




def test_rollout_information_limited_detects_first_player_out() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 1, 1, 1),
        current_player=0,
    )
    hands = {
        0: set(hand("6H")),
        1: set(hand("8H")),
        2: set(hand("7S")),
        3: set(hand("7C")),
    }

    result = rollout_information_limited(state, hands, max_turns=10, rng=random.Random(11))

    assert result.winner == 0
    assert result.final_hand_counts[0] == 0
    assert not result.timed_out


def test_mask_hand_helpers_match_set_helpers() -> None:
    hands = {
        0: set(hand("6H 8H")),
        1: set(hand("5H")),
        2: set(),
        3: set(hand("7S 7C")),
    }

    hand_masks = hand_masks_from_hands(hands)

    assert mask_hand_count_tuple(hand_masks) == hand_count_tuple(hands)
    assert first_empty_player_mask(hand_masks) == 2
    assert set(mask_to_cards(hand_masks[0])) == hands[0]


def test_mask_after_play_matches_apply_known_play_hands() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 1, 1, 1),
        current_player=0,
    )
    hands = {
        0: set(hand("6H 8H")),
        1: set(hand("5H")),
        2: set(hand("7S")),
        3: set(hand("7C")),
    }

    _, next_hands = apply_known_play(state, hands, 0, Card(Suit.HEARTS, 6))
    next_masks = mask_after_play(hand_masks_from_hands(hands), 0, Card(Suit.HEARTS, 6))

    assert next_masks == hand_masks_from_hands(next_hands)


def test_mask_after_play_rejects_card_not_in_hand() -> None:
    masks = hand_masks_from_hands(
        {
            0: set(hand("8H")),
            1: set(hand("5H")),
            2: set(hand("7S")),
            3: set(hand("7C")),
        }
    )

    try:
        mask_after_play(masks, 0, Card(Suit.HEARTS, 6))
    except ValueError as error:
        assert "not in hand mask" in str(error)
    else:
        raise AssertionError("expected missing-card mask play to raise ValueError")








def test_rollout_information_limited_transposition_cache_preserves_result() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 2, 1, 1),
        current_player=0,
    )
    hands = {
        0: set(hand("6H 8H")),
        1: set(hand("5H 9H")),
        2: set(hand("7S")),
        3: set(hand("7C")),
    }
    cache = BoundedRolloutTranspositionCache()

    uncached = rollout_information_limited(state, hands, max_turns=10, rng=random.Random(11))
    cached_first = rollout_information_limited(
        state,
        hands,
        max_turns=10,
        rng=random.Random(11),
        rollout_cache=cache,
    )
    cached_second = rollout_information_limited(
        state,
        hands,
        max_turns=10,
        rng=random.Random(11),
        rollout_cache=cache,
    )

    assert cached_first == uncached
    assert cached_second == uncached
    assert len(cache) > 0


def test_rollout_transposition_cache_key_includes_weights_and_budget() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 2, 1, 1),
        current_player=0,
    )
    hands = {
        0: set(hand("6H 8H")),
        1: set(hand("5H 9H")),
        2: set(hand("7S")),
        3: set(hand("7C")),
    }
    deck = frozenset(card for cards in hands.values() for card in cards) | {Card(Suit.HEARTS, 7)}
    default_key = rollout_transposition_cache_key(state, hands, deck, StrategyWeights(), "heuristic_greedy", 1.0, 10)
    changed_weights_key = rollout_transposition_cache_key(
        state,
        hands,
        deck,
        StrategyWeights(response_penalty=0.9),
        "heuristic_greedy",
        1.0,
        10,
    )
    changed_budget_key = rollout_transposition_cache_key(state, hands, deck, StrategyWeights(), "heuristic_greedy", 1.0, 9)

    assert changed_weights_key != default_key
    assert changed_budget_key != default_key




def test_information_limited_monte_carlo_enumerates_when_belief_fits_sample_budget() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 1, 0, 0),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        0,
        hand("6H 8H"),
        deck=frozenset(hand("6H 8H 5H")) | {Card(Suit.HEARTS, 7)},
    )

    result = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=20,
        max_turns=20,
        rng=random.Random(12),
    )

    assert result is not None
    assert result.samples == 1
    assert result.exhaustive is True


def test_information_limited_monte_carlo_samples_when_belief_exceeds_sample_budget() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 1, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        0,
        hand("6H 8H"),
        deck=frozenset(hand("6H 8H 5H 9H 7S")) | {Card(Suit.HEARTS, 7)},
    )

    result = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=2,
        max_turns=20,
        rng=random.Random(12),
    )

    assert result is not None
    assert result.samples == 2
    assert result.exhaustive is False


def test_information_limited_monte_carlo_shared_policy_cache_preserves_result() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {
            Suit.HEARTS: SuitRun(low=7, high=7),
            Suit.SPADES: SuitRun(low=7, high=7),
        },
        hand_counts=(4, 2, 2, 2),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        0,
        hand("6H 8H 6S 8S"),
        deck=frozenset(hand("6H 8H 6S 8S 5H 9H 5S 9S 7C 7D"))
        | {Card(Suit.HEARTS, 7), Card(Suit.SPADES, 7)},
    )
    shared_cache = BoundedInformationLimitedPolicyCache()

    uncached = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=4,
        max_turns=20,
        rng=random.Random(12),
    )
    cached = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=4,
        max_turns=20,
        rng=random.Random(12),
        policy_cache=shared_cache,
    )
    fully_cached = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=4,
        max_turns=20,
        rng=random.Random(12),
        policy_cache=BoundedInformationLimitedPolicyCache(),
        rollout_cache=BoundedRolloutTranspositionCache(),
    )

    assert cached == uncached
    assert fully_cached == uncached


def test_information_limited_monte_carlo_accepts_legacy_greedy_alias() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 1, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        0,
        hand("6H 8H"),
        deck=frozenset(hand("6H 8H 5H 9H 7S")) | {Card(Suit.HEARTS, 7)},
    )

    renamed = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=4,
        max_turns=20,
        rng=random.Random(12),
        policy="heuristic_greedy",
    )
    legacy = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=4,
        max_turns=20,
        rng=random.Random(12),
        policy="greedy",
    )

    assert legacy == renamed
    assert legacy is not None
    assert legacy.policy_name == "information_limited_heuristic_greedy"


def test_information_limited_monte_carlo_softmax_does_not_use_deterministic_caches() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 1, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        0,
        hand("6H 8H"),
        deck=frozenset(hand("6H 8H 5H 9H 7S")) | {Card(Suit.HEARTS, 7)},
    )
    policy_cache = BoundedInformationLimitedPolicyCache()
    rollout_cache = BoundedRolloutTranspositionCache()

    result = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=3,
        max_turns=20,
        rng=random.Random(13),
        policy="softmax",
        rationality=0.5,
        policy_cache=policy_cache,
        rollout_cache=rollout_cache,
    )

    assert result is not None
    assert result.samples == 3
    assert result.policy_name == "information_limited_softmax_heuristic_rationality_0.5"
    assert len(policy_cache) == 0
    assert len(rollout_cache) == 0


def test_recommend_move_information_limited_monte_carlo_skips_forced_move_sampling() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 1, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        0,
        hand("6H AS"),
        deck=frozenset(hand("6H AS 5H 9H 7S")) | {Card(Suit.HEARTS, 7)},
    )

    result = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=20,
        max_turns=20,
        rng=random.Random(12),
    )

    assert result is not None
    assert result.card == Card(Suit.HEARTS, 6)
    assert result.samples == 0
    assert result.policy_name == "information_limited_heuristic_greedy_forced_move"


def test_recommend_move_information_limited_monte_carlo_shortcuts_immediate_win() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 2, 3, 4),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        0,
        hand("6H"),
        deck=frozenset(hand("6H 5H 9H 7S 7C")) | {Card(Suit.HEARTS, 7)},
    )

    result = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=20,
        max_turns=20,
        rng=random.Random(12),
    )

    assert result is not None
    assert result.card == Card(Suit.HEARTS, 6)
    assert result.samples == 0
    assert result.win_rate == 1.0
    assert result.average_finish_margin == 2.0
    assert result.policy_name == "information_limited_heuristic_greedy_immediate_win"


def test_perfect_information_counterpart_oracle_uses_true_deal_without_hidden_sampling() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 1, 1, 1),
        current_player=0,
    )
    first_world = {
        0: set(hand("6H 8H")),
        1: set(hand("5H")),
        2: set(hand("9H")),
        3: set(hand("7S")),
    }
    second_world = {
        0: set(hand("6H 8H")),
        1: set(hand("9H")),
        2: set(hand("5H")),
        3: set(hand("7S")),
    }

    first_score = evaluate_move_perfect_information_rollout_ev(
        state,
        first_world,
        0,
        Card(Suit.HEARTS, 6),
        max_turns=1,
        rng=random.Random(21),
    )
    second_score = evaluate_move_perfect_information_rollout_ev(
        state,
        second_world,
        0,
        Card(Suit.HEARTS, 6),
        max_turns=1,
        rng=random.Random(21),
    )

    assert first_score.samples == 1
    assert second_score.samples == 1
    assert first_score.policy_name == "perfect_information_counterpart_information_limited_heuristic_greedy"
    assert first_score.score != second_score.score








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


def test_reduced_deck_is_centered_on_seven() -> None:
    deck = reduced_deck(cards_per_suit=5)

    assert len(deck) == 20
    assert {card.rank for card in deck if card.suit == Suit.HEARTS} == {5, 6, 7, 8, 9}
    assert Card(Suit.HEARTS, 7) in deck


def test_deal_random_hands_supports_reduced_cards_per_suit() -> None:
    hands = deal_random_hands(random.Random(7), cards_per_suit=5)
    all_cards = set().union(*hands.values())

    assert len(all_cards) == 20
    assert all(len(hand_cards) == 5 for hand_cards in hands.values())
    assert all_cards == reduced_deck(5)


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




def test_estimate_complete_game_metrics_reports_rates() -> None:
    estimate = estimate_complete_game_metrics(4, random.Random(9), max_turns=300)

    assert estimate.games == 4
    assert abs(sum(estimate.win_rates) - 1.0) < 0.000001
    assert estimate.average_turns > 0.0


def test_full_game_to_completion_reports_final_ranks_and_cards_left() -> None:
    hands = deal_random_hands(random.Random(12))
    agents = (
        FullGameAgent("Random", "random"),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent("Heuristic", "heuristic"),
        FullGameAgent("Oracle", "oracle_greedy"),
    )

    result = simulate_full_game_to_completion(hands, agents, random.Random(13), max_turns=500)

    assert result.winner in {0, 1, 2, 3}
    assert sorted(result.ranks_by_seat) == [1, 2, 3, 4]
    assert result.final_hand_counts[result.winner] == 0
    assert result.first_out_hand_counts is not None
    assert result.first_out_hand_counts[result.winner] == 0
    assert all(
        first_out >= final for first_out, final in zip(result.first_out_hand_counts, result.final_hand_counts)
    )
    assert sum(1 for count in result.final_hand_counts if count > 0) <= 1
    assert not result.timed_out


def test_duplicate_deal_seat_rotation_reports_paired_comparisons() -> None:
    agents = (
        FullGameAgent("Monte Carlo", "heuristic"),
        FullGameAgent("Random", "random"),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent("Heuristic", "heuristic"),
    )

    evaluation = evaluate_duplicate_deal_seat_rotation(
        agents,
        deals=2,
        rng=random.Random(14),
        max_turns=500,
    )

    assert evaluation.games == 8
    assert evaluation.deals == 2
    assert evaluation.rotations_per_deal == 4
    assert {summary.agent_name for summary in evaluation.agent_summaries} == {
        "Monte Carlo",
        "Random",
        "Greedy",
        "Heuristic",
    }
    assert all(summary.seats_played == 8 for summary in evaluation.agent_summaries)
    assert all(summary.turns > 0 for summary in evaluation.agent_summaries)
    assert {(comparison.metric, comparison.baseline_agent) for comparison in evaluation.paired_comparisons} == {
        (metric, baseline) for metric in PAIRED_METRICS for baseline in ("Random", "Greedy", "Heuristic")
    }
    assert all(comparison.comparisons == 8 for comparison in evaluation.paired_comparisons)
    assert evaluation.average_turns > 0.0


def test_paired_comparisons_measure_first_out_outcomes() -> None:
    names = ("Monte Carlo", "Heuristic", "Random", "Greedy")
    primary_wins = FullGameResult(
        agent_names_by_seat=names,
        winner=0,
        finish_order=(0, 2, 1, 3),
        ranks_by_seat=(1, 3, 2, 4),
        final_hand_counts=(0, 0, 0, 2),
        turns_played=50,
        first_out_hand_counts=(0, 4, 3, 6),
    )
    heuristic_wins = FullGameResult(
        agent_names_by_seat=names,
        winner=1,
        finish_order=(1, 0, 2, 3),
        ranks_by_seat=(2, 1, 3, 4),
        final_hand_counts=(0, 0, 0, 1),
        turns_played=50,
        first_out_hand_counts=(2, 0, 5, 5),
    )

    evaluation = summarize_duplicate_deal_results([primary_wins, heuristic_wins], deals=1, rotations_per_deal=2)

    win = evaluation.paired("first_out_win", "Heuristic")
    cards = evaluation.paired("cards_at_first_out", "Heuristic")
    rank = evaluation.paired("rank", "Heuristic")
    assert (win.average_advantage, win.comparisons) == (0.0, 2)
    assert cards.average_advantage == ((4 - 0) + (0 - 2)) / 2
    assert rank.average_advantage == ((3 - 1) + (1 - 2)) / 2
    assert evaluation.paired("first_out_win", "Random").average_advantage == 0.5
    assert evaluation.summary("Monte Carlo").average_cards_at_first_out == 1.0
    assert evaluation.summary("Monte Carlo").win_rate == 0.5


def test_duplicate_deal_evaluation_reports_progress() -> None:
    agents = (
        FullGameAgent("Monte Carlo", "heuristic"),
        FullGameAgent("Random", "random"),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent("Heuristic", "heuristic"),
    )
    progress: list[tuple[int, int, int, int]] = []

    evaluate_duplicate_deal_seat_rotation(
        agents,
        deals=1,
        rng=random.Random(15),
        max_turns=500,
        progress_callback=lambda completed, total, deal, rotation, result: progress.append(
            (completed, total, deal, rotation)
        ),
    )

    assert progress == [
        (1, 4, 1, 1),
        (2, 4, 1, 2),
        (3, 4, 1, 3),
        (4, 4, 1, 4),
    ]


def test_duplicate_deal_evaluation_records_deterministic_game_results() -> None:
    agents = (
        FullGameAgent("Monte Carlo", "heuristic"),
        FullGameAgent("Random", "random"),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent("Heuristic", "heuristic"),
    )

    first = evaluate_duplicate_deal_seat_rotation(
        agents,
        deals=1,
        rng=random.Random(17),
        max_turns=500,
        cards_per_suit=5,
    )
    second = evaluate_duplicate_deal_seat_rotation(
        agents,
        deals=1,
        rng=random.Random(17),
        max_turns=500,
        cards_per_suit=5,
    )

    assert len(first.game_results) == 4
    assert [game.seed for game in first.game_results] == [game.seed for game in second.game_results]
    assert [game.result for game in first.game_results] == [game.result for game in second.game_results]
    assert [(game.deal_index, game.rotation_index) for game in first.game_results] == [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ]


def test_full_game_eval_report_runs_are_unique_and_parameterized() -> None:
    import argparse
    import tempfile
    from pathlib import Path

    from full_game_eval import build_run_metadata, unique_run_output_dir, write_metadata_reports

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        first_run = unique_run_output_dir(output_dir)
        second_run = unique_run_output_dir(output_dir)
        assert first_run != second_run
        assert first_run.exists()
        assert second_run.exists()

        agents = (
            FullGameAgent(
                "Monte Carlo",
                "information_limited_monte_carlo",
                samples_per_move=5,
                rollout_max_turns=30,
                weights=StrategyWeights(self_unlock=4.0),
            ),
            FullGameAgent("Random", "random"),
            FullGameAgent("Greedy", "greedy_furthest_from_seven"),
            FullGameAgent("Heuristic", "heuristic"),
        )
        args = argparse.Namespace(
            deals=3,
            max_turns=200,
            samples_per_move=5,
            rollout_max_turns=30,
            cards_per_suit=7,
            seed=11,
            output_dir=output_dir,
            fast=False,
            progress_every=0,
            workers=1,
        )

        metadata = build_run_metadata(args, agents, first_run, elapsed_seconds=1.25)
        json_path, run_csv_path, agent_csv_path = write_metadata_reports(metadata, agents, first_run)

        assert metadata["eval_args"]["samples_per_move"] == 5
        assert metadata["agents"][0]["policy"] == "information_limited_monte_carlo"
        assert metadata["agents"][0]["uses_monte_carlo"]
        assert metadata["agents"][0]["uses_heuristic_weights"]
        assert metadata["agents"][0]["monte_carlo"]["rollout_max_turns"] == 30
        assert metadata["agents"][0]["heuristic_weights"]["self_unlock"] == 4.0
        assert metadata["agents"][0]["monte_carlo"]["score_weights"]["monte_carlo_win_rate_weight"] == 100.0
        assert metadata["implementation_defaults"]["rollout_transposition_cache_max_entries"] == 50_000
        assert json_path.exists()
        assert "eval_args.samples_per_move,5" in run_csv_path.read_text(encoding="utf-8")
        agent_csv = agent_csv_path.read_text(encoding="utf-8")
        assert "Monte Carlo,information_limited_monte_carlo,heuristic_weights.self_unlock,4.0" in agent_csv
        assert "Random,random,heuristic_weights.self_unlock" not in agent_csv


def test_duplicate_deal_evaluation_supports_reduced_decks() -> None:
    agents = (
        FullGameAgent("Monte Carlo", "information_limited_monte_carlo", samples_per_move=1, rollout_max_turns=20),
        FullGameAgent("Random", "random"),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent("Heuristic", "heuristic"),
    )

    evaluation = evaluate_duplicate_deal_seat_rotation(
        agents,
        deals=1,
        rng=random.Random(16),
        max_turns=200,
        cards_per_suit=5,
    )

    assert evaluation.games == 4
    assert evaluation.timeout_rate == 0.0
    assert all(summary.average_cards_left <= 5 for summary in evaluation.agent_summaries)


def test_full_game_eval_summary_csvs_include_big_run_reporting_columns() -> None:
    import tempfile
    from pathlib import Path

    from full_game_eval import write_summary_csv

    agents = (
        FullGameAgent("Monte Carlo", "information_limited_monte_carlo", samples_per_move=1, rollout_max_turns=20),
        FullGameAgent(
            "CounterpartOracle",
            "perfect_information_counterpart_oracle",
            samples_per_move=1,
            rollout_max_turns=20,
        ),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent("Heuristic", "heuristic"),
    )

    evaluation = evaluate_duplicate_deal_seat_rotation(
        agents,
        deals=1,
        rng=random.Random(19),
        max_turns=200,
        cards_per_suit=5,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        agent_path, paired_path, _ = write_summary_csv(evaluation, Path(temp_dir))
        agent_csv = agent_path.read_text(encoding="utf-8")
        paired_csv = paired_path.read_text(encoding="utf-8")

    assert "win_rate_standard_error,average_cards_at_first_out" in agent_csv
    assert "turns,forced_passes,single_legal_moves,immediate_wins,sampled_mc_decisions" in agent_csv
    assert "ci95_low,ci95_high,average_oracle_gap" in paired_csv
    assert "Monte Carlo,CounterpartOracle,first_out_win" in paired_csv


def test_tune_eval_random_search_writes_ranked_candidate_reports() -> None:
    import argparse
    import tempfile
    from pathlib import Path

    import numpy as np

    from tune_eval import parse_int_choices, run_tuning

    with tempfile.TemporaryDirectory() as temp_dir:
        args = argparse.Namespace(
            mode="heuristic",
            candidates=2,
            deals=1,
            cards_per_suit=5,
            max_turns=200,
            seed=23,
            eval_seed=7,
            workers=1,
            output_dir=Path(temp_dir),
            progress_every=0,
            samples_per_move=4,
            rollout_max_turns=20,
            mc_samples_choices=parse_int_choices("4,8"),
            mc_rollout_turn_choices=parse_int_choices("20,40"),
        )

        run_dir, paths = run_tuning(args)
        summary_path = run_dir / "candidate_summary.npz"
        parameters_path = run_dir / "candidate_parameters.npz"
        metadata_path = run_dir / "run_metadata.json"

        assert summary_path in paths
        assert parameters_path in paths
        assert metadata_path in paths
        with np.load(summary_path, allow_pickle=True) as summary:
            assert summary.files == [
                "rank",
                "candidate_id",
                "mode",
                "score",
                "target_baseline",
                "win_advantage",
                "win_advantage_standard_error",
                "win_advantage_ci95_low",
                "win_advantage_ci95_high",
                "cards_at_first_out_advantage",
                "cards_at_first_out_advantage_standard_error",
                "candidate_win_rate",
                "candidate_win_rate_standard_error",
                "candidate_average_cards_at_first_out",
                "candidate_rank",
                "sampled_mc_decisions",
                "samples_per_move",
                "rollout_max_turns",
                "games",
                "timeout_rate",
            ]
            assert summary["rank"].tolist() == [1, 2]
            assert "c0000" in summary["candidate_id"].tolist()
            assert summary["score"].dtype == np.float64
            assert summary["games"].dtype == np.int64
        with np.load(parameters_path, allow_pickle=True) as parameters:
            assert parameters.files == ["candidate_id", "parameter", "value"]
            assert "weights.self_unlock" in parameters["parameter"].tolist()
            assert parameters["value"].dtype == np.float64
        assert '"mode": "heuristic"' in metadata_path.read_text(encoding="utf-8")


def test_duplicate_deal_evaluation_can_trace_mc_heuristic_decisions() -> None:
    import tempfile
    from pathlib import Path

    from full_game_eval import write_mc_heuristic_decisions_csv

    agents = (
        FullGameAgent("Monte Carlo", "information_limited_monte_carlo", samples_per_move=1, rollout_max_turns=20),
        FullGameAgent("Random", "random"),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent("Heuristic", "heuristic"),
    )

    evaluation = evaluate_duplicate_deal_seat_rotation(
        agents,
        deals=1,
        rng=random.Random(18),
        max_turns=200,
        cards_per_suit=5,
        trace_mc_heuristic=True,
    )
    traces = [
        trace
        for game_result in evaluation.game_results
        for trace in game_result.result.decision_traces
    ]

    assert traces
    assert all(trace.agent_name == "Monte Carlo" for trace in traces)
    assert all(trace.final_rank is not None for trace in traces)
    assert all(trace.final_cards_left is not None for trace in traces)

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = write_mc_heuristic_decisions_csv(evaluation, Path(temp_dir))
        csv_text = csv_path.read_text(encoding="utf-8")

    assert "disagreed,heuristic_card,mc_card" in csv_text
    assert "Monte Carlo" in csv_text


def test_opponent_model_reports_impossible_known_counts() -> None:
    model = build_opponent_model(
        GameState(hand_counts=(13, 52, None, None)),
        PlayerKnowledge(player=0, hand=hand("AH")),
    )

    assert model.consistency_errors()




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
    try:
        state.assert_card_conservation(hand("6H 7H 8H 7S 7D 7C"))
    except ValueError as error:
        assert "missing [7C]" in str(error)
    else:
        raise AssertionError("expected an undeclared missing card to fail conservation")


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
    assert result.best_moves == ()
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


def test_full_information_memoized_solver_matches_random_tiny_brute_force() -> None:
    rng = random.Random(12)

    for _ in range(40):
        state, deck = random_reduced_full_information_state(rng)
        state.assert_card_conservation(deck)

        result = solve_full_information(state)

        assert result.value == brute_force_full_information_value(state)
        assert result.chosen_move in state.legal_moves() or result.chosen_move is None
        assert result.deadlock_states == 0


def random_reduced_decision_points(rng: random.Random, cards_per_suit: int, deals: int):
    """Yield (state, hands) at every multi-legal decision of random-play reduced games, before first-out."""
    for _ in range(deals):
        hands = deal_random_hands(rng, cards_per_suit=cards_per_suit)
        state = initial_state_for_hands(hands)
        while first_empty_player(hands) is None:
            legal = state.legal_moves(hands[state.current_player])
            if len(legal) >= 2:
                yield state, {player: set(cards) for player, cards in hands.items()}
            card = rng.choice(legal) if legal else None
            state, hands = apply_known_play(state, hands, state.current_player, card)


def test_first_out_solver_matches_exact_solver_on_reduced_games() -> None:
    rng = random.Random(21)
    checked = 0
    for cards_per_suit, deals in ((3, 40), (4, 40), (5, 12)):
        for state, hands in random_reduced_decision_points(rng, cards_per_suit, deals):
            player = state.current_player
            exact = solve_full_information(FullInformationState.from_hands(hands, state))
            legal = state.legal_moves(hands[player])
            solver = FirstOutSolver(hands)

            assert solver.move_wins(state.table, player, legal) == [
                exact.move_values[card][player] == 1.0 for card in legal
            ]
            assert solver.winner(table_suit_codes(state.table), player) == exact.value.index(1.0)
            checked += 1
    assert checked > 200


def test_first_out_solver_finishes_a_full_deck_deal() -> None:
    hands = deal_random_hands(random.Random(0))
    state = initial_state_for_hands(hands)
    solver = FirstOutSolver(hands)

    winner = solver.winner(table_suit_codes(state.table), state.current_player)

    assert winner in range(4)
    assert solver.states_evaluated < len(solver.memo)


def test_exact_layout_vote_picks_the_move_that_wins_most_layouts() -> None:
    rng = random.Random(22)
    found = 0
    for state, hands in random_reduced_decision_points(rng, 4, 60):
        player = state.current_player
        knowledge = PlayerKnowledge(player, frozenset(hands[player]), deck=frozenset(reduced_deck(4)))
        enumeration = enumerate_hidden_deals(state, knowledge, max_deals=65)
        if not enumeration.exhaustive:
            continue
        legal = state.legal_moves(hands[player])
        wins = {card: 0 for card in legal}
        for deal in enumeration.deals:
            layout = {p: set(deal.hand(p)) for p in range(4)} | {player: set(hands[player])}
            exact = solve_full_information(FullInformationState.from_hands(layout, state))
            for card in legal:
                wins[card] += exact.move_values[card][player] == 1.0
        most = max(wins.values())
        leaders = [card for card in legal if wins[card] == most]
        if len(leaders) != 1:
            continue

        vote = recommend_move_exact_layout_vote(state, knowledge, layouts=64, rng=random.Random(0))

        assert vote is not None and vote.exhaustive
        assert vote.card == leaders[0]
        assert vote.win_rate == most / len(enumeration.deals)
        found += 1
    assert found >= 10


def test_exact_vote_agent_plays_full_reduced_games() -> None:
    agents = (
        FullGameAgent("Exact Vote", EXACT_LAYOUT_VOTE_POLICY, samples_per_move=8),
        FullGameAgent("Random", "random"),
        FullGameAgent("Greedy", "greedy_furthest_from_seven"),
        FullGameAgent("Heuristic", "heuristic"),
    )
    rng = random.Random(23)
    for _ in range(3):
        hands = deal_random_hands(rng, cards_per_suit=5)
        result = simulate_full_game_to_completion(
            hands, agents, rng=rng, max_turns=200, deck=frozenset(reduced_deck(5))
        )

        assert not result.timed_out
        assert sorted(result.finish_order) == [0, 1, 2, 3]


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


def test_exact_information_limited_policy_ev_is_exact_for_immediate_win() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(1, 1, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        0,
        hand("6H"),
        deck=frozenset(hand("6H 8H 7S 7C")) | {Card(Suit.HEARTS, 7)},
    )

    score = evaluate_move_exact_information_limited_policy(state, knowledge, Card(Suit.HEARTS, 6))

    assert score.exhaustive
    assert score.expected_value == 1.0
    assert score.outcome_counts[0] == score.hidden_deals


def test_recommend_move_exact_information_limited_policy_reports_model() -> None:
    state = GameState(
        table={s: SuitRun() for s in Suit} | {Suit.HEARTS: SuitRun(low=7, high=7)},
        hand_counts=(2, 1, 1, 1),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        0,
        hand("6H 8H"),
        deck=frozenset(hand("6H 8H 5H 9H 7S")) | {Card(Suit.HEARTS, 7)},
    )

    result = recommend_move_exact_information_limited_policy(state, knowledge, max_turns=20)

    assert result is not None
    assert result.exhaustive
    assert result.hidden_deals == 6
    assert result.chosen_move in state.legal_moves(knowledge.hand)
    assert result.policy_name == "exact_hidden_deal_expectation_with_information_limited_heuristic_greedy_policy"
    assert "information-limited heuristic-greedy" in result.continuation_model


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


def test_rational_tie_breaker_prefers_starving_next_player() -> None:
    move = choose_rational_move(
        0,
        {
            Card(Suit.CLUBS, 6): (0.5, 0.5, 0.0, 0.0),
            Card(Suit.CLUBS, 8): (0.5, 0.0, 0.5, 0.0),
        },
    )

    assert move == Card(Suit.CLUBS, 8)




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




def test_hard_proof_benchmark_reaches_thousand_state_scale() -> None:
    state = dict(hard_benchmark_positions())["four_suit_5_cards_each"]

    result = solve_full_information(state)

    assert result.states_evaluated >= 1000
    assert result.deadlock_states == 0


def test_proof_eval_reports_engine_regret_against_exact_ev() -> None:
    scenario = dict((scenario.name, scenario) for scenario in evaluation_scenarios())["multi_suit_belief_ev"]

    evaluation = evaluate_scenario(scenario, include_monte_carlo=False)

    exact_ev = {move.card: move.exact_ev for move in evaluation.move_evaluations}

    assert evaluation.exhaustive
    assert evaluation.exact_choice == Card(Suit.CLUBS, 6)
    assert set(exact_ev) == set(evaluation.scenario.state.legal_moves(evaluation.scenario.knowledge.hand))
    assert evaluation.heuristic_regret == exact_ev[evaluation.exact_choice] - exact_ev[evaluation.heuristic_choice]
    assert evaluation.monte_carlo_choice is None


def direct_layout_value(state, hands, hero, opponents, hero_choice) -> float:
    """First-out win probability for one fully known layout, by plain recursion over every action.

    hero_choice(state, hand_mask) gives the hero's distribution, or None to take the best move
    knowing the layout (a clairvoyant hero).
    """
    player = state.current_player
    legal = state.legal_moves(hands[player])
    if not legal:
        next_state, next_hands = apply_known_play(state, hands, player, None)
        return direct_layout_value(next_state, next_hands, hero, opponents, hero_choice)
    hand_mask = card_set_mask(hands[player])
    if player == hero:
        options = hero_choice(state, hand_mask)
        if options is None:
            options = [(card, None) for card in legal]
    else:
        options = opponents[player](state, player, hand_mask)

    def after(card) -> float:
        next_state, next_hands = apply_known_play(state, hands, player, card)
        if not next_hands[player]:
            return 1.0 if player == hero else 0.0
        return direct_layout_value(next_state, next_hands, hero, opponents, hero_choice)

    if player == hero and options[0][1] is None:
        return max(after(card) for card, _ in options)
    return sum(probability * after(card) for card, probability in options)


def all_layouts_for_hero(hero_hand, hero, deck):
    """Every deal of the unseen cards to the other seats, equal hand sizes, by brute force."""
    unseen = sorted(set(deck) - set(hero_hand), key=card_index)
    seats = [seat for seat in range(4) if seat != hero]
    size = len(hero_hand)
    for assignment in itertools.product(seats, repeat=len(unseen)):
        if all(assignment.count(seat) == size for seat in seats):
            hands = {seat: {card for card, owner in zip(unseen, assignment) if owner == seat} for seat in seats}
            hands[hero] = set(hero_hand)
            yield hands


def direct_policy_value(hero_hand, hero, deck, opponents, hero_choice) -> float:
    values = [
        direct_layout_value(initial_state_for_hands(hands), hands, hero, opponents, hero_choice)
        for hands in all_layouts_for_hero(hero_hand, hero, deck)
    ]
    return sum(values) / len(values)


def best_response_cases(cards_per_suit: int, count: int, seed: int):
    rng = random.Random(seed)
    deck = frozenset(reduced_deck(cards_per_suit))
    for index in range(count):
        hero = index % 4
        hero_hand = frozenset(deal_random_hands(rng, cards_per_suit=cards_per_suit)[hero])
        names = ["random", "greedy", "heuristic"]
        rng.shuffle(names)
        seats = [seat for seat in range(4) if seat != hero]
        opponents = {seat: opponent_policy(name, deck) for seat, name in zip(seats, names)}
        yield deck, hero, hero_hand, opponents


def test_best_response_policy_values_match_direct_recursion() -> None:
    checked = 0
    for cards_per_suit, count in ((2, 8), (3, 1)):
        for deck, hero, hero_hand, opponents in best_response_cases(cards_per_suit, count, seed=31):
            solver = BestResponseSolver(hero_hand, opponents, hero=hero, deck=deck)
            assert solver.layouts == sum(1 for _ in all_layouts_for_hero(hero_hand, hero, deck))
            for name in ("random", "greedy", "heuristic"):
                policy = hero_policy(name, deck)
                direct = direct_policy_value(
                    hero_hand, hero, deck, opponents,
                    lambda state, hand_mask: policy(state, hero, hand_mask, {}),
                )
                assert abs(solver.value(policy) - direct) < 1e-9
                checked += 1
    assert checked == 27


def test_best_response_is_achievable_without_hidden_cards_and_bounded() -> None:
    information_costs = 0
    beats_heuristic = 0
    # Seed 40 gives a 3-card hand where the heuristic is (slightly) suboptimal.
    for cards_per_suit, count, seed in ((2, 16, 32), (3, 1, 40)):
        for deck, hero, hero_hand, opponents in best_response_cases(cards_per_suit, count, seed):
            solver = BestResponseSolver(hero_hand, opponents, hero=hero, deck=deck)
            best = solver.value()
            # The best-response policy only reads the public history, and achieves its value.
            policy = solver.best_response_policy()
            achieved = direct_policy_value(
                hero_hand, hero, deck, opponents, lambda state, hand_mask: policy(state, hero, hand_mask, {})
            )
            assert abs(achieved - best) < 1e-9
            clairvoyant = direct_policy_value(hero_hand, hero, deck, opponents, lambda state, hand_mask: None)
            assert best <= clairvoyant + 1e-9
            information_costs += best < clairvoyant - 1e-9
            for name in ("random", "greedy", "heuristic", "exact_vote", "posterior_vote"):
                value = solver.value(hero_policy(name, deck))
                assert value <= best + 1e-9
                if name == "heuristic":
                    beats_heuristic += value < best - 1e-9
    assert information_costs >= 1
    assert beats_heuristic >= 1


def test_assumed_table_best_response_matches_direct_recursion_and_bounds() -> None:
    checked = 0
    for cards_per_suit, count, seed in ((2, 8, 33), (3, 1, 40)):
        for deck, hero, hero_hand, opponents in best_response_cases(cards_per_suit, count, seed):
            # Random opponents give every legal action positive probability, so the best
            # response to them is defined at every history any table can produce.
            all_random = {seat: opponent_policy("random", deck) for seat in opponents}
            assumed = BestResponseSolver(hero_hand, all_random, hero=hero, deck=deck)
            assumed.value()
            actual = BestResponseSolver(hero_hand, opponents, hero=hero, deck=deck)
            best = actual.value()

            result = evaluate_assumed_table(assumed, actual)
            moves = assumed.best_moves
            direct = direct_policy_value(
                hero_hand, hero, deck, opponents, lambda state, hand_mask: ((moves[state.history], 1.0),)
            )
            assert abs(result.value - direct) < 1e-9
            assert result.value <= best + 1e-9
            assert result.agree_mass <= result.decision_mass + 1e-9

            same = evaluate_assumed_table(actual, actual)
            assert abs(same.value - best) < 1e-9
            assert abs(same.agree_mass - same.decision_mass) < 1e-9
            checked += 1
    assert checked == 9


def test_best_response_from_position_matches_the_full_game_solver() -> None:
    """Starting mid-game gives the same posterior and values as walking the full solver there."""
    checked = 0
    for cards_per_suit, count, seed in ((2, 6, 35), (3, 2, 36)):
        rng = random.Random(seed)
        for deck, hero, hero_hand, opponents in best_response_cases(cards_per_suit, count, seed):
            full = BestResponseSolver(hero_hand, opponents, hero=hero, deck=deck)
            heuristic = hero_policy("heuristic", deck)
            hands = {seat: set(cards) for seat, cards in enumerate(deal_consistent_hands(rng, hero, hero_hand, deck))}
            starter = next(seat for seat in range(4) if Card(Suit.HEARTS, 7) in hands[seat])
            state, belief = next((s, b) for s, b in full.roots if s.current_player == starter)
            while all(hands.values()):
                player = state.current_player
                mask = card_set_mask(hands[player])
                if player == hero and state.history:
                    position = BestResponseSolver.from_position(state, hero, hands[hero], opponents, deck)
                    (_, posterior), = position.roots
                    mass, total = sum(belief.values()), sum(posterior.values())
                    support = {layout for layout, weight in belief.items() if weight > 0.0}
                    assert set(posterior) == support
                    assert all(abs(posterior[l] / total - belief[l] / mass) < 1e-12 for l in support)
                    assert abs(position.value() - full._node(state, belief, None) / mass) < 1e-9
                    assert abs(position.value(heuristic) - full._node(state, belief, heuristic) / mass) < 1e-9
                    checked += 1
                if player == hero:
                    choices = heuristic(state, player, mask, belief)
                else:
                    choices = opponents[player](state, player, mask)
                cards, weights = zip(*choices)
                card = rng.choices(cards, weights)[0]
                if player != hero:
                    belief = next(child for child_state, child in full._opponent_children(state, belief)
                                  if child_state.history[-1].card == card) if hands[player] != {card} else belief
                if card is not None:
                    hands[player].discard(card)
                state = state.after_play(player, card)
    assert checked >= 6


def test_model_belief_matches_exact_posterior_and_advances_consistently() -> None:
    """Against a table that really plays the model, the belief is the exact posterior's support."""
    from best_response import ModelBelief
    checked = rejected_seen = 0
    for cards_per_suit, count, seed, names in ((3, 3, 41, ("heuristic",) * 3), (3, 3, 42, ("random", "greedy", "heuristic"))):
        rng = random.Random(seed)
        for deck, hero, hero_hand, _ in best_response_cases(cards_per_suit, count, seed):
            seats = [seat for seat in range(4) if seat != hero]
            opponents = {seat: opponent_policy(name, deck) for seat, name in zip(seats, names)}
            model = opponent_policy("heuristic", deck)
            hands = {seat: set(cards) for seat, cards in enumerate(deal_consistent_hands(rng, hero, hero_hand, deck))}
            starter = next(seat for seat in range(4) if Card(Suit.HEARTS, 7) in hands[seat])
            state = GameState(hand_counts=(cards_per_suit,) * 4, current_player=starter)
            belief = None
            while all(hands.values()):
                player = state.current_player
                mask = card_set_mask(hands[player])
                if player == hero and state.history:
                    fresh = ModelBelief(state, hero, mask, model, deck)
                    belief = fresh if belief is None else belief.advanced(state)
                    assert belief.explained == fresh.explained and belief.possible == fresh.possible
                    layouts = belief.layouts(mask, 10**6, random.Random(0))
                    assert len(set(layouts)) == len(layouts)
                    for layout in layouts:
                        assert sum(layout) == card_set_mask(deck) & ~mask & ~state.played_cards_mask()
                        assert all(layout[seat].bit_count() == state.hand_counts[seat] for seat in seats)
                        assert all(layout[seat] & ~belief.possible[seat] == 0 for seat in seats)
                    true_layout = tuple(card_set_mask(hands[seat]) if seat != hero else 0 for seat in range(4))
                    if names == ("heuristic",) * 3:
                        assert set(layouts) == set(position_posterior(state, hero, mask, opponents, deck))
                        assert true_layout in layouts
                    rejected_seen += any(belief.explained[seat] is None for seat in seats)
                    checked += 1
                choices = opponents[player](state, player, mask) if player != hero else \
                    hero_policy("heuristic", deck)(state, player, mask, {})
                cards, weights = zip(*choices)
                card = rng.choices(cards, weights)[0]
                if card is not None:
                    hands[player].discard(card)
                state = state.after_play(player, card)
    assert checked >= 8 and rejected_seen >= 1


def deal_consistent_hands(rng, hero, hero_hand, deck):
    rest = sorted(deck - hero_hand, key=card_index)
    rng.shuffle(rest)
    size = len(hero_hand)
    others = [rest[i * size:(i + 1) * size] for i in range(3)]
    return [hero_hand if seat == hero else others.pop(0) for seat in range(4)]


def test_policy_decision_losses_add_up_to_regret() -> None:
    nonzero = 0
    # Seed 40 gives a 3-card hand where the heuristic is (slightly) suboptimal.
    for cards_per_suit, count, seed in ((2, 6, 34), (3, 1, 40)):
        for deck, hero, hero_hand, opponents in best_response_cases(cards_per_suit, count, seed):
            solver = BestResponseSolver(hero_hand, opponents, hero=hero, deck=deck)
            best = solver.value()
            for name in ("heuristic", "random"):
                policy = hero_policy(name, deck)
                records = solver.policy_decisions(policy)
                regret = best - solver.value(policy)
                assert abs(sum(record.reach * record.loss() for record in records) - regret) < 1e-9
                assert all(len(record.legal) > 1 and record.loss() >= -1e-12 for record in records)
                nonzero += regret > 1e-9
    assert nonzero >= 2
