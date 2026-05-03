from seven_hearts import (
    Card,
    GameState,
    MoveEvent,
    PlayerKnowledge,
    Suit,
    SuitRun,
    build_opponent_model,
    endgame_urgency,
    future_chain_impact,
    recommend_move,
    score_move,
    time_to_playable,
)


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
