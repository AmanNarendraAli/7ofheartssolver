from seven_hearts import Card, GameState, MoveEvent, PlayerKnowledge, Suit, SuitRun, build_opponent_model, recommend_move


def hand(text: str) -> frozenset[Card]:
    return frozenset(Card.parse(part) for part in text.split())


def test_closed_suit_only_allows_seven() -> None:
    state = GameState()

    assert state.public_legal_cards() == {
        Card(Suit.CLUBS, 7),
        Card(Suit.DIAMONDS, 7),
        Card(Suit.HEARTS, 7),
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
