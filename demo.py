from seven_hearts import Card, GameState, PlayerKnowledge, Suit, SuitRun, labels, recommend_move, score_move


def cards(text: str) -> frozenset[Card]:
    return frozenset(Card.parse(part) for part in text.split())


def main() -> None:
    state = GameState(
        table={
            Suit.CLUBS: SuitRun(),
            Suit.DIAMONDS: SuitRun(),
            Suit.HEARTS: SuitRun(low=6, high=8),
            Suit.SPADES: SuitRun(),
        },
        hand_counts=(10, 11, 9, 10),
        current_player=0,
    )
    knowledge = PlayerKnowledge(
        player=0,
        hand=cards("7S AS 5H 9H 7D KC 3C 10D 2S 8S"),
    )

    legal = state.legal_moves(knowledge.hand)
    print(f"Legal moves: {labels(legal)}")

    for card in legal:
        result = score_move(state, knowledge, card)
        print(f"\n{result.card}: {result.score:.1f}")
        for reason in result.reasons:
            print(f"  - {reason}")

    best = recommend_move(state, knowledge)
    if best is None:
        print("\nRecommendation: pass")
    else:
        print(f"\nRecommendation: play {best.card} ({best.score:.1f})")


if __name__ == "__main__":
    main()

