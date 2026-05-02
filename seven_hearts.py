from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Suit(str, Enum):
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"


RANK_NAMES = {
    1: "A",
    11: "J",
    12: "Q",
    13: "K",
}


@dataclass(frozen=True, order=True)
class Card:
    suit: Suit
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1 or self.rank > 13:
            raise ValueError(f"rank must be between 1 and 13, got {self.rank}")

    @classmethod
    def parse(cls, text: str) -> "Card":
        text = text.strip().upper()
        if len(text) < 2:
            raise ValueError(f"invalid card: {text!r}")

        suit = Suit(text[-1])
        rank_text = text[:-1]
        rank = {"A": 1, "J": 11, "Q": 12, "K": 13}.get(rank_text)
        if rank is None:
            rank = int(rank_text)
        return cls(suit=suit, rank=rank)

    def label(self) -> str:
        return f"{RANK_NAMES.get(self.rank, str(self.rank))}{self.suit.value}"

    def __str__(self) -> str:
        return self.label()


def full_deck() -> set[Card]:
    return {Card(suit, rank) for suit in Suit for rank in range(1, 14)}


@dataclass(frozen=True)
class SuitRun:
    low: int | None = None
    high: int | None = None

    @property
    def is_open(self) -> bool:
        return self.low is not None and self.high is not None

    def legal_cards(self, suit: Suit) -> set[Card]:
        if not self.is_open:
            return {Card(suit, 7)}

        cards: set[Card] = set()
        assert self.low is not None and self.high is not None
        if self.low > 1:
            cards.add(Card(suit, self.low - 1))
        if self.high < 13:
            cards.add(Card(suit, self.high + 1))
        return cards

    def play(self, rank: int) -> "SuitRun":
        if not self.is_open:
            if rank != 7:
                raise ValueError("closed suits can only be opened with a 7")
            return SuitRun(low=7, high=7)

        assert self.low is not None and self.high is not None
        if rank == self.low - 1:
            return SuitRun(low=rank, high=self.high)
        if rank == self.high + 1:
            return SuitRun(low=self.low, high=rank)
        raise ValueError(f"rank {rank} is not adjacent to run {self.low}-{self.high}")


@dataclass(frozen=True)
class MoveEvent:
    player: int
    card: Card | None

    @property
    def is_pass(self) -> bool:
        return self.card is None


@dataclass(frozen=True)
class GameState:
    table: dict[Suit, SuitRun] = field(default_factory=lambda: {s: SuitRun() for s in Suit})
    hand_counts: tuple[int | None, int | None, int | None, int | None] | None = None
    current_player: int = 0
    history: tuple[MoveEvent, ...] = ()

    def played_cards(self) -> set[Card]:
        cards: set[Card] = set()
        for suit, run in self.table.items():
            if not run.is_open:
                continue
            assert run.low is not None and run.high is not None
            cards.update(Card(suit, rank) for rank in range(run.low, run.high + 1))
        return cards

    def public_legal_cards(self) -> set[Card]:
        legal: set[Card] = set()
        for suit, run in self.table.items():
            legal.update(run.legal_cards(suit))
        return legal - self.played_cards()

    def legal_moves(self, hand: Iterable[Card]) -> list[Card]:
        hand_set = set(hand)
        return sorted(hand_set & self.public_legal_cards())

    def after_play(self, player: int, card: Card | None) -> "GameState":
        if card is None:
            return GameState(
                table=self.table,
                hand_counts=self.hand_counts,
                current_player=(player + 1) % 4,
                history=self.history + (MoveEvent(player, None),),
            )

        next_table = dict(self.table)
        next_table[card.suit] = next_table[card.suit].play(card.rank)
        counts = None
        if self.hand_counts is not None:
            counts = list(self.hand_counts)
            if counts[player] is not None:
                counts[player] -= 1
        return GameState(
            table=next_table,
            hand_counts=tuple(counts) if counts is not None else None,  # type: ignore[arg-type]
            current_player=(player + 1) % 4,
            history=self.history + (MoveEvent(player, card),),
        )


@dataclass(frozen=True)
class PlayerKnowledge:
    player: int
    hand: frozenset[Card]

    def unseen_cards(self, state: GameState) -> set[Card]:
        return full_deck() - set(self.hand) - state.played_cards()


@dataclass(frozen=True)
class OpponentModel:
    possible_cards: dict[int, set[Card]]

    def probability_any_opponent_has(self, card: Card) -> float:
        probs = [1.0 if card in cards else 0.0 for cards in self.possible_cards.values()]
        if not probs:
            return 0.0
        return sum(probs) / len(probs)


def build_opponent_model(state: GameState, knowledge: PlayerKnowledge) -> OpponentModel:
    unseen = knowledge.unseen_cards(state)
    opponents = [p for p in range(4) if p != knowledge.player]
    possible = {player: set(unseen) for player in opponents}

    simulated = GameState()
    for event in state.history:
        if event.player == knowledge.player:
            if event.card is not None:
                simulated = simulated.after_play(event.player, event.card)
            continue

        if event.is_pass:
            possible[event.player] -= simulated.public_legal_cards()
            simulated = simulated.after_play(event.player, None)
        else:
            assert event.card is not None
            for player in opponents:
                if player != event.player:
                    possible[player].discard(event.card)
            simulated = simulated.after_play(event.player, event.card)

    return OpponentModel(possible_cards=possible)


@dataclass(frozen=True)
class MoveScore:
    card: Card
    score: float
    reasons: tuple[str, ...]


def score_move(state: GameState, knowledge: PlayerKnowledge, card: Card) -> MoveScore:
    hand_after = set(knowledge.hand) - {card}
    next_state = state.after_play(knowledge.player, card)
    model = build_opponent_model(state, knowledge)

    score = 10.0
    reasons = ["played one card: +10.0"]

    before_legal = set(state.legal_moves(knowledge.hand))
    after_legal = set(next_state.legal_moves(hand_after))
    newly_self_playable = after_legal - before_legal
    self_unlock = 3.0 * len(newly_self_playable)
    if self_unlock:
        score += self_unlock
        reasons.append(f"unlocks own cards {labels(newly_self_playable)}: +{self_unlock:.1f}")

    opened_for_table = next_state.public_legal_cards() - state.public_legal_cards()
    opened_for_others = opened_for_table - hand_after
    opponent_risk = sum(model.probability_any_opponent_has(c) for c in opened_for_others) * 2.0
    if opponent_risk:
        score -= opponent_risk
        reasons.append(f"may unlock opponents {labels(opened_for_others)}: -{opponent_risk:.1f}")

    control_value = retained_control_value(state, knowledge, card)
    if control_value:
        score -= control_value
        reasons.append(f"gives up blocking value: -{control_value:.1f}")

    urgency = endgame_urgency(state, knowledge.player)
    if urgency:
        score += urgency
        reasons.append(f"endgame urgency: +{urgency:.1f}")

    return MoveScore(card=card, score=score, reasons=tuple(reasons))


def retained_control_value(state: GameState, knowledge: PlayerKnowledge, card: Card) -> float:
    run = state.table[card.suit]
    if not run.is_open and card.rank == 7:
        same_suit = {c for c in knowledge.hand if c.suit == card.suit and c != card}
        if not same_suit:
            return 5.0
        distance_pressure = sum(abs(c.rank - 7) for c in same_suit)
        return max(0.0, 4.0 - distance_pressure * 0.35)

    if run.is_open:
        assert run.low is not None and run.high is not None
        if card.rank in {run.low - 1, run.high + 1}:
            outward_cards = cards_blocked_behind(card, knowledge.hand)
            if not outward_cards:
                return 2.0
    return 0.0


def cards_blocked_behind(card: Card, hand: Iterable[Card]) -> set[Card]:
    if card.rank < 7:
        return {c for c in hand if c.suit == card.suit and c.rank < card.rank}
    if card.rank > 7:
        return {c for c in hand if c.suit == card.suit and c.rank > card.rank}
    return set()


def endgame_urgency(state: GameState, player: int) -> float:
    if state.hand_counts is None or state.hand_counts[player] is None:
        return 0.0

    my_count = state.hand_counts[player]
    known_opponent_counts = [
        count for p, count in enumerate(state.hand_counts) if p != player and count is not None
    ]
    if not known_opponent_counts:
        return 4.0 if my_count <= 3 else 0.0

    lowest_opponent = min(known_opponent_counts)
    if my_count <= 3:
        return 4.0
    if lowest_opponent <= 3:
        return 2.0
    return 0.0


def recommend_move(state: GameState, knowledge: PlayerKnowledge) -> MoveScore | None:
    legal = state.legal_moves(knowledge.hand)
    if not legal:
        return None
    return max((score_move(state, knowledge, card) for card in legal), key=lambda s: s.score)


def labels(cards: Iterable[Card]) -> str:
    ordered = sorted(cards)
    if not ordered:
        return "[]"
    return "[" + ", ".join(card.label() for card in ordered) + "]"
