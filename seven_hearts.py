from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping


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
        if not self.played_cards():
            return {Card(Suit.HEARTS, 7)}

        legal: set[Card] = set()
        for suit, run in self.table.items():
            legal.update(run.legal_cards(suit))
        return legal - self.played_cards()

    def legal_moves(self, hand: Iterable[Card]) -> list[Card]:
        hand_set = set(hand)
        return sorted(hand_set & self.public_legal_cards())

    def validate_turn(self, hand: Iterable[Card], card: Card | None) -> None:
        legal = set(self.legal_moves(hand))
        if card is None:
            if legal:
                raise ValueError(f"cannot pass with legal moves available: {labels(legal)}")
            return

        if card not in set(hand):
            raise ValueError(f"cannot play {card}: card is not in hand")
        if card not in legal:
            raise ValueError(f"cannot play {card}: legal moves are {labels(legal)}")

    def after_play(
        self,
        player: int,
        card: Card | None,
        hand: Iterable[Card] | None = None,
        validate: bool = False,
    ) -> "GameState":
        if validate or hand is not None:
            if hand is None:
                raise ValueError("hand is required when validate=True")
            self.validate_turn(hand, card)

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
    hand_counts: tuple[int | None, int | None, int | None, int | None] | None = None

    def holder_probability(self, player: int, card: Card) -> float:
        weights = self._holder_weights(card)
        total = sum(weights.values())
        if total == 0.0:
            return 0.0
        return weights.get(player, 0.0) / total

    def _holder_weights(self, card: Card) -> dict[int, float]:
        weights: dict[int, float] = {}
        for player, possible in self.possible_cards.items():
            if card not in possible:
                continue
            count = self.hand_counts[player] if self.hand_counts is not None else None
            if count is None:
                weights[player] = 1.0
            else:
                weights[player] = max(count, 0) / max(len(possible), 1)
        return weights

    def consistency_errors(self) -> tuple[str, ...]:
        if self.hand_counts is None:
            return ()

        errors: list[str] = []
        for player, possible in self.possible_cards.items():
            count = self.hand_counts[player]
            if count is not None and count > len(possible):
                errors.append(
                    f"player {player} has count {count}, but only {len(possible)} possible cards"
                )
        return tuple(errors)


def build_opponent_model(state: GameState, knowledge: PlayerKnowledge) -> OpponentModel:
    unseen = knowledge.unseen_cards(state)
    opponents = [p for p in range(4) if p != knowledge.player]
    possible = {player: set(unseen) for player in opponents}

    simulated = GameState()
    for event in state.history:
        if event.is_pass:
            if event.player in possible:
                possible[event.player] -= simulated.public_legal_cards()
            simulated = simulated.after_play(event.player, None)
        else:
            assert event.card is not None
            for player in opponents:
                if player != event.player:
                    possible[player].discard(event.card)
            simulated = simulated.after_play(event.player, event.card)

    return OpponentModel(possible_cards=possible, hand_counts=state.hand_counts)


@dataclass(frozen=True)
class MoveScore:
    card: Card
    score: float
    components: Mapping[str, float]
    reasons: tuple[str, ...]


def score_move(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    model: OpponentModel | None = None,
) -> MoveScore:
    hand_after = set(knowledge.hand) - {card}
    next_state = state.after_play(knowledge.player, card)
    model = model or build_opponent_model(state, knowledge)

    score = 0.0
    components: dict[str, float] = {}
    reasons: list[str] = []

    before_legal = set(state.legal_moves(knowledge.hand))
    after_legal = set(next_state.legal_moves(hand_after))
    newly_self_playable = after_legal - before_legal
    self_unlock = 3.0 * len(newly_self_playable)
    if self_unlock:
        score += self_unlock
        components["self_unlock"] = self_unlock
        reasons.append(f"unlocks own cards {labels(newly_self_playable)}: +{self_unlock:.1f}")

    opened_for_table = next_state.public_legal_cards() - state.public_legal_cards()
    opened_for_others = opened_for_table - hand_after
    opponent_risk = opponent_unlock_risk(state, knowledge, model, opened_for_others)
    if opponent_risk:
        score -= opponent_risk
        components["opponent_unlock_risk"] = -opponent_risk
        reasons.append(f"may unlock opponents {labels(opened_for_others)}: -{opponent_risk:.1f}")

    chain_impact = future_chain_impact(state, knowledge, card, model)
    if chain_impact:
        score += chain_impact
        components["future_chain_impact"] = chain_impact
        sign = "+" if chain_impact > 0 else ""
        reasons.append(f"future chain impact: {sign}{chain_impact:.1f}")

    return MoveScore(
        card=card,
        score=score,
        components=MappingProxyType(dict(components)),
        reasons=tuple(reasons),
    )


def opponent_unlock_risk(
    state: GameState,
    knowledge: PlayerKnowledge,
    model: OpponentModel,
    opened_cards: set[Card],
) -> float:
    risk = 0.0
    for opponent in model.possible_cards:
        playable_mass = sum(model.holder_probability(opponent, card) for card in opened_cards)
        capped_mass = min(1.0, playable_mass)
        risk += capped_mass * turn_order_weight(knowledge.player, opponent)
    return risk * 2.0


def turn_order_weight(player: int, opponent: int) -> float:
    turns_until = (opponent - player) % 4
    if turns_until == 0:
        return 0.0
    return 1.0 / turns_until


def future_chain_impact(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    model: OpponentModel | None = None,
) -> float:
    """Estimate the long-term suit control gained or released by playing a card."""
    hand_after = set(knowledge.hand) - {card}
    unseen = knowledge.unseen_cards(state)
    model = model or build_opponent_model(state, knowledge)

    impact = 0.0
    for side in released_sides(card):
        future_cards = {Card(card.suit, rank) for rank in side}
        own_future = future_cards & hand_after
        unseen_future = future_cards & unseen

        # Releasing a long chain mostly benefits the table unless we own part of that runway.
        impact -= 0.35 * len(unseen_future)
        impact += 0.75 * len(own_future)

        tail_card = owned_tail_card(card.suit, side, own_future)
        if tail_card is not None:
            impact += max(0.0, 2.0 - time_to_playable(state, tail_card) * 0.25)

        if side and owns_gate_card(card.suit, side, hand_after):
            impact += 1.0

        opponent_runway_mass = sum(
            model.holder_probability(opponent, future_card) * turn_order_weight(knowledge.player, opponent)
            for future_card in future_cards
            for opponent in model.possible_cards
        )
        impact -= 0.08 * opponent_runway_mass

    return impact


def released_sides(card: Card) -> list[range]:
    if card.rank == 7:
        return [range(1, 7), range(8, 14)]
    if card.rank < 7:
        return [range(1, card.rank)]
    return [range(card.rank + 1, 14)]


def owned_tail_card(suit: Suit, side: range, cards: set[Card]) -> Card | None:
    if not side:
        return None
    tail_rank = 1 if max(side) < 7 else 13
    tail_card = Card(suit, tail_rank)
    return tail_card if tail_card in cards else None


def owns_gate_card(suit: Suit, side: range, cards: set[Card]) -> bool:
    if not side:
        return False
    gate_rank = max(side) if max(side) < 7 else min(side)
    return Card(suit, gate_rank) in cards


def time_to_playable(state: GameState, card: Card) -> int:
    run = state.table[card.suit]
    if not run.is_open:
        return abs(card.rank - 7)
    assert run.low is not None and run.high is not None
    if run.low <= card.rank <= run.high:
        return 0
    if card.rank < run.low:
        return run.low - card.rank
    return card.rank - run.high


def endgame_urgency(state: GameState, player: int) -> float:
    if state.hand_counts is None or state.hand_counts[player] is None:
        return 0.0

    my_count = state.hand_counts[player]
    known_opponent_counts = [
        count for p, count in enumerate(state.hand_counts) if p != player and count is not None
    ]
    if not known_opponent_counts:
        return max(0.0, 5.0 - my_count)

    lowest_opponent = min(known_opponent_counts)
    my_urgency = max(0.0, 5.0 - my_count)
    opponent_urgency = max(0.0, 4.0 - lowest_opponent) * 0.75
    race_pressure = 0.0
    if min(my_count, lowest_opponent) <= 5:
        race_pressure = max(0.0, my_count - lowest_opponent) * 0.25
    return my_urgency + opponent_urgency + race_pressure


def recommend_move(state: GameState, knowledge: PlayerKnowledge) -> MoveScore | None:
    legal = state.legal_moves(knowledge.hand)
    if not legal:
        return None
    model = build_opponent_model(state, knowledge)
    return max((score_move(state, knowledge, card, model) for card in legal), key=lambda s: s.score)


def labels(cards: Iterable[Card]) -> str:
    ordered = sorted(cards)
    if not ordered:
        return "[]"
    return "[" + ", ".join(card.label() for card in ordered) + "]"
