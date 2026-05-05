from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from math import isnan, nan, sqrt
from time import perf_counter
from types import MappingProxyType
from typing import Callable, Iterable, Mapping


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


SUIT_ORDER = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
NEUTRAL_VALUE = (0.0, 0.0, 0.0, 0.0)


def canonical_table(table: Mapping[Suit, SuitRun] | None = None) -> tuple[SuitRun, SuitRun, SuitRun, SuitRun]:
    table = table or {suit: SuitRun() for suit in Suit}
    return tuple(table.get(suit, SuitRun()) for suit in SUIT_ORDER)  # type: ignore[return-value]


def table_mapping(table_key: tuple[SuitRun, SuitRun, SuitRun, SuitRun]) -> dict[Suit, SuitRun]:
    return {suit: table_key[index] for index, suit in enumerate(SUIT_ORDER)}


def trailing_pass_count(history: Iterable[MoveEvent]) -> int:
    count = 0
    for event in reversed(tuple(history)):
        if not event.is_pass:
            break
        count += 1
    return count


def card_sort_key(card: Card) -> tuple[str, int]:
    return (card.suit.value, card.rank)


@dataclass(frozen=True)
class FullInformationState:
    hands: tuple[frozenset[Card], frozenset[Card], frozenset[Card], frozenset[Card]]
    table: tuple[SuitRun, SuitRun, SuitRun, SuitRun] = field(default_factory=canonical_table)
    current_player: int = 0
    winner: int | None = None
    consecutive_passes: int = 0

    def __post_init__(self) -> None:
        if len(self.hands) != 4:
            raise ValueError("full-information state must contain exactly four hands")
        if len(self.table) != 4:
            raise ValueError("full-information table must contain exactly four suit runs")
        if self.current_player not in range(4):
            raise ValueError(f"current_player must be 0..3, got {self.current_player}")
        if self.winner is not None and self.winner not in range(4):
            raise ValueError(f"winner must be 0..3 or None, got {self.winner}")
        if self.consecutive_passes < 0:
            raise ValueError("consecutive_passes cannot be negative")

        seen: set[Card] = set()
        for player, hand in enumerate(self.hands):
            overlap = seen & hand
            if overlap:
                raise ValueError(f"duplicate cards in hands for player {player}: {labels(overlap)}")
            seen.update(hand)

        table_cards = self.played_cards()
        overlap = seen & table_cards
        if overlap:
            raise ValueError(f"cards cannot be both in hand and on table: {labels(overlap)}")

    @classmethod
    def from_hands(
        cls,
        hands: Mapping[int, Iterable[Card]],
        state: GameState | None = None,
        consecutive_passes: int | None = None,
        winner: int | None = None,
    ) -> "FullInformationState":
        state = state or GameState()
        return cls(
            hands=tuple(frozenset(hands.get(player, ())) for player in range(4)),  # type: ignore[arg-type]
            table=canonical_table(state.table),
            current_player=state.current_player,
            winner=winner if winner is not None else first_empty_player(hands),
            consecutive_passes=trailing_pass_count(state.history) if consecutive_passes is None else consecutive_passes,
        )

    def table_dict(self) -> dict[Suit, SuitRun]:
        return table_mapping(self.table)

    def as_public_state(self) -> GameState:
        return GameState(
            table=self.table_dict(),
            hand_counts=tuple(len(hand) for hand in self.hands),  # type: ignore[arg-type]
            current_player=self.current_player,
        )

    def played_cards(self) -> set[Card]:
        return GameState(table=self.table_dict()).played_cards()

    def public_legal_cards(self) -> set[Card]:
        return GameState(table=self.table_dict()).public_legal_cards()

    def legal_moves(self) -> tuple[Card, ...]:
        if self.winner is not None:
            return ()
        legal = self.hands[self.current_player] & self.public_legal_cards()
        return tuple(sorted(legal))

    def after_action(self, card: Card | None) -> "FullInformationState":
        legal = set(self.legal_moves())
        player = self.current_player
        if card is None:
            if legal:
                raise ValueError(f"cannot pass with legal moves available: {labels(legal)}")
            return FullInformationState(
                hands=self.hands,
                table=self.table,
                current_player=(player + 1) % 4,
                winner=self.winner,
                consecutive_passes=self.consecutive_passes + 1,
            )

        if card not in legal:
            raise ValueError(f"cannot play {card}: legal moves are {labels(legal)}")

        next_hands = list(self.hands)
        next_hands[player] = frozenset(next_hands[player] - {card})
        next_table = list(self.table)
        suit_index = SUIT_ORDER.index(card.suit)
        next_table[suit_index] = next_table[suit_index].play(card.rank)
        winner = player if not next_hands[player] else None
        return FullInformationState(
            hands=tuple(next_hands),  # type: ignore[arg-type]
            table=tuple(next_table),  # type: ignore[arg-type]
            current_player=(player + 1) % 4,
            winner=winner,
            consecutive_passes=0,
        )

    def assert_card_conservation(self, deck: Iterable[Card] | None = None) -> None:
        expected = set(deck if deck is not None else full_deck())
        actual = set().union(*self.hands, self.played_cards())
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            raise ValueError(f"card conservation failed; missing {labels(missing)}, extra {labels(extra)}")


@dataclass(frozen=True)
class ExactSolverResult:
    value: tuple[float, float, float, float]
    best_moves: tuple[Card, ...]
    chosen_move: Card | None
    move_values: Mapping[Card, tuple[float, float, float, float]]
    states_evaluated: int
    cache_hits: int
    terminal_states: int
    deadlock_states: int
    policy_name: str = "full_information_rational_first_out"
    tie_break_description: str = (
        "maximize acting player's first-out value; then minimize next player's "
        "value; then minimize strongest opponent value; then deterministic card order"
    )
    exhaustive: bool = True


@dataclass(frozen=True)
class ExactSearchBenchmark:
    name: str
    value: tuple[float, float, float, float]
    chosen_move: Card | None
    states_evaluated: int
    cache_hits: int
    terminal_states: int
    deadlock_states: int
    elapsed_seconds: float

    @property
    def states_per_second(self) -> float:
        if self.elapsed_seconds == 0.0:
            return float("inf")
        return self.states_evaluated / self.elapsed_seconds


def terminal_value(winner: int) -> tuple[float, float, float, float]:
    values = [0.0, 0.0, 0.0, 0.0]
    values[winner] = 1.0
    return tuple(values)  # type: ignore[return-value]


def solve_full_information(state: FullInformationState) -> ExactSolverResult:
    cache: dict[FullInformationState, tuple[float, float, float, float]] = {}
    stats = {
        "states_evaluated": 0,
        "cache_hits": 0,
        "terminal_states": 0,
        "deadlock_states": 0,
    }

    def solve_value(current: FullInformationState) -> tuple[float, float, float, float]:
        return solve_full_information_value(current, cache, stats)

    root_value = solve_value(state)
    root_legal = state.legal_moves()
    move_values = {
        card: solve_value(state.after_action(card))
        for card in root_legal
    }
    if move_values:
        primary_best = max(value[state.current_player] for value in move_values.values())
        best_moves = tuple(
            card for card, value in sorted(move_values.items()) if value[state.current_player] == primary_best
        )
        chosen_move = choose_rational_move(state.current_player, move_values)
    else:
        best_moves = ()
        chosen_move = None

    return ExactSolverResult(
        value=root_value,
        best_moves=best_moves,
        chosen_move=chosen_move,
        move_values=MappingProxyType(dict(move_values)),
        states_evaluated=stats["states_evaluated"],
        cache_hits=stats["cache_hits"],
        terminal_states=stats["terminal_states"],
        deadlock_states=stats["deadlock_states"],
    )


def benchmark_full_information_search(name: str, state: FullInformationState) -> ExactSearchBenchmark:
    started = perf_counter()
    result = solve_full_information(state)
    elapsed = perf_counter() - started
    return ExactSearchBenchmark(
        name=name,
        value=result.value,
        chosen_move=result.chosen_move,
        states_evaluated=result.states_evaluated,
        cache_hits=result.cache_hits,
        terminal_states=result.terminal_states,
        deadlock_states=result.deadlock_states,
        elapsed_seconds=elapsed,
    )


def solve_full_information_value(
    current: FullInformationState,
    cache: dict[FullInformationState, tuple[float, float, float, float]],
    stats: dict[str, int],
) -> tuple[float, float, float, float]:
    cached = cache.get(current)
    if cached is not None:
        stats["cache_hits"] += 1
        return cached

    stats["states_evaluated"] += 1
    if current.winner is not None:
        stats["terminal_states"] += 1
        value = terminal_value(current.winner)
        cache[current] = value
        return value

    legal = current.legal_moves()
    if not legal:
        canonical = canonicalize_forced_pass_chain(current)
        if canonical != current:
            value = solve_full_information_value(canonical, cache, stats)
            cache[current] = value
            return value
        if canonical.consecutive_passes >= 4:
            stats["deadlock_states"] += 1
            cache[current] = NEUTRAL_VALUE
            return NEUTRAL_VALUE

    child_values = {
        card: solve_full_information_value(current.after_action(card), cache, stats)
        for card in legal
    }
    chosen = choose_rational_move(current.current_player, child_values)
    value = child_values[chosen]
    cache[current] = value
    return value


def solve_full_information_against_policy(
    state: FullInformationState,
    policy: Callable[[FullInformationState], Card],
    policy_name: str = "fixed_policy",
) -> ExactSolverResult:
    cache: dict[FullInformationState, tuple[float, float, float, float]] = {}
    stats = {
        "states_evaluated": 0,
        "cache_hits": 0,
        "terminal_states": 0,
        "deadlock_states": 0,
    }

    def solve_policy_value(current: FullInformationState) -> tuple[float, float, float, float]:
        cached = cache.get(current)
        if cached is not None:
            stats["cache_hits"] += 1
            return cached

        stats["states_evaluated"] += 1
        if current.winner is not None:
            stats["terminal_states"] += 1
            value = terminal_value(current.winner)
            cache[current] = value
            return value

        legal = current.legal_moves()
        if not legal:
            canonical = canonicalize_forced_pass_chain(current)
            if canonical != current:
                value = solve_policy_value(canonical)
                cache[current] = value
                return value
            if canonical.consecutive_passes >= 4:
                stats["deadlock_states"] += 1
                cache[current] = NEUTRAL_VALUE
                return NEUTRAL_VALUE

        chosen = policy(current)
        if chosen not in legal:
            raise ValueError(f"fixed policy chose illegal move {chosen}; legal moves are {labels(legal)}")
        value = solve_policy_value(current.after_action(chosen))
        cache[current] = value
        return value

    root_legal = state.legal_moves()
    move_values = {
        card: solve_policy_value(state.after_action(card))
        for card in root_legal
    }
    if move_values:
        primary_best = max(value[state.current_player] for value in move_values.values())
        best_moves = tuple(
            card for card, value in sorted(move_values.items()) if value[state.current_player] == primary_best
        )
        chosen_move = sorted(best_moves)[0]
        root_value = move_values[chosen_move]
    else:
        best_moves = ()
        chosen_move = None
        root_value = solve_policy_value(state)

    return ExactSolverResult(
        value=root_value,
        best_moves=best_moves,
        chosen_move=chosen_move,
        move_values=MappingProxyType(dict(move_values)),
        states_evaluated=stats["states_evaluated"],
        cache_hits=stats["cache_hits"],
        terminal_states=stats["terminal_states"],
        deadlock_states=stats["deadlock_states"],
        policy_name=policy_name,
        tie_break_description="root chooses best move against fixed future policy; tied root moves use deterministic card order",
    )


def canonicalize_forced_pass_chain(state: FullInformationState) -> FullInformationState:
    current = state
    while current.winner is None and not current.legal_moves() and current.consecutive_passes < 4:
        current = current.after_action(None)
    return current


def lowest_legal_card_policy(state: FullInformationState) -> Card:
    legal = state.legal_moves()
    if not legal:
        raise ValueError("lowest_legal_card_policy called with no legal moves")
    return legal[0]


def highest_legal_card_policy(state: FullInformationState) -> Card:
    legal = state.legal_moves()
    if not legal:
        raise ValueError("highest_legal_card_policy called with no legal moves")
    return legal[-1]


def choose_rational_move(
    player: int,
    move_values: Mapping[Card, tuple[float, float, float, float]],
) -> Card:
    tied = dict(move_values)
    for score in (
        lambda value: value[player],
        lambda value: -value[(player + 1) % 4],
        lambda value: -max(value[opponent] for opponent in range(4) if opponent != player),
    ):
        best = max(score(value) for value in tied.values())
        tied = {card: value for card, value in tied.items() if score(value) == best}
        if len(tied) == 1:
            return next(iter(tied))
    return sorted(tied)[0]


def choose_expected_value_move(move_values: Mapping[Card, float]) -> Card:
    best = max(move_values.values())
    tied = [card for card, value in move_values.items() if value == best]
    return sorted(tied)[0]


def choose_expected_value_vector_move(
    player: int,
    move_values: Mapping[Card, tuple[float, float, float, float]],
) -> Card:
    return choose_rational_move(player, move_values)


def format_value(value: tuple[float, float, float, float]) -> str:
    return "(" + ", ".join(f"P{player}={value[player]:.3f}" for player in range(4)) + ")"


def format_exact_solver_certificate(
    state: FullInformationState,
    result: ExactSolverResult,
) -> str:
    lines = [
        "Exact full-information certificate",
        f"policy: {result.policy_name}",
        f"current_player: P{state.current_player}",
        f"hand_counts: {tuple(len(hand) for hand in state.hands)}",
        f"legal_moves: {labels(state.legal_moves())}",
        f"value: {format_value(result.value)}",
        f"best_moves: {labels(result.best_moves)}",
        f"chosen_move: {result.chosen_move.label() if result.chosen_move else 'pass'}",
        f"exhaustive: {result.exhaustive}",
        (
            "search: "
            f"states={result.states_evaluated}, cache_hits={result.cache_hits}, "
            f"terminals={result.terminal_states}, deadlocks={result.deadlock_states}"
        ),
        f"tie_break: {result.tie_break_description}",
    ]
    if result.move_values:
        lines.append("move_values:")
        for card, value in sorted(result.move_values.items()):
            marker = "*" if card == result.chosen_move else " "
            lines.append(f"  {marker} {card.label()}: {format_value(value)}")
    return "\n".join(lines)


@dataclass(frozen=True)
class PlayerKnowledge:
    player: int
    hand: frozenset[Card]
    deck: frozenset[Card] | None = None

    def unseen_cards(self, state: GameState) -> set[Card]:
        deck = set(self.deck) if self.deck is not None else full_deck()
        return deck - set(self.hand) - state.played_cards()


@dataclass(frozen=True)
class HiddenDeal:
    hands: Mapping[int, frozenset[Card]]

    def hand(self, player: int) -> frozenset[Card]:
        return self.hands.get(player, frozenset())


@dataclass(frozen=True)
class HiddenDealEnumeration:
    deals: tuple[HiddenDeal, ...]
    exhaustive: bool
    deal_count: int


@dataclass(frozen=True)
class ExactImperfectInformationMoveScore:
    card: Card
    expected_value: float
    expected_value_vector: tuple[float, float, float, float]
    hidden_deals: int
    exhaustive: bool
    outcome_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    neutral_outcomes: int = 0
    value_standard_error: float = 0.0


@dataclass(frozen=True)
class ExactImperfectInformationResult:
    chosen_move: Card | None
    best_moves: tuple[Card, ...]
    move_scores: tuple[ExactImperfectInformationMoveScore, ...]
    hidden_deals: int
    exhaustive: bool
    states_evaluated: int = 0
    cache_hits: int = 0
    terminal_states: int = 0
    deadlock_states: int = 0
    policy_name: str = "exact_hidden_deal_expectation_with_full_information_rational_continuation"
    continuation_model: str = "opponents play full-information rationally from each materialized hidden deal"


@dataclass(frozen=True)
class RolloutResult:
    winner: int | None
    turns_played: int
    final_hand_counts: tuple[int, int, int, int]
    timed_out: bool = False


@dataclass(frozen=True)
class MonteCarloMoveScore:
    card: Card
    score: float
    win_rate: float
    average_finish_margin: float
    samples: int
    average_turns: float = 0.0
    timeout_rate: float = 0.0
    win_rate_standard_error: float = 0.0


@dataclass(frozen=True)
class CompleteGameEstimate:
    games: int
    win_rates: tuple[float, float, float, float]
    win_rate_standard_errors: tuple[float, float, float, float]
    average_finish_margins: tuple[float, float, float, float]
    average_turns: float
    timeout_rate: float


@dataclass(frozen=True)
class StrategyWeights:
    self_unlock: float = 3.0
    opponent_unlock_risk: float = 2.0
    future_chain_unseen_penalty: float = 0.35
    future_chain_own_credit: float = 0.75
    tail_base_credit: float = 2.6
    tail_distance_penalty: float = 0.16
    gate_card_credit: float = 1.0
    opponent_runway_penalty: float = 0.08
    race_pressure_credit: float = 0.25
    response_penalty: float = 0.35
    monte_carlo_win_rate_weight: float = 100.0
    monte_carlo_timeout_penalty: float = 10.0


DEFAULT_WEIGHTS = StrategyWeights()


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    weights: StrategyWeights = DEFAULT_WEIGHTS


@dataclass(frozen=True)
class StrategySelfPlayEstimate:
    candidate: StrategyCandidate
    games: int
    win_rate: float
    win_rate_standard_error: float
    average_finish_margin: float
    average_turns: float
    timeout_rate: float
    score: float


@dataclass(frozen=True)
class OpponentModel:
    possible_cards: dict[int, set[Card]]
    hand_counts: tuple[int | None, int | None, int | None, int | None] | None = None
    holder_marginals: Mapping[int, Mapping[Card, float]] | None = None
    inference_mode: str = "weighted_possible_holders"

    def holder_probability(self, player: int, card: Card) -> float:
        if self.holder_marginals is not None:
            return self.holder_marginals.get(player, {}).get(card, 0.0)

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
        if self.holder_marginals is None and any(
            count is not None for player, count in enumerate(self.hand_counts) if player in self.possible_cards
        ):
            errors.append("known hand counts are inconsistent with possible hidden-card assignments")
        return tuple(errors)


def build_opponent_model(state: GameState, knowledge: PlayerKnowledge) -> OpponentModel:
    unseen = knowledge.unseen_cards(state)
    possible = possible_opponent_cards(state, knowledge)

    marginals = exact_holder_marginals(possible, state.hand_counts, unseen)
    return OpponentModel(
        possible_cards=possible,
        hand_counts=state.hand_counts,
        holder_marginals=marginals,
        inference_mode="exact_count_dp" if marginals is not None else "weighted_possible_holders",
    )


def possible_opponent_cards(state: GameState, knowledge: PlayerKnowledge) -> dict[int, set[Card]]:
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

    return possible


def exact_holder_marginals(
    possible_cards: Mapping[int, set[Card]],
    hand_counts: tuple[int | None, int | None, int | None, int | None] | None,
    hidden_cards: Iterable[Card] | None = None,
) -> dict[int, dict[Card, float]] | None:
    if hand_counts is None:
        return None

    quota_players = tuple(
        player
        for player in possible_cards
        if hand_counts[player] is not None
    )
    if not quota_players:
        return None

    quotas = tuple(hand_counts[player] for player in quota_players)
    assert all(count is not None for count in quotas)
    initial = tuple(int(count) for count in quotas)
    if any(count < 0 for count in initial):
        return None

    cards = sorted(hidden_cards if hidden_cards is not None else set().union(*possible_cards.values()))
    holder_options: list[tuple[int, ...]] = []
    for card in cards:
        holders = tuple(player for player in possible_cards if card in possible_cards[player])
        if not holders:
            return None
        holder_options.append(holders)

    quota_index = {player: index for index, player in enumerate(quota_players)}

    def next_state(state: tuple[int, ...], holder: int) -> tuple[int, ...] | None:
        index = quota_index.get(holder)
        if index is None:
            return state
        if state[index] == 0:
            return None
        values = list(state)
        values[index] -= 1
        return tuple(values)

    @lru_cache(maxsize=None)
    def ways_from(index: int, state: tuple[int, ...]) -> int:
        if index == len(cards):
            return 1 if all(value == 0 for value in state) else 0

        total = 0
        for holder in holder_options[index]:
            state_after = next_state(state, holder)
            if state_after is not None:
                total += ways_from(index + 1, state_after)
        return total

    total_ways = ways_from(0, initial)
    if total_ways == 0:
        return None

    prefix: list[dict[tuple[int, ...], int]] = [{initial: 1}]
    for index, holders in enumerate(holder_options):
        next_prefix: dict[tuple[int, ...], int] = {}
        for state, count in prefix[index].items():
            for holder in holders:
                state_after = next_state(state, holder)
                if state_after is not None and ways_from(index + 1, state_after):
                    next_prefix[state_after] = next_prefix.get(state_after, 0) + count
        prefix.append(next_prefix)

    assignment_counts: dict[int, dict[Card, int]] = {player: {} for player in possible_cards}
    for index, card in enumerate(cards):
        for state, prefix_count in prefix[index].items():
            for holder in holder_options[index]:
                state_after = next_state(state, holder)
                if state_after is None:
                    continue
                completion_count = ways_from(index + 1, state_after)
                if completion_count:
                    player_counts = assignment_counts[holder]
                    player_counts[card] = player_counts.get(card, 0) + prefix_count * completion_count

    return {
        player: {card: count / total_ways for card, count in player_counts.items()}
        for player, player_counts in assignment_counts.items()
    }


def sample_hidden_deal(
    state: GameState,
    knowledge: PlayerKnowledge,
    rng: random.Random | None = None,
) -> HiddenDeal | None:
    rng = rng or random.Random()
    hidden_cards = sorted(knowledge.unseen_cards(state))
    possible = possible_opponent_cards(state, knowledge)

    if state.hand_counts is not None and any(
        count is not None for player, count in enumerate(state.hand_counts) if player in possible
    ):
        assignment = sample_count_consistent_assignment(possible, state.hand_counts, hidden_cards, rng)
    else:
        assignment = sample_unconstrained_assignment(possible, hidden_cards, rng)

    if assignment is None:
        return None

    hands: dict[int, set[Card]] = {player: set() for player in possible}
    for card, player in assignment.items():
        hands[player].add(card)

    return HiddenDeal({player: frozenset(cards) for player, cards in hands.items()})


def sample_hidden_deals(
    state: GameState,
    knowledge: PlayerKnowledge,
    count: int,
    rng: random.Random | None = None,
) -> tuple[HiddenDeal, ...]:
    rng = rng or random.Random()
    deals: list[HiddenDeal] = []
    for _ in range(count):
        deal = sample_hidden_deal(state, knowledge, rng)
        if deal is None:
            break
        deals.append(deal)
    return tuple(deals)


def enumerate_hidden_deals(
    state: GameState,
    knowledge: PlayerKnowledge,
    max_deals: int | None = None,
) -> HiddenDealEnumeration:
    hidden_cards = sorted(knowledge.unseen_cards(state))
    possible = possible_opponent_cards(state, knowledge)
    quota_players = tuple(
        player
        for player in possible
        if state.hand_counts is not None and state.hand_counts[player] is not None
    )
    quotas = {
        player: int(state.hand_counts[player])  # type: ignore[index]
        for player in quota_players
    }
    if any(count < 0 for count in quotas.values()):
        return HiddenDealEnumeration((), exhaustive=True, deal_count=0)

    holder_options: list[tuple[int, ...]] = []
    for card in hidden_cards:
        holders = tuple(player for player in possible if card in possible[player])
        if not holders:
            return HiddenDealEnumeration((), exhaustive=True, deal_count=0)
        holder_options.append(holders)

    deals: list[HiddenDeal] = []
    current_hands: dict[int, set[Card]] = {player: set() for player in possible}
    remaining_quotas = dict(quotas)
    exhaustive = True

    def can_still_satisfy_quotas(index: int) -> bool:
        remaining_cards = hidden_cards[index:]
        for player, quota in remaining_quotas.items():
            if quota < 0:
                return False
            possible_remaining = sum(1 for card in remaining_cards if card in possible[player])
            if quota > possible_remaining:
                return False
        return True

    def walk(index: int) -> None:
        nonlocal exhaustive
        if max_deals is not None and len(deals) >= max_deals:
            exhaustive = False
            return
        if not can_still_satisfy_quotas(index):
            return
        if index == len(hidden_cards):
            if any(quota != 0 for quota in remaining_quotas.values()):
                return
            deals.append(
                HiddenDeal(
                    {
                        player: frozenset(cards)
                        for player, cards in current_hands.items()
                    }
                )
            )
            return

        card = hidden_cards[index]
        for holder in holder_options[index]:
            if holder in remaining_quotas and remaining_quotas[holder] == 0:
                continue
            current_hands[holder].add(card)
            if holder in remaining_quotas:
                remaining_quotas[holder] -= 1
            walk(index + 1)
            if holder in remaining_quotas:
                remaining_quotas[holder] += 1
            current_hands[holder].remove(card)
            if not exhaustive:
                return

    walk(0)
    return HiddenDealEnumeration(tuple(deals), exhaustive=exhaustive, deal_count=len(deals))


def evaluate_move_exact_imperfect_information(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    max_deals: int | None = None,
) -> ExactImperfectInformationMoveScore:
    if state.current_player != knowledge.player:
        raise ValueError("exact imperfect-information evaluation requires knowledge.player to act")
    if card not in state.legal_moves(knowledge.hand):
        raise ValueError(f"cannot evaluate illegal move {card}")

    enumeration = enumerate_hidden_deals(state, knowledge, max_deals=max_deals)
    return evaluate_move_exact_imperfect_information_from_deals(state, knowledge, card, enumeration)


def evaluate_move_exact_imperfect_information_from_deals(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    enumeration: HiddenDealEnumeration,
    exact_cache: dict[FullInformationState, tuple[float, float, float, float]] | None = None,
    exact_stats: dict[str, int] | None = None,
) -> ExactImperfectInformationMoveScore:
    if state.current_player != knowledge.player:
        raise ValueError("exact imperfect-information evaluation requires knowledge.player to act")
    if card not in state.legal_moves(knowledge.hand):
        raise ValueError(f"cannot evaluate illegal move {card}")

    if not enumeration.deals:
        return ExactImperfectInformationMoveScore(
            card=card,
            expected_value=float("-inf"),
            expected_value_vector=(float("-inf"), float("-inf"), float("-inf"), float("-inf")),
            hidden_deals=0,
            exhaustive=enumeration.exhaustive,
        )

    exact_cache = exact_cache if exact_cache is not None else {}
    exact_stats = exact_stats if exact_stats is not None else {
        "states_evaluated": 0,
        "cache_hits": 0,
        "terminal_states": 0,
        "deadlock_states": 0,
    }
    values: list[tuple[float, float, float, float]] = []
    outcome_counts = [0, 0, 0, 0]
    neutral_outcomes = 0
    public_legal_moves = tuple(state.legal_moves(knowledge.hand))
    for hidden_deal in enumeration.deals:
        hands = complete_hands(knowledge, hidden_deal)
        full_state = FullInformationState.from_hands(hands, state)
        if full_state.legal_moves() != public_legal_moves:
            raise ValueError(
                "materialized hidden deal changed the solver's legal move set: "
                f"public={labels(public_legal_moves)}, materialized={labels(full_state.legal_moves())}"
            )
        child_state = full_state.after_action(card)
        value = solve_full_information_value(child_state, exact_cache, exact_stats)
        values.append(value)
        winners = [player for player, player_value in enumerate(value) if player_value == 1.0]
        if len(winners) == 1:
            outcome_counts[winners[0]] += 1
        else:
            neutral_outcomes += 1

    expected_value_vector = tuple(
        sum(value[player] for value in values) / len(values)
        for player in range(4)
    )
    return ExactImperfectInformationMoveScore(
        card=card,
        expected_value=expected_value_vector[knowledge.player],
        expected_value_vector=expected_value_vector,  # type: ignore[arg-type]
        hidden_deals=enumeration.deal_count,
        exhaustive=enumeration.exhaustive,
        outcome_counts=tuple(outcome_counts),  # type: ignore[arg-type]
        neutral_outcomes=neutral_outcomes,
        value_standard_error=0.0 if enumeration.exhaustive else nan,
    )


def recommend_move_exact_imperfect_information(
    state: GameState,
    knowledge: PlayerKnowledge,
    max_deals: int | None = None,
) -> ExactImperfectInformationResult | None:
    legal = state.legal_moves(knowledge.hand)
    if not legal:
        return None

    enumeration = enumerate_hidden_deals(state, knowledge, max_deals=max_deals)
    exact_cache: dict[FullInformationState, tuple[float, float, float, float]] = {}
    exact_stats = {
        "states_evaluated": 0,
        "cache_hits": 0,
        "terminal_states": 0,
        "deadlock_states": 0,
    }
    scores = tuple(
        evaluate_move_exact_imperfect_information_from_deals(
            state,
            knowledge,
            card,
            enumeration,
            exact_cache,
            exact_stats,
        )
        for card in legal
    )
    hidden_deals = scores[0].hidden_deals if scores else 0
    exhaustive = all(score.exhaustive for score in scores)
    value_vectors = {score.card: score.expected_value_vector for score in scores}
    best_value = max(score.expected_value for score in scores)
    best_moves = tuple(score.card for score in scores if score.expected_value == best_value)
    chosen_move = choose_expected_value_vector_move(knowledge.player, value_vectors) if scores else None
    return ExactImperfectInformationResult(
        chosen_move=chosen_move,
        best_moves=best_moves,
        move_scores=scores,
        hidden_deals=hidden_deals,
        exhaustive=exhaustive,
        states_evaluated=exact_stats["states_evaluated"],
        cache_hits=exact_stats["cache_hits"],
        terminal_states=exact_stats["terminal_states"],
        deadlock_states=exact_stats["deadlock_states"],
    )


def format_exact_imperfect_information_certificate(
    state: GameState,
    knowledge: PlayerKnowledge,
    result: ExactImperfectInformationResult,
) -> str:
    lines = [
        "Exact imperfect-information certificate",
        f"policy: {result.policy_name}",
        f"continuation_model: {result.continuation_model}",
        f"solver_player: P{knowledge.player}",
        f"current_player: P{state.current_player}",
        f"solver_hand: {labels(knowledge.hand)}",
        f"hand_counts: {state.hand_counts}",
        f"legal_moves: {labels(state.legal_moves(knowledge.hand))}",
        f"hidden_deals: {result.hidden_deals}",
        f"exhaustive: {result.exhaustive}",
        f"best_moves: {labels(result.best_moves)}",
        f"chosen_move: {result.chosen_move.label() if result.chosen_move else 'pass'}",
        (
            "search: "
            f"states={result.states_evaluated}, cache_hits={result.cache_hits}, "
            f"terminals={result.terminal_states}, deadlocks={result.deadlock_states}"
        ),
        "move_expected_values:",
    ]
    for score in sorted(result.move_scores, key=lambda item: item.card):
        marker = "*" if score.card == result.chosen_move else " "
        suffix = ""
        if not score.exhaustive:
            suffix = ", uncertainty=not available for deterministic truncation"
        elif not isnan(score.value_standard_error):
            suffix = f", se={score.value_standard_error:.3f}"
        lines.append(
            f"  {marker} {score.card.label()}: EV={score.expected_value:.3f}, "
            f"vector={format_value(score.expected_value_vector)}, "
            f"outcomes=(P0={score.outcome_counts[0]}, P1={score.outcome_counts[1]}, "
            f"P2={score.outcome_counts[2]}, P3={score.outcome_counts[3]}, "
            f"neutral={score.neutral_outcomes}), deals={score.hidden_deals}, "
            f"exhaustive={score.exhaustive}{suffix}"
        )
    return "\n".join(lines)


def sample_unconstrained_assignment(
    possible_cards: Mapping[int, set[Card]],
    hidden_cards: Iterable[Card],
    rng: random.Random,
) -> dict[Card, int] | None:
    assignment: dict[Card, int] = {}
    for card in hidden_cards:
        holders = [player for player in possible_cards if card in possible_cards[player]]
        if not holders:
            return None
        assignment[card] = rng.choice(holders)
    return assignment


def sample_count_consistent_assignment(
    possible_cards: Mapping[int, set[Card]],
    hand_counts: tuple[int | None, int | None, int | None, int | None],
    hidden_cards: Iterable[Card],
    rng: random.Random,
) -> dict[Card, int] | None:
    quota_players = tuple(
        player
        for player in possible_cards
        if hand_counts[player] is not None
    )
    if not quota_players:
        return sample_unconstrained_assignment(possible_cards, hidden_cards, rng)

    initial = tuple(int(hand_counts[player]) for player in quota_players)
    if any(count < 0 for count in initial):
        return None

    cards = sorted(hidden_cards)
    holder_options: list[tuple[int, ...]] = []
    for card in cards:
        holders = tuple(player for player in possible_cards if card in possible_cards[player])
        if not holders:
            return None
        holder_options.append(holders)

    quota_index = {player: index for index, player in enumerate(quota_players)}

    def next_state(state: tuple[int, ...], holder: int) -> tuple[int, ...] | None:
        index = quota_index.get(holder)
        if index is None:
            return state
        if state[index] == 0:
            return None
        values = list(state)
        values[index] -= 1
        return tuple(values)

    @lru_cache(maxsize=None)
    def ways_from(index: int, quota_state: tuple[int, ...]) -> int:
        if index == len(cards):
            return 1 if all(value == 0 for value in quota_state) else 0

        total = 0
        for holder in holder_options[index]:
            state_after = next_state(quota_state, holder)
            if state_after is not None:
                total += ways_from(index + 1, state_after)
        return total

    quota_state = initial
    if ways_from(0, quota_state) == 0:
        return None

    assignment: dict[Card, int] = {}
    for index, card in enumerate(cards):
        weighted_holders: list[tuple[int, tuple[int, ...], int]] = []
        for holder in holder_options[index]:
            state_after = next_state(quota_state, holder)
            if state_after is None:
                continue
            ways = ways_from(index + 1, state_after)
            if ways:
                weighted_holders.append((holder, state_after, ways))

        total = sum(ways for _, _, ways in weighted_holders)
        pick = rng.randrange(total)
        running = 0
        for holder, state_after, ways in weighted_holders:
            running += ways
            if pick < running:
                assignment[card] = holder
                quota_state = state_after
                break

    return assignment


def recommend_move_monte_carlo(
    state: GameState,
    knowledge: PlayerKnowledge,
    samples_per_move: int = 100,
    max_turns: int = 200,
    rng: random.Random | None = None,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> MonteCarloMoveScore | None:
    legal = state.legal_moves(knowledge.hand)
    if not legal:
        return None

    rng = rng or random.Random()
    results = [
        evaluate_move_monte_carlo(state, knowledge, card, samples_per_move, max_turns, rng, weights)
        for card in legal
    ]
    return max(results, key=lambda result: result.score)


def evaluate_move_monte_carlo(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    samples: int = 100,
    max_turns: int = 200,
    rng: random.Random | None = None,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> MonteCarloMoveScore:
    rng = rng or random.Random()
    wins = 0
    finish_margin_total = 0.0
    turns_total = 0
    timeouts = 0
    completed_samples = 0

    for _ in range(samples):
        hidden_deal = sample_hidden_deal(state, knowledge, rng)
        if hidden_deal is None:
            break

        hands = complete_hands(knowledge, hidden_deal)
        if card not in hands[knowledge.player]:
            continue

        next_state, next_hands = apply_known_play(state, hands, knowledge.player, card)
        result = rollout_oracle(next_state, next_hands, max_turns, weights)
        completed_samples += 1
        if result.winner == knowledge.player:
            wins += 1
        finish_margin_total += finish_margin(result.final_hand_counts, knowledge.player)
        turns_total += result.turns_played
        if result.timed_out:
            timeouts += 1

    if completed_samples == 0:
        return MonteCarloMoveScore(card=card, score=float("-inf"), win_rate=0.0, average_finish_margin=0.0, samples=0)

    win_rate = wins / completed_samples
    average_finish_margin = finish_margin_total / completed_samples
    average_turns = turns_total / completed_samples
    timeout_rate = timeouts / completed_samples
    win_rate_standard_error = sqrt(win_rate * (1.0 - win_rate) / completed_samples)
    score = (
        win_rate * weights.monte_carlo_win_rate_weight
        + average_finish_margin
        - timeout_rate * weights.monte_carlo_timeout_penalty
    )
    return MonteCarloMoveScore(
        card=card,
        score=score,
        win_rate=win_rate,
        average_finish_margin=average_finish_margin,
        samples=completed_samples,
        average_turns=average_turns,
        timeout_rate=timeout_rate,
        win_rate_standard_error=win_rate_standard_error,
    )


def complete_hands(knowledge: PlayerKnowledge, hidden_deal: HiddenDeal) -> dict[int, set[Card]]:
    hands = {player: set(hidden_deal.hand(player)) for player in range(4)}
    hands[knowledge.player] = set(knowledge.hand)
    return hands


def rollout_oracle(
    state: GameState,
    hands: Mapping[int, set[Card]],
    max_turns: int = 200,
    weights: StrategyWeights | Mapping[int, StrategyWeights] = DEFAULT_WEIGHTS,
) -> RolloutResult:
    current_state = state_with_hand_counts(state, hands)
    current_hands = {player: set(hand) for player, hand in hands.items()}

    for turns_played in range(max_turns + 1):
        winner = first_empty_player(current_hands)
        if winner is not None:
            return RolloutResult(winner, turns_played, hand_count_tuple(current_hands))
        if turns_played == max_turns:
            break

        player = current_state.current_player
        card = choose_oracle_move(current_state, current_hands, player, weights_for_player(weights, player))
        current_state, current_hands = apply_known_play(current_state, current_hands, player, card)

    return RolloutResult(None, max_turns, hand_count_tuple(current_hands), timed_out=True)


def weights_for_player(
    weights: StrategyWeights | Mapping[int, StrategyWeights],
    player: int,
) -> StrategyWeights:
    if isinstance(weights, StrategyWeights):
        return weights
    return weights.get(player, DEFAULT_WEIGHTS)


def choose_oracle_move(
    state: GameState,
    hands: Mapping[int, set[Card]],
    player: int,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> Card | None:
    legal = state.legal_moves(hands[player])
    if not legal:
        return None
    return max(legal, key=lambda card: score_oracle_move(state, hands, player, card, weights))


def score_oracle_move(
    state: GameState,
    hands: Mapping[int, set[Card]],
    player: int,
    card: Card,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> float:
    next_state = state.after_play(player, card)
    hands_after = hands_after_play(hands, player, card)

    local_value = exact_local_move_value(state, hands, player, card, weights)
    next_player = next_state.current_player
    next_legal = next_state.legal_moves(hands_after[next_player])
    if not next_legal:
        return local_value

    response_value = max(
        exact_local_move_value(next_state, hands_after, next_player, response, weights)
        for response in next_legal
    )
    return local_value - weights.response_penalty * response_value * turn_order_weight(player, next_player)


def exact_local_move_value(
    state: GameState,
    hands: Mapping[int, set[Card]],
    player: int,
    card: Card,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> float:
    hand_after = set(hands[player]) - {card}
    hands_after = hands_after_play(hands, player, card)
    next_state = state.after_play(player, card)
    before_legal = set(state.legal_moves(hands[player]))
    after_legal = set(next_state.legal_moves(hand_after))
    newly_self_playable = after_legal - before_legal

    opened_for_table = next_state.public_legal_cards() - state.public_legal_cards()
    opponent_risk = exact_opponent_unlock_risk(player, hands_after, opened_for_table, weights)
    chain_impact = exact_future_chain_impact(state, hands_after, player, card, weights)
    race_pressure = exact_race_pressure(hands_after, player, weights)

    return weights.self_unlock * len(newly_self_playable) - opponent_risk + chain_impact + race_pressure


def hands_after_play(
    hands: Mapping[int, set[Card]],
    player: int,
    card: Card,
) -> dict[int, set[Card]]:
    next_hands = {p: set(hand) for p, hand in hands.items()}
    next_hands[player].remove(card)
    return next_hands


def exact_opponent_unlock_risk(
    player: int,
    hands: Mapping[int, set[Card]],
    opened_cards: set[Card],
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> float:
    risk = 0.0
    for opponent, hand in hands.items():
        if opponent == player:
            continue
        playable_count = len(opened_cards & hand)
        risk += min(1.0, float(playable_count)) * turn_order_weight(player, opponent)
    return risk * weights.opponent_unlock_risk


def exact_future_chain_impact(
    state: GameState,
    hands: Mapping[int, set[Card]],
    player: int,
    card: Card,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> float:
    hand_after = hands[player]
    impact = 0.0
    for side in released_sides(card):
        future_cards = {Card(card.suit, rank) for rank in side}
        own_future = future_cards & hand_after
        opponent_future = {
            future_card
            for opponent, hand in hands.items()
            if opponent != player
            for future_card in future_cards & hand
        }

        impact -= weights.future_chain_unseen_penalty * len(opponent_future)
        impact += weights.future_chain_own_credit * len(own_future)

        tail_card = owned_tail_card(card.suit, side, own_future)
        if tail_card is not None:
            impact += max(
                0.0,
                weights.tail_base_credit
                - time_to_playable(state, tail_card) * weights.tail_distance_penalty,
            )

        if side and owns_gate_card(card.suit, side, own_future):
            impact += weights.gate_card_credit

        opponent_runway_mass = sum(
            turn_order_weight(player, opponent)
            for opponent, hand in hands.items()
            if opponent != player
            for future_card in future_cards & hand
        )
        impact -= weights.opponent_runway_penalty * opponent_runway_mass

    return impact


def exact_race_pressure(
    hands: Mapping[int, set[Card]],
    player: int,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> float:
    my_count = len(hands[player])
    opponent_counts = [len(hand) for p, hand in hands.items() if p != player]
    lowest_opponent = min(opponent_counts)
    return max(0.0, lowest_opponent - my_count) * weights.race_pressure_credit


def apply_known_play(
    state: GameState,
    hands: Mapping[int, set[Card]],
    player: int,
    card: Card | None,
) -> tuple[GameState, dict[int, set[Card]]]:
    next_hands = {p: set(hand) for p, hand in hands.items()}
    if card is not None:
        next_hands[player].remove(card)

    counted_state = state_with_hand_counts(state, hands)
    next_state = counted_state.after_play(player, card, hand=hands[player])
    return state_with_hand_counts(next_state, next_hands), next_hands


def state_with_hand_counts(state: GameState, hands: Mapping[int, set[Card]]) -> GameState:
    return GameState(
        table=state.table,
        hand_counts=hand_count_tuple(hands),
        current_player=state.current_player,
        history=state.history,
    )


def hand_count_tuple(hands: Mapping[int, set[Card]]) -> tuple[int, int, int, int]:
    return tuple(len(hands[player]) for player in range(4))  # type: ignore[return-value]


def first_empty_player(hands: Mapping[int, set[Card]]) -> int | None:
    for player in range(4):
        if not hands.get(player, set()):
            return player
    return None


def finish_margin(hand_counts: tuple[int, int, int, int], player: int) -> float:
    opponents = [count for p, count in enumerate(hand_counts) if p != player]
    return min(opponents) - hand_counts[player]


def deal_random_hands(rng: random.Random | None = None) -> dict[int, set[Card]]:
    rng = rng or random.Random()
    deck = sorted(full_deck(), key=lambda card: (card.suit.value, card.rank))
    rng.shuffle(deck)
    return {player: set(deck[player * 13 : (player + 1) * 13]) for player in range(4)}


def initial_state_for_hands(hands: Mapping[int, set[Card]]) -> GameState:
    seven_hearts = Card(Suit.HEARTS, 7)
    starting_player = next(player for player, hand in hands.items() if seven_hearts in hand)
    return GameState(hand_counts=hand_count_tuple(hands), current_player=starting_player)


def simulate_complete_game(
    rng: random.Random | None = None,
    max_turns: int = 300,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> RolloutResult:
    hands = deal_random_hands(rng)
    state = initial_state_for_hands(hands)
    return rollout_oracle(state, hands, max_turns=max_turns, weights=weights)


def estimate_complete_game_metrics(
    games: int,
    rng: random.Random | None = None,
    max_turns: int = 300,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> CompleteGameEstimate:
    rng = rng or random.Random()
    wins = [0, 0, 0, 0]
    finish_margin_totals = [0.0, 0.0, 0.0, 0.0]
    turns_total = 0
    timeouts = 0

    for _ in range(games):
        result = simulate_complete_game(rng, max_turns=max_turns, weights=weights)
        if result.winner is not None:
            wins[result.winner] += 1
        for player in range(4):
            finish_margin_totals[player] += finish_margin(result.final_hand_counts, player)
        turns_total += result.turns_played
        if result.timed_out:
            timeouts += 1

    if games == 0:
        return CompleteGameEstimate(
            games=0,
            win_rates=(0.0, 0.0, 0.0, 0.0),
            win_rate_standard_errors=(0.0, 0.0, 0.0, 0.0),
            average_finish_margins=(0.0, 0.0, 0.0, 0.0),
            average_turns=0.0,
            timeout_rate=0.0,
        )

    win_rates = tuple(wins[player] / games for player in range(4))
    return CompleteGameEstimate(
        games=games,
        win_rates=win_rates,  # type: ignore[arg-type]
        win_rate_standard_errors=tuple(
            sqrt(rate * (1.0 - rate) / games) for rate in win_rates
        ),  # type: ignore[arg-type]
        average_finish_margins=tuple(
            finish_margin_totals[player] / games for player in range(4)
        ),  # type: ignore[arg-type]
        average_turns=turns_total / games,
        timeout_rate=timeouts / games,
    )


def estimate_strategy_self_play(
    candidates: Iterable[StrategyCandidate],
    games: int,
    rng: random.Random | None = None,
    max_turns: int = 300,
    player: int = 0,
    baseline_weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> tuple[StrategySelfPlayEstimate, ...]:
    rng = rng or random.Random()
    candidates = tuple(candidates)
    deals = [deal_random_hands(rng) for _ in range(games)]
    estimates: list[StrategySelfPlayEstimate] = []

    for candidate in candidates:
        wins = 0
        finish_margin_total = 0.0
        turns_total = 0
        timeouts = 0

        for deal in deals:
            hands = {seat: set(cards) for seat, cards in deal.items()}
            state = initial_state_for_hands(hands)
            rollout_weights = {seat: baseline_weights for seat in range(4)}
            rollout_weights[player] = candidate.weights
            result = rollout_oracle(state, hands, max_turns=max_turns, weights=rollout_weights)

            if result.winner == player:
                wins += 1
            finish_margin_total += finish_margin(result.final_hand_counts, player)
            turns_total += result.turns_played
            if result.timed_out:
                timeouts += 1

        if games == 0:
            estimates.append(
                StrategySelfPlayEstimate(
                    candidate=candidate,
                    games=0,
                    win_rate=0.0,
                    win_rate_standard_error=0.0,
                    average_finish_margin=0.0,
                    average_turns=0.0,
                    timeout_rate=0.0,
                    score=0.0,
                )
            )
            continue

        win_rate = wins / games
        timeout_rate = timeouts / games
        average_finish_margin = finish_margin_total / games
        score = (
            win_rate * candidate.weights.monte_carlo_win_rate_weight
            + average_finish_margin
            - timeout_rate * candidate.weights.monte_carlo_timeout_penalty
        )
        estimates.append(
            StrategySelfPlayEstimate(
                candidate=candidate,
                games=games,
                win_rate=win_rate,
                win_rate_standard_error=sqrt(win_rate * (1.0 - win_rate) / games),
                average_finish_margin=average_finish_margin,
                average_turns=turns_total / games,
                timeout_rate=timeout_rate,
                score=score,
            )
        )

    return tuple(sorted(estimates, key=lambda estimate: estimate.score, reverse=True))


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
    weights: StrategyWeights = DEFAULT_WEIGHTS,
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
    self_unlock = weights.self_unlock * len(newly_self_playable)
    if self_unlock:
        score += self_unlock
        components["self_unlock"] = self_unlock
        reasons.append(f"unlocks own cards {labels(newly_self_playable)}: +{self_unlock:.1f}")

    opened_for_table = next_state.public_legal_cards() - state.public_legal_cards()
    opened_for_others = opened_for_table - hand_after
    opponent_risk = opponent_unlock_risk(state, knowledge, model, opened_for_others, weights)
    if opponent_risk:
        score -= opponent_risk
        components["opponent_unlock_risk"] = -opponent_risk
        reasons.append(f"may unlock opponents {labels(opened_for_others)}: -{opponent_risk:.1f}")

    chain_impact = future_chain_impact(state, knowledge, card, model, weights)
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
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> float:
    risk = 0.0
    for opponent in model.possible_cards:
        playable_mass = sum(model.holder_probability(opponent, card) for card in opened_cards)
        capped_mass = min(1.0, playable_mass)
        risk += capped_mass * turn_order_weight(knowledge.player, opponent)
    return risk * weights.opponent_unlock_risk


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
    weights: StrategyWeights = DEFAULT_WEIGHTS,
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
        impact -= weights.future_chain_unseen_penalty * len(unseen_future)
        impact += weights.future_chain_own_credit * len(own_future)

        tail_card = owned_tail_card(card.suit, side, own_future)
        if tail_card is not None:
            impact += max(
                0.0,
                weights.tail_base_credit
                - time_to_playable(state, tail_card) * weights.tail_distance_penalty,
            )

        if side and owns_gate_card(card.suit, side, hand_after):
            impact += weights.gate_card_credit

        opponent_runway_mass = sum(
            model.holder_probability(opponent, future_card) * turn_order_weight(knowledge.player, opponent)
            for future_card in future_cards
            for opponent in model.possible_cards
        )
        impact -= weights.opponent_runway_penalty * opponent_runway_mass

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


def recommend_move(
    state: GameState,
    knowledge: PlayerKnowledge,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> MoveScore | None:
    legal = state.legal_moves(knowledge.hand)
    if not legal:
        return None
    model = build_opponent_model(state, knowledge)
    return max((score_move(state, knowledge, card, model, weights) for card in legal), key=lambda s: s.score)


def labels(cards: Iterable[Card]) -> str:
    ordered = sorted(cards)
    if not ordered:
        return "[]"
    return "[" + ", ".join(card.label() for card in ordered) + "]"
