from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from math import comb, exp, isnan, nan, sqrt
from operator import add, le
from time import perf_counter
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence


class Suit(str, Enum):
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"


SUIT_ORDER = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
SUIT_INDEX = {suit: index for index, suit in enumerate(SUIT_ORDER)}
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


CARD_BY_INDEX = tuple(Card(suit, rank) for suit in SUIT_ORDER for rank in range(1, 14))
CARD_TO_INDEX = {card: index for index, card in enumerate(CARD_BY_INDEX)}
FULL_DECK_MASK = (1 << 52) - 1


def card_index(card: Card) -> int:
    return CARD_TO_INDEX[card]


def card_bit(card: Card) -> int:
    return 1 << card_index(card)


def card_from_index(index: int) -> Card:
    return CARD_BY_INDEX[index]


def card_set_mask(cards: Iterable[Card]) -> int:
    mask = 0
    for card in cards:
        mask |= card_bit(card)
    return mask


def mask_to_cards(mask: int) -> tuple[Card, ...]:
    cards: list[Card] = []
    while mask:
        bit = mask & -mask
        cards.append(card_from_index(bit.bit_length() - 1))
        mask ^= bit
    return tuple(cards)


def maybe_only_card(mask: int) -> Card | None:
    if mask and mask & (mask - 1) == 0:
        return card_from_index(mask.bit_length() - 1)
    return None


def full_deck() -> set[Card]:
    return set(CARD_BY_INDEX)


def reduced_deck(cards_per_suit: int) -> set[Card]:
    if cards_per_suit < 1 or cards_per_suit > 13:
        raise ValueError(f"cards_per_suit must be between 1 and 13, got {cards_per_suit}")

    lower_span = (cards_per_suit - 1) // 2
    low = 7 - lower_span
    high = low + cards_per_suit - 1
    if high > 13:
        high = 13
        low = high - cards_per_suit + 1
    if low < 1:
        low = 1
        high = low + cards_per_suit - 1

    return {Card(suit, rank) for suit in Suit for rank in range(low, high + 1)}


@dataclass(frozen=True)
class SuitRun:
    low: int | None = None
    high: int | None = None

    @property
    def is_open(self) -> bool:
        return self.low is not None and self.high is not None

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

    def played_mask(self, suit: Suit) -> int:
        return run_masks(suit, self.low, self.high)[0]

    def legal_mask(self, suit: Suit) -> int:
        return run_masks(suit, self.low, self.high)[1]


@lru_cache(maxsize=None)
def run_masks(suit: Suit, low: int | None, high: int | None) -> tuple[int, int]:
    """(played cards, next legal cards) masks for one suit's run; at most 4 * 92 distinct inputs."""
    base = SUIT_INDEX[suit] * 13
    if low is None or high is None:
        return 0, 1 << (base + 6)
    played = 0
    for rank in range(low, high + 1):
        played |= 1 << (base + rank - 1)
    legal = 0
    if low > 1:
        legal |= 1 << (base + low - 2)
    if high < 13:
        legal |= 1 << (base + high)
    return played, legal


def table_played_mask(runs: Iterable[tuple[Suit, SuitRun]]) -> int:
    mask = 0
    for suit, run in runs:
        mask |= run.played_mask(suit)
    return mask


def table_legal_mask(runs: Iterable[tuple[Suit, SuitRun]]) -> int:
    """Cards that may be played next on this table; only 7H before anything is played."""
    runs = tuple(runs)
    played_mask = table_played_mask(runs)
    if not played_mask:
        return card_bit(Card(Suit.HEARTS, 7))
    legal_mask = 0
    for suit, run in runs:
        legal_mask |= run.legal_mask(suit)
    return legal_mask & ~played_mask


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
        return set(mask_to_cards(self.played_cards_mask()))

    def played_cards_mask(self) -> int:
        return table_played_mask(self.table.items())

    def public_legal_cards(self) -> set[Card]:
        return set(mask_to_cards(self.public_legal_mask()))

    def public_legal_mask(self) -> int:
        return table_legal_mask(self.table.items())

    def legal_moves(self, hand: Iterable[Card]) -> list[Card]:
        return list(mask_to_cards(card_set_mask(hand) & self.public_legal_mask()))

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


NEUTRAL_VALUE = (0.0, 0.0, 0.0, 0.0)
ExactCacheKey = tuple[tuple[int, int, int, int], tuple[tuple[int | None, int | None], ...], int, int | None, int]


def canonical_table(table: Mapping[Suit, SuitRun] | None = None) -> tuple[SuitRun, SuitRun, SuitRun, SuitRun]:
    table = table or {suit: SuitRun() for suit in Suit}
    return tuple(table.get(suit, SuitRun()) for suit in SUIT_ORDER)  # type: ignore[return-value]


def trailing_pass_count(history: Iterable[MoveEvent]) -> int:
    count = 0
    for event in reversed(tuple(history)):
        if not event.is_pass:
            break
        count += 1
    return count

@dataclass(frozen=True)
class FullInformationState:
    hands: tuple[frozenset[Card], frozenset[Card], frozenset[Card], frozenset[Card]]
    table: tuple[SuitRun, SuitRun, SuitRun, SuitRun] = field(default_factory=canonical_table)
    current_player: int = 0
    winner: int | None = None
    consecutive_passes: int = 0
    hand_masks: tuple[int, int, int, int] = field(init=False, compare=False, repr=False)

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
        object.__setattr__(self, "hand_masks", tuple(card_set_mask(hand) for hand in self.hands))

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

    def played_cards(self) -> set[Card]:
        return set(mask_to_cards(self.played_cards_mask()))

    def played_cards_mask(self) -> int:
        return table_played_mask(zip(SUIT_ORDER, self.table))

    def public_legal_cards(self) -> set[Card]:
        return set(mask_to_cards(self.public_legal_mask()))

    def public_legal_mask(self) -> int:
        return table_legal_mask(zip(SUIT_ORDER, self.table))

    def legal_moves(self) -> tuple[Card, ...]:
        if self.winner is not None:
            return ()
        legal_mask = self.hand_masks[self.current_player] & self.public_legal_mask()
        return mask_to_cards(legal_mask)

    def compact_key(self) -> ExactCacheKey:
        table_key = tuple((run.low, run.high) for run in self.table)
        return self.hand_masks, table_key, self.current_player, self.winner, self.consecutive_passes

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
        suit_index = SUIT_INDEX[card.suit]
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


def new_search_stats() -> dict[str, int]:
    return {"states_evaluated": 0, "cache_hits": 0, "terminal_states": 0, "deadlock_states": 0}


def terminal_value(winner: int) -> tuple[float, float, float, float]:
    values = [0.0, 0.0, 0.0, 0.0]
    values[winner] = 1.0
    return tuple(values)  # type: ignore[return-value]


def solve_full_information(state: FullInformationState) -> ExactSolverResult:
    cache: dict[ExactCacheKey, tuple[float, float, float, float]] = {}
    stats = new_search_stats()

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
    cache: dict[ExactCacheKey, tuple[float, float, float, float]],
    stats: dict[str, int],
    policy: Callable[[FullInformationState], Card] | None = None,
) -> tuple[float, float, float, float]:
    """Exact value of `current`. Every player moves rationally unless `policy` fixes their choice."""
    cache_key = current.compact_key()
    cached = cache.get(cache_key)
    if cached is not None:
        stats["cache_hits"] += 1
        return cached

    stats["states_evaluated"] += 1
    if current.winner is not None:
        stats["terminal_states"] += 1
        value = terminal_value(current.winner)
        cache[cache_key] = value
        return value

    legal = current.legal_moves()
    if not legal:
        canonical = canonicalize_forced_pass_chain(current)
        if canonical != current:
            value = solve_full_information_value(canonical, cache, stats, policy)
            cache[cache_key] = value
            return value
        if canonical.consecutive_passes >= 4:
            stats["deadlock_states"] += 1
            cache[cache_key] = NEUTRAL_VALUE
            return NEUTRAL_VALUE

    if policy is not None:
        chosen = policy(current)
        if chosen not in legal:
            raise ValueError(f"fixed policy chose illegal move {chosen}; legal moves are {labels(legal)}")
        value = solve_full_information_value(current.after_action(chosen), cache, stats, policy)
    else:
        child_values = {
            card: solve_full_information_value(current.after_action(card), cache, stats)
            for card in legal
        }
        value = child_values[choose_rational_move(current.current_player, child_values)]
    cache[cache_key] = value
    return value


def solve_full_information_against_policy(
    state: FullInformationState,
    policy: Callable[[FullInformationState], Card],
    policy_name: str = "fixed_policy",
) -> ExactSolverResult:
    cache: dict[ExactCacheKey, tuple[float, float, float, float]] = {}
    stats = new_search_stats()

    def solve_policy_value(current: FullInformationState) -> tuple[float, float, float, float]:
        return solve_full_information_value(current, cache, stats, policy)

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


SUIT_STATES = ((None, None),) + tuple((low, high) for low in range(1, 8) for high in range(7, 14))
OPENED_SUIT_STATE = 1 + 6 * 7  # (7, 7)
COMPACT_MEMO_MAX_CARDS = 24


def suit_state_code(low: int | None, high: int | None) -> int:
    """Index into SUIT_STATES: 0 for an unopened suit, else 1 + 7 * (low - 1) + (high - 7)."""
    return 0 if low is None or high is None else 1 + (low - 1) * 7 + (high - 7)


def table_suit_codes(table: Mapping[Suit, SuitRun]) -> tuple[int, int, int, int]:
    return tuple(suit_state_code(table[suit].low, table[suit].high) for suit in SUIT_ORDER)  # type: ignore[return-value]


class FirstOutSolver:
    """Exact first-out winner under full-information rational play, for one fixed layout.

    Once the remaining hands are fixed, the table determines who still holds what,
    so a position is just (suit states, player to move): at most 50**4 * 4 of them,
    stored one byte each. Rational play here equals `solve_full_information` with
    `choose_rational_move`: values are one-hot, so a player wins if any move wins,
    otherwise takes the lowest card whose winner is not the next player, otherwise
    the lowest card.
    """

    def __init__(self, hands: Mapping[int, Iterable[Card]]) -> None:
        owner = [[-1] * 15 for _ in range(4)]
        counts = [0, 0, 0, 0]
        for player, hand in hands.items():
            for card in hand:
                owner[SUIT_INDEX[card.suit]][card.rank] = player
                counts[player] += 1
        self.owner = owner
        self.counts = tuple(counts)
        # moves[suit][code] = ((holder, next_code), ...) in card order, low side first
        self.moves = tuple(
            tuple(self._suit_moves(owner[suit], low, high) for low, high in SUIT_STATES)
            for suit in range(4)
        )
        # played[suit][code][player] = how many of player's current cards that suit state has used
        self.played = tuple(
            tuple(
                tuple(
                    0 if low is None else sum(1 for rank in range(low, high + 1) if owner[suit][rank] == player)
                    for player in range(4)
                )
                for low, high in SUIT_STATES
            )
            for suit in range(4)
        )
        # 0 = unsolved, else winner + 1. Allocating the full table costs more than a whole
        # small solve, so positions with few cards left use a dict that reads 0 when unset.
        if sum(counts) > COMPACT_MEMO_MAX_CARDS:
            self.memo = bytearray(len(SUIT_STATES) ** 4 * 4)
        else:
            self.memo = defaultdict(int)
        self.states_evaluated = 0

    @staticmethod
    def _suit_moves(holders: list[int], low: int | None, high: int | None) -> tuple[tuple[int, int], ...]:
        if low is None or high is None:
            return ((holders[7], OPENED_SUIT_STATE),) if holders[7] >= 0 else ()
        moves = []
        if low > 1 and holders[low - 1] >= 0:
            moves.append((holders[low - 1], suit_state_code(low - 1, high)))
        if high < 13 and holders[high + 1] >= 0:
            moves.append((holders[high + 1], suit_state_code(low, high + 1)))
        return tuple(moves)

    def cards_left(self, codes: Sequence[int], player: int) -> int:
        played = self.played
        return (
            self.counts[player]
            - played[0][codes[0]][player]
            - played[1][codes[1]][player]
            - played[2][codes[2]][player]
            - played[3][codes[3]][player]
        )

    def legal(self, codes: Sequence[int], player: int) -> list[tuple[int, int]]:
        """(suit, next_code) for each legal card of `player`, in card order."""
        if not any(codes):
            hearts = SUIT_INDEX[Suit.HEARTS]
            return [(hearts, OPENED_SUIT_STATE)] if self.owner[hearts][7] == player else []
        return [
            (suit, next_code)
            for suit in range(4)
            for holder, next_code in self.moves[suit][codes[suit]]
            if holder == player
        ]

    def winner(self, codes: Sequence[int], player: int) -> int:
        """First-out winner from this position with `player` to act."""
        for _ in range(4):
            legal = self.legal(codes, player)
            if legal:
                break
            player = (player + 1) % 4
        else:
            raise ValueError("deadlock: nobody can play")

        key = (((codes[0] * 50 + codes[1]) * 50 + codes[2]) * 50 + codes[3]) * 4 + player
        cached = self.memo[key]
        if cached:
            return cached - 1

        self.states_evaluated += 1
        next_player = (player + 1) % 4
        result = None
        fallback = None
        for suit, next_code in legal:
            child = list(codes)
            child[suit] = next_code
            if self.cards_left(child, player) == 0:
                result = player
                break
            child_winner = self.winner(child, next_player)
            if child_winner == player:
                result = player
                break
            if fallback is None and child_winner != next_player:
                fallback = child_winner
        if result is None:
            result = fallback if fallback is not None else next_player
        self.memo[key] = result + 1
        return result

    def move_wins(self, table: Mapping[Suit, SuitRun], player: int, cards: Sequence[Card]) -> list[bool]:
        """For each card `player` could play now, whether they go out first under exact play."""
        codes = table_suit_codes(table)
        out = []
        for card in cards:
            run = table[card.suit].play(card.rank)
            child = list(codes)
            child[SUIT_INDEX[card.suit]] = suit_state_code(run.low, run.high)
            out.append(self.cards_left(child, player) == 0 or self.winner(child, (player + 1) % 4) == player)
        return out


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
class HiddenDealSelection:
    deals: tuple[HiddenDeal, ...]
    exhaustive: bool


class HiddenDealSampler:
    def __init__(self, state: GameState, knowledge: PlayerKnowledge) -> None:
        self.hidden_cards = tuple(sorted(knowledge.unseen_cards(state)))
        possible = possible_opponent_cards(state, knowledge)
        self.possible: dict[int, frozenset[Card]] = {
            player: frozenset(cards) for player, cards in possible.items()
        }
        self.possible_masks = {
            player: card_set_mask(cards) for player, cards in self.possible.items()
        }
        self.holder_options: tuple[tuple[int, ...], ...] = tuple(
            tuple(player for player, mask in self.possible_masks.items() if mask & card_bit(card))
            for card in self.hidden_cards
        )
        self.valid = all(self.holder_options)
        self.quota_players: tuple[int, ...] = ()
        self.initial_quotas: tuple[int, ...] = ()
        self.quota_index: dict[int, int] = {}
        self._ways_cache: dict[tuple[int, tuple[int, ...]], int] = {}

        if state.hand_counts is not None:
            self.quota_players = tuple(
                player
                for player in self.possible
                if state.hand_counts[player] is not None
            )
            self.initial_quotas = tuple(int(state.hand_counts[player]) for player in self.quota_players)
            self.quota_index = {player: index for index, player in enumerate(self.quota_players)}
            if any(count < 0 for count in self.initial_quotas):
                self.valid = False

        if self.valid and self.quota_players and self._ways_from(0, self.initial_quotas) == 0:
            self.valid = False

    def sample(self, rng: random.Random) -> HiddenDeal | None:
        if not self.valid:
            return None
        if self.quota_players:
            assignment = self._sample_count_consistent_assignment(rng)
        else:
            assignment = self._sample_unconstrained_assignment(rng)
        if assignment is None:
            return None

        hands: dict[int, set[Card]] = {player: set() for player in self.possible}
        for card, player in assignment.items():
            hands[player].add(card)

        return HiddenDeal({player: frozenset(cards) for player, cards in hands.items()})

    def _next_quota_state(self, state: tuple[int, ...], holder: int) -> tuple[int, ...] | None:
        index = self.quota_index.get(holder)
        if index is None:
            return state
        if state[index] == 0:
            return None
        values = list(state)
        values[index] -= 1
        return tuple(values)

    def _ways_from(self, index: int, quota_state: tuple[int, ...]) -> int:
        cache_key = (index, quota_state)
        cached = self._ways_cache.get(cache_key)
        if cached is not None:
            return cached
        if index == len(self.hidden_cards):
            total = 1 if all(value == 0 for value in quota_state) else 0
        else:
            total = 0
            for holder in self.holder_options[index]:
                state_after = self._next_quota_state(quota_state, holder)
                if state_after is not None:
                    total += self._ways_from(index + 1, state_after)
        self._ways_cache[cache_key] = total
        return total

    def _sample_unconstrained_assignment(self, rng: random.Random) -> dict[Card, int] | None:
        assignment: dict[Card, int] = {}
        for card, holders in zip(self.hidden_cards, self.holder_options):
            if not holders:
                return None
            assignment[card] = rng.choice(holders)
        return assignment

    def _sample_count_consistent_assignment(self, rng: random.Random) -> dict[Card, int] | None:
        quota_state = self.initial_quotas
        if self._ways_from(0, quota_state) == 0:
            return None

        assignment: dict[Card, int] = {}
        for index, card in enumerate(self.hidden_cards):
            weighted_holders: list[tuple[int, tuple[int, ...], int]] = []
            for holder in self.holder_options[index]:
                state_after = self._next_quota_state(quota_state, holder)
                if state_after is None:
                    continue
                ways = self._ways_from(index + 1, state_after)
                if ways:
                    weighted_holders.append((holder, state_after, ways))

            total = sum(ways for _, _, ways in weighted_holders)
            if total == 0:
                return None
            pick = rng.randrange(total)
            running = 0
            for holder, state_after, ways in weighted_holders:
                running += ways
                if pick < running:
                    assignment[card] = holder
                    quota_state = state_after
                    break

        return assignment


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
    policy_name: str = "oracle_greedy_full_information"
    exhaustive: bool = False


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
DEFAULT_MONTE_CARLO_AGENT_NAME = "Monte Carlo"
INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY = "heuristic_greedy"
LEGACY_INFORMATION_LIMITED_GREEDY_POLICY = "greedy"


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
class FullGameAgent:
    name: str
    policy: str
    weights: StrategyWeights = DEFAULT_WEIGHTS
    samples_per_move: int = 24
    rollout_max_turns: int = 160
    rollout_policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY
    rationality: float = 1.0


@dataclass(frozen=True)
class FullGameDecisionCounts:
    turns: int = 0
    forced_passes: int = 0
    single_legal_moves: int = 0
    immediate_wins: int = 0
    sampled_mc_decisions: int = 0


@dataclass(frozen=True)
class FullGameResult:
    agent_names_by_seat: tuple[str, str, str, str]
    winner: int | None
    finish_order: tuple[int, ...]
    ranks_by_seat: tuple[int, int, int, int]
    final_hand_counts: tuple[int, int, int, int]
    turns_played: int
    timed_out: bool = False
    decision_traces: tuple["DecisionTrace", ...] = ()
    # Hand sizes when the first player went out (final counts if nobody did).
    first_out_hand_counts: tuple[int, int, int, int] | None = None
    decision_counts_by_seat: tuple[
        FullGameDecisionCounts,
        FullGameDecisionCounts,
        FullGameDecisionCounts,
        FullGameDecisionCounts,
    ] = (FullGameDecisionCounts(),) * 4


@dataclass(frozen=True)
class DecisionTrace:
    deal_index: int | None
    rotation_index: int | None
    turn: int
    player: int
    agent_name: str
    legal_moves: tuple[Card, ...]
    hand: tuple[Card, ...]
    table_key: tuple[tuple[int | None, int | None], ...]
    heuristic_card: Card | None
    heuristic_score: float
    heuristic_reasons: tuple[str, ...]
    mc_card: Card | None
    mc_score: float
    mc_win_rate: float
    mc_average_finish_margin: float
    mc_samples: int
    disagreed: bool
    final_rank: int | None = None
    final_cards_left: int | None = None
    winner: int | None = None


@dataclass(frozen=True)
class AgentFullGameSummary:
    agent_name: str
    seats_played: int
    win_rate: float
    win_rate_standard_error: float
    average_cards_at_first_out: float
    cards_at_first_out_standard_error: float
    average_rank: float
    rank_standard_error: float
    # Cards held when play stopped. Only last place has any, so this is mostly noise.
    average_cards_left: float
    cards_left_standard_error: float
    turns: int = 0
    forced_passes: int = 0
    single_legal_moves: int = 0
    immediate_wins: int = 0
    sampled_mc_decisions: int = 0


# Paired metrics, oriented so a positive advantage means the primary agent did better:
#   first_out_win       primary won minus baseline won (the solver's objective)
#   cards_at_first_out  baseline's cards minus primary's when someone first went out
#   rank                baseline's finishing rank minus primary's
PAIRED_METRICS = ("first_out_win", "cards_at_first_out", "rank")


@dataclass(frozen=True)
class PairedComparison:
    primary_agent: str
    baseline_agent: str
    metric: str
    comparisons: int
    average_advantage: float
    standard_error: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class DuplicateDealGameResult:
    deal_index: int
    rotation_index: int
    seed: int
    result: FullGameResult


@dataclass(frozen=True)
class DuplicateDealEvaluation:
    games: int
    deals: int
    rotations_per_deal: int
    agent_summaries: tuple[AgentFullGameSummary, ...]
    paired_comparisons: tuple[PairedComparison, ...]
    average_turns: float
    timeout_rate: float
    game_results: tuple[DuplicateDealGameResult, ...] = ()

    def paired(self, metric: str, baseline_agent: str) -> PairedComparison:
        for comparison in self.paired_comparisons:
            if comparison.metric == metric and comparison.baseline_agent == baseline_agent:
                return comparison
        raise KeyError(f"no paired {metric} comparison against {baseline_agent}")

    def summary(self, agent_name: str) -> AgentFullGameSummary:
        for summary in self.agent_summaries:
            if summary.agent_name == agent_name:
                return summary
        raise KeyError(f"no summary for {agent_name}")


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
    replay_valid = True
    replayed_events = 0
    for event in state.history:
        if event.is_pass:
            if replay_valid and event.player in possible:
                possible[event.player] -= simulated.public_legal_cards()
            if replay_valid:
                try:
                    simulated = simulated.after_play(event.player, None)
                    replayed_events += 1
                except ValueError as error:
                    if replayed_events:
                        raise ValueError(
                            f"cannot replay public history while deriving opponent possibilities: {event}"
                        ) from error
                    replay_valid = False
        else:
            assert event.card is not None
            for player in opponents:
                if player != event.player:
                    possible[player].discard(event.card)
            if replay_valid:
                try:
                    simulated = simulated.after_play(event.player, event.card)
                    replayed_events += 1
                except ValueError as error:
                    if replayed_events:
                        raise ValueError(
                            f"cannot replay public history while deriving opponent possibilities: {event}"
                        ) from error
                    replay_valid = False

    if replay_valid and state.history and canonical_table(simulated.table) != canonical_table(state.table):
        raise ValueError("public history does not reproduce the state's table")

    return possible


def exact_holder_marginals(
    possible_cards: Mapping[int, set[Card]],
    hand_counts: tuple[int | None, int | None, int | None, int | None] | None,
    hidden_cards: Iterable[Card] | None = None,
) -> dict[int, dict[Card, float]] | None:
    """P(player holds card) under a uniform prior over deals consistent with the known hand counts.

    Cards that the same set of opponents could hold are interchangeable, so assignments are
    counted per group of such cards with multinomials rather than card by card. Counts are exact
    integers until the final division.
    """
    if hand_counts is None:
        return None

    quota_players = tuple(player for player in possible_cards if hand_counts[player] is not None)
    if not quota_players:
        return None
    quotas = tuple(int(hand_counts[player]) for player in quota_players)  # type: ignore[arg-type]
    if any(count < 0 for count in quotas):
        return None

    cards = sorted(hidden_cards if hidden_cards is not None else set().union(*possible_cards.values()))
    groups: dict[tuple[int, ...], list[Card]] = {}
    for card in cards:
        holders = tuple(player for player in possible_cards if card in possible_cards[player])
        if not holders:
            return None
        groups.setdefault(holders, []).append(card)

    # For each group: every way to split its cards among its holders, as
    # (cards per holder, cards per quota player, number of labeled assignments).
    # Splits that already exceed a quota can never complete, so they are dropped.
    group_splits = [
        (
            holders,
            group_cards,
            [
                split
                for split in holder_splits(holders, len(group_cards), quota_players)
                if all(take <= limit for take, limit in zip(split[1], quotas))
            ],
        )
        for holders, group_cards in groups.items()
    ]

    def quota_usage_ways(skip: int | None) -> dict[tuple[int, ...], int]:
        usage: dict[tuple[int, ...], int] = {(0,) * len(quotas): 1}
        for index, (_, _, splits) in enumerate(group_splits):
            if index == skip:
                continue
            next_usage: dict[tuple[int, ...], int] = {}
            for used, ways in usage.items():
                for _, quota_take, split_ways in splits:
                    combined = tuple(map(add, used, quota_take))
                    if all(map(le, combined, quotas)):
                        next_usage[combined] = next_usage.get(combined, 0) + ways * split_ways
            usage = next_usage
        return usage

    total_ways = quota_usage_ways(None).get(quotas, 0)
    if total_ways == 0:
        return None

    per_card_counts: dict[int, dict[tuple[int, ...], int]] = {player: {} for player in possible_cards}
    for index, (holders, group_cards, splits) in enumerate(group_splits):
        others = quota_usage_ways(index)
        size = len(group_cards)
        for holder_take, quota_take, split_ways in splits:
            rest = others.get(tuple(limit - take for limit, take in zip(quotas, quota_take)), 0)
            if not rest:
                continue
            for holder, take in zip(holders, holder_take):
                if take:
                    # assignments where one specific card of the group goes to this holder
                    counts = per_card_counts[holder]
                    counts[holders] = counts.get(holders, 0) + rest * split_ways * take // size

    marginals: dict[int, dict[Card, float]] = {player: {} for player in possible_cards}
    for card in cards:
        holders = tuple(player for player in possible_cards if card in possible_cards[player])
        for holder in holders:
            count = per_card_counts[holder].get(holders, 0)
            if count:
                marginals[holder][card] = count / total_ways
    return marginals


@lru_cache(maxsize=4096)
def holder_splits(
    holders: tuple[int, ...],
    size: int,
    quota_players: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]:
    """Every split of `size` labeled cards among `holders`, with its multinomial count."""
    def compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first, *rest)

    splits = []
    for holder_take in compositions(size, len(holders)):
        ways = 1
        remaining = size
        for take in holder_take:
            ways *= comb(remaining, take)
            remaining -= take
        quota_take = tuple(
            sum(take for holder, take in zip(holders, holder_take) if holder == quota_player)
            for quota_player in quota_players
        )
        splits.append((holder_take, quota_take, ways))
    return tuple(splits)


def sample_hidden_deal(
    state: GameState,
    knowledge: PlayerKnowledge,
    rng: random.Random | None = None,
) -> HiddenDeal | None:
    rng = rng or random.Random()
    return HiddenDealSampler(state, knowledge).sample(rng)


def sample_hidden_deals(
    state: GameState,
    knowledge: PlayerKnowledge,
    count: int,
    rng: random.Random | None = None,
) -> tuple[HiddenDeal, ...]:
    rng = rng or random.Random()
    sampler = HiddenDealSampler(state, knowledge)
    deals: list[HiddenDeal] = []
    for _ in range(count):
        deal = sampler.sample(rng)
        if deal is None:
            break
        deals.append(deal)
    return tuple(deals)


def choose_hidden_deals_for_budget(
    state: GameState,
    knowledge: PlayerKnowledge,
    max_count: int,
    rng: random.Random | None = None,
) -> HiddenDealSelection:
    if max_count <= 0:
        return HiddenDealSelection((), False)

    enumeration = enumerate_hidden_deals(state, knowledge, max_deals=max_count + 1)
    if enumeration.exhaustive:
        return HiddenDealSelection(enumeration.deals, True)

    return HiddenDealSelection(sample_hidden_deals(state, knowledge, max_count, rng), False)


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
    require_solver_legal_move(state, knowledge, card, "exact imperfect-information")
    enumeration = enumerate_hidden_deals(state, knowledge, max_deals=max_deals)
    return evaluate_move_exact_imperfect_information_from_deals(state, knowledge, card, enumeration)


def require_solver_legal_move(state: GameState, knowledge: PlayerKnowledge, card: Card, evaluation: str) -> None:
    if state.current_player != knowledge.player:
        raise ValueError(f"{evaluation} evaluation requires knowledge.player to act")
    if card not in state.legal_moves(knowledge.hand):
        raise ValueError(f"cannot evaluate illegal move {card}")


def no_deal_exact_score(card: Card, exhaustive: bool) -> ExactImperfectInformationMoveScore:
    return ExactImperfectInformationMoveScore(
        card=card,
        expected_value=float("-inf"),
        expected_value_vector=(float("-inf"), float("-inf"), float("-inf"), float("-inf")),
        hidden_deals=0,
        exhaustive=exhaustive,
    )


def materialize_hidden_deal(
    state: GameState,
    knowledge: PlayerKnowledge,
    hidden_deal: HiddenDeal,
    public_legal_moves: tuple[Card, ...],
) -> tuple[dict[int, set[Card]], FullInformationState]:
    """Combine the solver's hand with a hidden deal and check it keeps the same legal moves."""
    hands = complete_hands(knowledge, hidden_deal)
    full_state = FullInformationState.from_hands(hands, state)
    if full_state.legal_moves() != public_legal_moves:
        raise ValueError(
            "materialized hidden deal changed the solver's legal move set: "
            f"public={labels(public_legal_moves)}, materialized={labels(full_state.legal_moves())}"
        )
    return hands, full_state


def evaluate_move_exact_imperfect_information_from_deals(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    enumeration: HiddenDealEnumeration,
    exact_cache: dict[ExactCacheKey, tuple[float, float, float, float]] | None = None,
    exact_stats: dict[str, int] | None = None,
) -> ExactImperfectInformationMoveScore:
    require_solver_legal_move(state, knowledge, card, "exact imperfect-information")
    if not enumeration.deals:
        return no_deal_exact_score(card, enumeration.exhaustive)

    exact_cache = exact_cache if exact_cache is not None else {}
    exact_stats = exact_stats if exact_stats is not None else new_search_stats()
    values: list[tuple[float, float, float, float]] = []
    outcome_counts = [0, 0, 0, 0]
    neutral_outcomes = 0
    public_legal_moves = tuple(state.legal_moves(knowledge.hand))
    for hidden_deal in enumeration.deals:
        _, full_state = materialize_hidden_deal(state, knowledge, hidden_deal, public_legal_moves)
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
    exact_cache: dict[ExactCacheKey, tuple[float, float, float, float]] = {}
    exact_stats = new_search_stats()
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
    chosen_move = choose_rational_move(knowledge.player, value_vectors) if scores else None
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


def evaluate_move_exact_information_limited_policy_from_deals(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    enumeration: HiddenDealEnumeration,
    max_turns: int = 200,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
) -> ExactImperfectInformationMoveScore:
    policy = normalize_information_limited_policy(policy)
    if policy != INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY:
        raise ValueError("exact information-limited policy EV currently requires deterministic heuristic-greedy policy")
    require_solver_legal_move(state, knowledge, card, "exact information-limited")
    if not enumeration.deals:
        return no_deal_exact_score(card, enumeration.exhaustive)

    outcome_counts = [0, 0, 0, 0]
    neutral_outcomes = 0
    public_legal_moves = tuple(state.legal_moves(knowledge.hand))
    for hidden_deal in enumeration.deals:
        hands, _ = materialize_hidden_deal(state, knowledge, hidden_deal, public_legal_moves)
        next_state, next_hands = apply_known_play(state, hands, knowledge.player, card)
        result = rollout_information_limited(
            next_state,
            next_hands,
            max_turns=max_turns,
            weights=weights,
            policy=policy,
        )
        if result.winner is None:
            neutral_outcomes += 1
        else:
            outcome_counts[result.winner] += 1

    expected_value_vector = tuple(count / len(enumeration.deals) for count in outcome_counts)
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


def evaluate_move_exact_information_limited_policy(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    max_deals: int | None = None,
    max_turns: int = 200,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
) -> ExactImperfectInformationMoveScore:
    enumeration = enumerate_hidden_deals(state, knowledge, max_deals=max_deals)
    return evaluate_move_exact_information_limited_policy_from_deals(
        state,
        knowledge,
        card,
        enumeration,
        max_turns=max_turns,
        weights=weights,
        policy=policy,
    )


def recommend_move_exact_information_limited_policy(
    state: GameState,
    knowledge: PlayerKnowledge,
    max_deals: int | None = None,
    max_turns: int = 200,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
) -> ExactImperfectInformationResult | None:
    policy = normalize_information_limited_policy(policy)
    legal = state.legal_moves(knowledge.hand)
    if not legal:
        return None

    enumeration = enumerate_hidden_deals(state, knowledge, max_deals=max_deals)
    scores = tuple(
        evaluate_move_exact_information_limited_policy_from_deals(
            state,
            knowledge,
            card,
            enumeration,
            max_turns=max_turns,
            weights=weights,
            policy=policy,
        )
        for card in legal
    )
    hidden_deals = scores[0].hidden_deals if scores else 0
    exhaustive = all(score.exhaustive for score in scores)
    value_vectors = {score.card: score.expected_value_vector for score in scores}
    best_value = max(score.expected_value for score in scores)
    best_moves = tuple(score.card for score in scores if score.expected_value == best_value)
    chosen_move = choose_rational_move(knowledge.player, value_vectors) if scores else None
    return ExactImperfectInformationResult(
        chosen_move=chosen_move,
        best_moves=best_moves,
        move_scores=scores,
        hidden_deals=hidden_deals,
        exhaustive=exhaustive,
        policy_name="exact_hidden_deal_expectation_with_information_limited_heuristic_greedy_policy",
        continuation_model="players use deterministic information-limited heuristic-greedy policy from their own hand and public evidence",
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
    immediate_win = immediate_winning_move(knowledge, legal)
    if immediate_win is not None:
        return terminal_win_monte_carlo_score(
            state,
            knowledge,
            immediate_win,
            weights,
            "oracle_greedy_full_information_immediate_win",
        )
    if len(legal) == 1:
        return forced_move_monte_carlo_score(legal[0], "oracle_greedy_full_information_forced_move")

    rng = rng or random.Random()
    hidden_deal_selection = choose_hidden_deals_for_budget(state, knowledge, samples_per_move, rng)
    results = [
        evaluate_move_monte_carlo_from_deals(
            state,
            knowledge,
            card,
            hidden_deal_selection.deals,
            max_turns,
            weights,
            hidden_deal_selection.exhaustive,
        )
        for card in legal
    ]
    return max(results, key=lambda result: result.score)


def rollout_move_score(
    card: Card,
    player: int,
    results: Sequence[RolloutResult],
    weights: StrategyWeights,
    policy_name: str,
    exhaustive: bool = False,
) -> MonteCarloMoveScore:
    """Aggregate rollouts after `card` into a score: weighted win rate + finish margin - timeout penalty."""
    samples = len(results)
    if samples == 0:
        return MonteCarloMoveScore(
            card=card,
            score=float("-inf"),
            win_rate=0.0,
            average_finish_margin=0.0,
            samples=0,
            policy_name=policy_name,
            exhaustive=exhaustive,
        )

    wins = sum(1 for result in results if result.winner == player)
    finish_margin_total = 0.0
    for result in results:
        finish_margin_total += finish_margin(result.final_hand_counts, player)
    win_rate = wins / samples
    average_finish_margin = finish_margin_total / samples
    timeout_rate = sum(1 for result in results if result.timed_out) / samples
    return MonteCarloMoveScore(
        card=card,
        score=(
            win_rate * weights.monte_carlo_win_rate_weight
            + average_finish_margin
            - timeout_rate * weights.monte_carlo_timeout_penalty
        ),
        win_rate=win_rate,
        average_finish_margin=average_finish_margin,
        samples=samples,
        average_turns=sum(result.turns_played for result in results) / samples,
        timeout_rate=timeout_rate,
        win_rate_standard_error=sqrt(win_rate * (1.0 - win_rate) / samples),
        policy_name=policy_name,
        exhaustive=exhaustive,
    )


def forced_move_monte_carlo_score(card: Card, policy_name: str) -> MonteCarloMoveScore:
    return MonteCarloMoveScore(
        card=card,
        score=0.0,
        win_rate=0.0,
        average_finish_margin=0.0,
        samples=0,
        average_turns=0.0,
        timeout_rate=0.0,
        win_rate_standard_error=0.0,
        policy_name=policy_name,
    )


def immediate_winning_move(knowledge: PlayerKnowledge, legal: Sequence[Card]) -> Card | None:
    if len(knowledge.hand) != 1:
        return None
    only_card = next(iter(knowledge.hand))
    return only_card if only_card in legal else None


def is_immediate_winning_move(state: GameState, knowledge: PlayerKnowledge, card: Card) -> bool:
    return immediate_winning_move(knowledge, state.legal_moves(knowledge.hand)) == card


def terminal_win_monte_carlo_score(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    weights: StrategyWeights,
    policy_name: str,
) -> MonteCarloMoveScore:
    average_finish_margin = 0.0
    if state.hand_counts is not None and all(count is not None for count in state.hand_counts):
        counts = list(state.hand_counts)
        counts[knowledge.player] = 0
        average_finish_margin = finish_margin(tuple(counts), knowledge.player)  # type: ignore[arg-type]
    score = weights.monte_carlo_win_rate_weight + average_finish_margin
    return MonteCarloMoveScore(
        card=card,
        score=score,
        win_rate=1.0,
        average_finish_margin=average_finish_margin,
        samples=0,
        average_turns=0.0,
        timeout_rate=0.0,
        win_rate_standard_error=0.0,
        policy_name=policy_name,
    )


def evaluate_move_monte_carlo(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    samples: int = 100,
    max_turns: int = 200,
    rng: random.Random | None = None,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> MonteCarloMoveScore:
    if is_immediate_winning_move(state, knowledge, card):
        return terminal_win_monte_carlo_score(
            state,
            knowledge,
            card,
            weights,
            "oracle_greedy_full_information_immediate_win",
        )
    rng = rng or random.Random()
    hidden_deal_selection = choose_hidden_deals_for_budget(state, knowledge, samples, rng)
    return evaluate_move_monte_carlo_from_deals(
        state,
        knowledge,
        card,
        hidden_deal_selection.deals,
        max_turns,
        weights,
        hidden_deal_selection.exhaustive,
    )


def evaluate_move_monte_carlo_from_deals(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    hidden_deals: Iterable[HiddenDeal],
    max_turns: int = 200,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    exhaustive: bool = False,
) -> MonteCarloMoveScore:
    if is_immediate_winning_move(state, knowledge, card):
        return terminal_win_monte_carlo_score(
            state,
            knowledge,
            card,
            weights,
            "oracle_greedy_full_information_immediate_win",
        )

    results: list[RolloutResult] = []
    for hidden_deal in hidden_deals:
        hands = complete_hands(knowledge, hidden_deal)
        if card not in hands[knowledge.player]:
            continue
        next_state, next_hands = apply_known_play(state, hands, knowledge.player, card)
        results.append(rollout_oracle(next_state, next_hands, max_turns, weights))

    return rollout_move_score(
        card, knowledge.player, results, weights, "oracle_greedy_full_information", exhaustive
    )


def complete_hands(knowledge: PlayerKnowledge, hidden_deal: HiddenDeal) -> dict[int, set[Card]]:
    hands = {player: set(hidden_deal.hand(player)) for player in range(4)}
    hands[knowledge.player] = set(knowledge.hand)
    return hands


def hand_masks_from_hands(hands: Mapping[int, Iterable[Card]]) -> tuple[int, int, int, int]:
    return tuple(card_set_mask(hands[player]) for player in range(4))  # type: ignore[return-value]


def mask_hand_count_tuple(hand_masks: Sequence[int]) -> tuple[int, int, int, int]:
    return tuple(mask.bit_count() for mask in hand_masks)  # type: ignore[return-value]


def first_empty_player_mask(hand_masks: Sequence[int]) -> int | None:
    for player, mask in enumerate(hand_masks):
        if mask == 0:
            return player
    return None


def mask_after_play(hand_masks: Sequence[int], player: int, card: Card | None) -> tuple[int, int, int, int]:
    if card is None:
        return tuple(hand_masks)  # type: ignore[return-value]
    next_masks = list(hand_masks)
    bit = card_bit(card)
    if not next_masks[player] & bit:
        raise ValueError(f"cannot play {card}: card is not in hand mask")
    next_masks[player] &= ~bit
    return tuple(next_masks)  # type: ignore[return-value]


def masks_after_known_play(
    state: GameState,
    hand_masks: Sequence[int],
    player: int,
    card: Card | None,
) -> tuple[GameState, tuple[int, int, int, int]]:
    """Mask counterpart of apply_known_play: play `card` and refresh public hand counts."""
    next_state = state_with_hand_counts_from_masks(state, hand_masks).after_play(player, card)
    next_hands = mask_after_play(hand_masks, player, card)
    return state_with_hand_counts_from_masks(next_state, next_hands), next_hands


def complete_hand_masks(knowledge: PlayerKnowledge, hidden_deal: HiddenDeal) -> tuple[int, int, int, int]:
    masks = [card_set_mask(hidden_deal.hand(player)) for player in range(4)]
    masks[knowledge.player] = card_set_mask(knowledge.hand)
    return tuple(masks)  # type: ignore[return-value]


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


def recommend_move_information_limited_monte_carlo(
    state: GameState,
    knowledge: PlayerKnowledge,
    samples_per_move: int = 100,
    max_turns: int = 200,
    rng: random.Random | None = None,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
    rationality: float = 1.0,
    policy_cache: InformationLimitedPolicyCache | None = None,
    rollout_cache: RolloutTranspositionCache | None = None,
) -> MonteCarloMoveScore | None:
    policy = normalize_information_limited_policy(policy)
    legal = state.legal_moves(knowledge.hand)
    if not legal:
        return None
    immediate_win = immediate_winning_move(knowledge, legal)
    if immediate_win is not None:
        return terminal_win_monte_carlo_score(
            state,
            knowledge,
            immediate_win,
            weights,
            f"{information_limited_policy_name(policy, rationality)}_immediate_win",
        )
    if len(legal) == 1:
        policy_name = f"{information_limited_policy_name(policy, rationality)}_forced_move"
        return forced_move_monte_carlo_score(legal[0], policy_name)

    rng = rng or random.Random()
    hidden_deal_selection = choose_hidden_deals_for_budget(state, knowledge, samples_per_move, rng)
    decision_policy_cache, decision_rollout_cache = decision_caches(policy, policy_cache, rollout_cache)
    results = [
        evaluate_move_information_limited_monte_carlo_from_deals(
            state,
            knowledge,
            card,
            hidden_deal_selection.deals,
            max_turns,
            weights,
            policy,
            rationality,
            rng,
            decision_policy_cache,
            decision_rollout_cache,
            hidden_deal_selection.exhaustive,
        )
        for card in legal
    ]
    return max(results, key=lambda result: result.score)


def evaluate_move_information_limited_monte_carlo(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    samples: int = 100,
    max_turns: int = 200,
    rng: random.Random | None = None,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
    rationality: float = 1.0,
    policy_cache: InformationLimitedPolicyCache | None = None,
    rollout_cache: RolloutTranspositionCache | None = None,
) -> MonteCarloMoveScore:
    policy = normalize_information_limited_policy(policy)
    if is_immediate_winning_move(state, knowledge, card):
        return terminal_win_monte_carlo_score(
            state,
            knowledge,
            card,
            weights,
            f"{information_limited_policy_name(policy, rationality)}_immediate_win",
        )
    rng = rng or random.Random()
    hidden_deal_selection = choose_hidden_deals_for_budget(state, knowledge, samples, rng)
    decision_policy_cache, decision_rollout_cache = decision_caches(policy, policy_cache, rollout_cache)
    return evaluate_move_information_limited_monte_carlo_from_deals(
        state,
        knowledge,
        card,
        hidden_deal_selection.deals,
        max_turns,
        weights,
        policy,
        rationality,
        rng,
        decision_policy_cache,
        decision_rollout_cache,
        hidden_deal_selection.exhaustive,
    )


def evaluate_move_information_limited_monte_carlo_from_deals(
    state: GameState,
    knowledge: PlayerKnowledge,
    card: Card,
    hidden_deals: Iterable[HiddenDeal],
    max_turns: int = 200,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
    rationality: float = 1.0,
    rng: random.Random | None = None,
    policy_cache: InformationLimitedPolicyCache | None = None,
    rollout_cache: RolloutTranspositionCache | None = None,
    exhaustive: bool = False,
) -> MonteCarloMoveScore:
    policy = normalize_information_limited_policy(policy)
    rng = rng or random.Random()
    policy_name = information_limited_policy_name(policy, rationality)
    if is_immediate_winning_move(state, knowledge, card):
        return terminal_win_monte_carlo_score(
            state,
            knowledge,
            card,
            weights,
            f"{policy_name}_immediate_win",
        )

    results: list[RolloutResult] = []
    for hidden_deal in hidden_deals:
        hand_masks = complete_hand_masks(knowledge, hidden_deal)
        if not hand_masks[knowledge.player] & card_bit(card):
            continue
        next_state, next_hands = masks_after_known_play(state, hand_masks, knowledge.player, card)
        results.append(
            rollout_information_limited_masks(
                next_state,
                next_hands,
                max_turns,
                weights,
                policy,
                rationality,
                rng,
                None,
                policy_cache,
                rollout_cache,
            )
        )

    return rollout_move_score(card, knowledge.player, results, weights, policy_name, exhaustive)


EXACT_LAYOUT_VOTE_POLICY = "exact_layout_vote"


def recommend_move_exact_layout_vote(
    state: GameState,
    knowledge: PlayerKnowledge,
    layouts: int = 64,
    rng: random.Random | None = None,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
) -> MonteCarloMoveScore | None:
    """Play the move that goes out first in the most hidden layouts, each solved exactly.

    Layouts are drawn uniformly from those consistent with the player's information
    (exhaustively when there are at most `layouts`). After the move every layout is
    solved with full-information rational play, so this is determinized search with
    an exact continuation. Ties go to the heuristic's preferred move.
    """
    legal = state.legal_moves(knowledge.hand)
    if not legal:
        return None
    immediate_win = immediate_winning_move(knowledge, legal)
    if immediate_win is not None:
        return terminal_win_monte_carlo_score(
            state, knowledge, immediate_win, weights, f"{EXACT_LAYOUT_VOTE_POLICY}_immediate_win"
        )
    if len(legal) == 1:
        return forced_move_monte_carlo_score(legal[0], f"{EXACT_LAYOUT_VOTE_POLICY}_forced_move")

    rng = rng or random.Random()
    selection = choose_hidden_deals_for_budget(state, knowledge, layouts, rng)
    wins = [0] * len(legal)
    for hidden_deal in selection.deals:
        layout_wins = FirstOutSolver(complete_hands(knowledge, hidden_deal)).move_wins(
            state.table, knowledge.player, legal
        )
        for index, won in enumerate(layout_wins):
            wins[index] += won

    model = build_opponent_model(state, knowledge)
    heuristic = [score_move(state, knowledge, card, model=model, weights=weights).score for card in legal]
    best = max(range(len(legal)), key=lambda index: (wins[index], heuristic[index]))
    samples = len(selection.deals)
    win_rate = wins[best] / samples if samples else 0.0
    return MonteCarloMoveScore(
        card=legal[best],
        score=win_rate * weights.monte_carlo_win_rate_weight,
        win_rate=win_rate,
        average_finish_margin=0.0,
        samples=samples,
        win_rate_standard_error=sqrt(win_rate * (1.0 - win_rate) / samples) if samples else 0.0,
        policy_name=EXACT_LAYOUT_VOTE_POLICY,
        exhaustive=selection.exhaustive,
    )


def recommend_move_perfect_information_rollout_ev(
    state: GameState,
    hands: Mapping[int, set[Card]],
    player: int,
    rollouts_per_move: int = 1,
    max_turns: int = 200,
    rng: random.Random | None = None,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
    rationality: float = 1.0,
    policy_cache: InformationLimitedPolicyCache | None = None,
    rollout_cache: RolloutTranspositionCache | None = None,
) -> MonteCarloMoveScore | None:
    policy = normalize_information_limited_policy(policy)
    legal = state.legal_moves(hands[player])
    if not legal:
        return None

    deck = frozenset(card for hand in hands.values() for card in hand) | frozenset(state.played_cards())
    knowledge = PlayerKnowledge(player, frozenset(hands[player]), deck=deck)
    policy_name = perfect_information_counterpart_policy_name(policy, rationality)
    immediate_win = immediate_winning_move(knowledge, legal)
    if immediate_win is not None:
        return terminal_win_monte_carlo_score(
            state,
            knowledge,
            immediate_win,
            weights,
            f"{policy_name}_immediate_win",
        )
    if len(legal) == 1:
        return forced_move_monte_carlo_score(legal[0], f"{policy_name}_forced_move")

    rng = rng or random.Random()
    decision_policy_cache, decision_rollout_cache = decision_caches(policy, policy_cache, rollout_cache)
    scores = tuple(
        evaluate_move_perfect_information_rollout_ev(
            state,
            hands,
            player,
            card,
            rollouts=rollouts_per_move,
            max_turns=max_turns,
            rng=rng,
            weights=weights,
            policy=policy,
            rationality=rationality,
            policy_cache=decision_policy_cache,
            rollout_cache=decision_rollout_cache,
        )
        for card in legal
    )
    return max(scores, key=lambda result: result.score)


def evaluate_move_perfect_information_rollout_ev(
    state: GameState,
    hands: Mapping[int, set[Card]],
    player: int,
    card: Card,
    rollouts: int = 1,
    max_turns: int = 200,
    rng: random.Random | None = None,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
    rationality: float = 1.0,
    policy_cache: InformationLimitedPolicyCache | None = None,
    rollout_cache: RolloutTranspositionCache | None = None,
) -> MonteCarloMoveScore:
    policy = normalize_information_limited_policy(policy)
    policy_name = perfect_information_counterpart_policy_name(policy, rationality)
    deck = frozenset(card_ for hand in hands.values() for card_ in hand) | frozenset(state.played_cards())
    knowledge = PlayerKnowledge(player, frozenset(hands[player]), deck=deck)
    if is_immediate_winning_move(state, knowledge, card):
        return terminal_win_monte_carlo_score(
            state,
            knowledge,
            card,
            weights,
            f"{policy_name}_immediate_win",
        )
    if card not in state.legal_moves(hands[player]):
        return MonteCarloMoveScore(
            card=card,
            score=float("-inf"),
            win_rate=0.0,
            average_finish_margin=0.0,
            samples=0,
            policy_name=policy_name,
        )

    rng = rng or random.Random()
    next_state, next_hands = masks_after_known_play(state, hand_masks_from_hands(hands), player, card)
    results = [
        rollout_information_limited_masks(
            next_state,
            next_hands,
            max_turns,
            weights,
            policy,
            rationality,
            rng,
            deck,
            policy_cache,
            rollout_cache,
        )
        for _ in range(max(1, rollouts))
    ]
    return rollout_move_score(card, player, results, weights, policy_name)


def decision_caches(
    policy: str,
    policy_cache: InformationLimitedPolicyCache | None,
    rollout_cache: RolloutTranspositionCache | None,
) -> tuple[InformationLimitedPolicyCache | None, RolloutTranspositionCache | None]:
    """Caches for one decision; caching is only sound for the deterministic greedy policy."""
    if not is_deterministic_heuristic_policy(policy):
        return policy_cache, None
    return (policy_cache if policy_cache is not None else {}), rollout_cache


def perfect_information_counterpart_policy_name(policy: str, rationality: float) -> str:
    return f"perfect_information_counterpart_{information_limited_policy_name(policy, rationality)}"


def rollout_information_limited(
    state: GameState,
    hands: Mapping[int, set[Card]],
    max_turns: int = 200,
    weights: StrategyWeights | Mapping[int, StrategyWeights] = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
    rationality: float = 1.0,
    rng: random.Random | None = None,
    policy_cache: InformationLimitedPolicyCache | None = None,
    rollout_cache: RolloutTranspositionCache | None = None,
) -> RolloutResult:
    return rollout_information_limited_masks(
        state,
        hand_masks_from_hands(hands),
        max_turns,
        weights,
        policy,
        rationality,
        rng,
        None,
        policy_cache,
        rollout_cache,
    )


def rollout_information_limited_masks(
    state: GameState,
    hand_masks: Sequence[int],
    max_turns: int = 200,
    weights: StrategyWeights | Mapping[int, StrategyWeights] = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
    rationality: float = 1.0,
    rng: random.Random | None = None,
    deck: frozenset[Card] | None = None,
    policy_cache: InformationLimitedPolicyCache | None = None,
    rollout_cache: RolloutTranspositionCache | None = None,
) -> RolloutResult:
    policy = normalize_information_limited_policy(policy)
    rng = rng or random.Random()
    current_hands = tuple(hand_masks)  # type: ignore[assignment]
    current_state = state_with_hand_counts_from_masks(state, current_hands)
    deck_mask = card_set_mask(deck) if deck is not None else (
        current_state.played_cards_mask() | _combined_hand_mask(current_hands)
    )
    use_rollout_cache = rollout_cache is not None and is_deterministic_heuristic_policy(policy)
    visited: list[tuple[RolloutTranspositionCacheKey, int]] = []

    for turns_played in range(max_turns + 1):
        if use_rollout_cache:
            cache_key = rollout_transposition_cache_key_from_masks(
                current_state,
                current_hands,
                deck_mask,
                weights,
                policy,
                rationality,
                max_turns - turns_played,
            )
            cached = rollout_cache.get(cache_key) if rollout_cache is not None else None
            if cached is not None:
                result = RolloutResult(
                    cached.winner,
                    turns_played + cached.turns_played,
                    cached.final_hand_counts,
                    cached.timed_out,
                )
                store_rollout_suffixes(rollout_cache, visited, result)
                return result
            visited.append((cache_key, turns_played))

        winner = first_empty_player_mask(current_hands)
        if winner is not None:
            result = RolloutResult(winner, turns_played, mask_hand_count_tuple(current_hands))
            store_rollout_suffixes(rollout_cache, visited, result)
            return result
        if turns_played == max_turns:
            break

        player = current_state.current_player
        card = choose_information_limited_move_from_mask(
            current_state,
            current_hands[player],
            player,
            weights_for_player(weights, player),
            policy,
            rationality,
            rng,
            deck_mask,
            policy_cache,
        )
        current_state, current_hands = masks_after_known_play(current_state, current_hands, player, card)

    result = RolloutResult(None, max_turns, mask_hand_count_tuple(current_hands), timed_out=True)
    store_rollout_suffixes(rollout_cache, visited, result)
    return result


def choose_information_limited_move(
    state: GameState,
    hand: Iterable[Card],
    player: int,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
    rationality: float = 1.0,
    rng: random.Random | None = None,
    deck: frozenset[Card] | None = None,
    cache: InformationLimitedPolicyCache | None = None,
) -> Card | None:
    return choose_information_limited_move_from_mask(
        state,
        card_set_mask(hand),
        player,
        weights,
        policy,
        rationality,
        rng,
        card_set_mask(deck) if deck is not None else None,
        cache,
    )


def choose_information_limited_move_from_mask(
    state: GameState,
    hand_mask: int,
    player: int,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    policy: str = INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY,
    rationality: float = 1.0,
    rng: random.Random | None = None,
    deck_mask: int | None = None,
    cache: InformationLimitedPolicyCache | None = None,
) -> Card | None:
    policy = normalize_information_limited_policy(policy)
    legal_mask = hand_mask & state.public_legal_mask()
    only_legal = maybe_only_card(legal_mask)
    if only_legal is not None:
        return only_legal
    legal = list(mask_to_cards(legal_mask))
    if not legal:
        return None

    deck = frozenset(mask_to_cards(deck_mask)) if deck_mask is not None else None
    cache_key = None
    if cache is not None and is_deterministic_heuristic_policy(policy):
        cache_key = information_limited_policy_cache_key(
            state,
            hand_mask,
            player,
            weights,
            policy,
            rationality,
            deck,
        )
        cached = cache.get(cache_key)
        if cached is not None or cache_key in cache:
            return cached

    hand = frozenset(mask_to_cards(hand_mask))
    knowledge = PlayerKnowledge(player, hand, deck=deck)
    model = build_opponent_model(state, knowledge)
    scores = tuple(score_move(state, knowledge, card, model=model, weights=weights) for card in legal)
    if is_deterministic_heuristic_policy(policy):
        choice = max(scores, key=lambda scored: scored.score).card
        if cache is not None and cache_key is not None:
            cache[cache_key] = choice
        return choice
    if policy == "softmax":
        rng = rng or random.Random()
        return choose_softmax_move(scores, rationality, rng)
    raise ValueError(f"unknown information-limited policy: {policy}")


def choose_softmax_move(scores: Iterable[MoveScore], rationality: float, rng: random.Random) -> Card:
    scored_moves = tuple(scores)
    if not scored_moves:
        raise ValueError("softmax policy requires at least one move")
    if rationality < 0.0:
        raise ValueError("softmax rationality cannot be negative")
    if rationality == 0.0:
        return rng.choice([scored.card for scored in scored_moves])

    best = max(scored.score for scored in scored_moves)
    weighted: list[tuple[Card, float]] = [
        (scored.card, exp((scored.score - best) * rationality))
        for scored in scored_moves
    ]
    total = sum(weight for _, weight in weighted)
    pick = rng.random() * total
    running = 0.0
    for card, weight in weighted:
        running += weight
        if pick <= running:
            return card
    return weighted[-1][0]


def information_limited_policy_name(policy: str, rationality: float) -> str:
    policy = normalize_information_limited_policy(policy)
    if policy == INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY:
        return "information_limited_heuristic_greedy"
    if policy == "softmax":
        return f"information_limited_softmax_heuristic_rationality_{rationality:g}"
    return f"information_limited_{policy}"


def normalize_information_limited_policy(policy: str) -> str:
    if policy == LEGACY_INFORMATION_LIMITED_GREEDY_POLICY:
        return INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY
    return policy


def is_deterministic_heuristic_policy(policy: str) -> bool:
    return normalize_information_limited_policy(policy) == INFORMATION_LIMITED_HEURISTIC_GREEDY_POLICY


InformationLimitedPolicyCacheKey = tuple[
    tuple[tuple[int | None, int | None], ...],
    tuple[int | None, int | None, int | None, int | None] | None,
    int,
    tuple[tuple[int, int], ...],
    int,
    int,
    StrategyWeights,
    str,
    float,
    int | None,
]
InformationLimitedPolicyCache = MutableMapping[InformationLimitedPolicyCacheKey, Card | None]


DEFAULT_CACHE_MAX_ENTRIES = 50_000


class BoundedLRUCache(OrderedDict):
    """Dict that evicts its least recently used entry beyond `max_entries` (0 means unbounded)."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_MAX_ENTRIES) -> None:
        super().__init__()
        self.max_entries = max_entries

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]

    def __setitem__(self, key, value) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if self.max_entries > 0:
            while len(self) > self.max_entries:
                self.popitem(last=False)


class BoundedInformationLimitedPolicyCache(BoundedLRUCache):
    pass


RolloutTranspositionCacheKey = tuple[
    tuple[tuple[int | None, int | None], ...],
    tuple[tuple[int, int], ...],
    int,
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    int,
    tuple[StrategyWeights, StrategyWeights, StrategyWeights, StrategyWeights],
    str,
    float,
    int,
]
RolloutTranspositionCache = MutableMapping[RolloutTranspositionCacheKey, RolloutResult]


class BoundedRolloutTranspositionCache(BoundedLRUCache):
    pass


def information_limited_policy_cache_key(
    state: GameState,
    hand_mask: int,
    player: int,
    weights: StrategyWeights,
    policy: str,
    rationality: float,
    deck: frozenset[Card] | None,
) -> InformationLimitedPolicyCacheKey:
    policy = normalize_information_limited_policy(policy)
    deck_mask = card_set_mask(deck) if deck is not None else None
    return (
        table_key_of(state),
        state.hand_counts,
        state.current_player,
        history_key_of(state),
        hand_mask,
        player,
        weights,
        policy,
        rationality,
        deck_mask,
    )


def table_key_of(state: GameState) -> tuple[tuple[int | None, int | None], ...]:
    return tuple((run.low, run.high) for run in canonical_table(state.table))


def history_key_of(state: GameState) -> tuple[tuple[int, int], ...]:
    return tuple((event.player, card_index(event.card) if event.card is not None else -1) for event in state.history)


def weights_for_player(
    weights: StrategyWeights | Mapping[int, StrategyWeights],
    player: int,
) -> StrategyWeights:
    if isinstance(weights, StrategyWeights):
        return weights
    return weights.get(player, DEFAULT_WEIGHTS)


def canonical_rollout_weights(
    weights: StrategyWeights | Mapping[int, StrategyWeights],
) -> tuple[StrategyWeights, StrategyWeights, StrategyWeights, StrategyWeights]:
    return tuple(weights_for_player(weights, player) for player in range(4))  # type: ignore[return-value]


def rollout_transposition_cache_key(
    state: GameState,
    hands: Mapping[int, set[Card]],
    deck: frozenset[Card],
    weights: StrategyWeights | Mapping[int, StrategyWeights],
    policy: str,
    rationality: float,
    remaining_turns: int,
) -> RolloutTranspositionCacheKey:
    return rollout_transposition_cache_key_from_masks(
        state,
        hand_masks_from_hands(hands),
        card_set_mask(deck),
        weights,
        policy,
        rationality,
        remaining_turns,
    )


def rollout_transposition_cache_key_from_masks(
    state: GameState,
    hand_masks: Sequence[int],
    deck_mask: int,
    weights: StrategyWeights | Mapping[int, StrategyWeights],
    policy: str,
    rationality: float,
    remaining_turns: int,
) -> RolloutTranspositionCacheKey:
    policy = normalize_information_limited_policy(policy)
    return (
        table_key_of(state),
        history_key_of(state),
        state.current_player,
        tuple(hand_masks),  # type: ignore[return-value]
        mask_hand_count_tuple(hand_masks),
        deck_mask,
        canonical_rollout_weights(weights),
        policy,
        rationality,
        remaining_turns,
    )


def store_rollout_suffixes(
    cache: RolloutTranspositionCache | None,
    visited: Iterable[tuple[RolloutTranspositionCacheKey, int]],
    result: RolloutResult,
) -> None:
    if cache is None:
        return
    for key, turns_at_state in visited:
        cache[key] = RolloutResult(
            result.winner,
            result.turns_played - turns_at_state,
            result.final_hand_counts,
            result.timed_out,
        )


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
    hands_after = hands_after_play(hands, player, card)
    next_state = state.after_play(player, card)
    before_legal = set(state.legal_moves(hands[player]))
    after_legal = set(next_state.legal_moves(hands_after[player]))
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


def state_with_hand_counts_from_masks(state: GameState, hand_masks: Sequence[int]) -> GameState:
    return GameState(
        table=state.table,
        hand_counts=mask_hand_count_tuple(hand_masks),
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


def _combined_hand_mask(hand_masks: Sequence[int]) -> int:
    mask = 0
    for hand_mask in hand_masks:
        mask |= hand_mask
    return mask


def finish_margin(hand_counts: tuple[int, int, int, int], player: int) -> float:
    opponents = [count for p, count in enumerate(hand_counts) if p != player]
    return min(opponents) - hand_counts[player]


def deal_random_hands(
    rng: random.Random | None = None,
    cards_per_suit: int = 13,
) -> dict[int, set[Card]]:
    rng = rng or random.Random()
    deck = sorted(reduced_deck(cards_per_suit), key=lambda card: (card.suit.value, card.rank))
    rng.shuffle(deck)
    cards_per_player = len(deck) // 4
    return {player: set(deck[player * cards_per_player : (player + 1) * cards_per_player]) for player in range(4)}


def initial_state_for_hands(hands: Mapping[int, set[Card]]) -> GameState:
    seven_hearts = Card(Suit.HEARTS, 7)
    starting_player = next(player for player, hand in hands.items() if seven_hearts in hand)
    return GameState(hand_counts=hand_count_tuple(hands), current_player=starting_player)


def replace_current_player(state: GameState, current_player: int) -> GameState:
    return GameState(
        table=state.table,
        hand_counts=state.hand_counts,
        current_player=current_player,
        history=state.history,
    )


def next_active_player(start_player: int, hands: Mapping[int, set[Card]]) -> int | None:
    for offset in range(4):
        player = (start_player + offset) % 4
        if hands[player]:
            return player
    return None


def unfinished_players(hands: Mapping[int, set[Card]]) -> tuple[int, ...]:
    return tuple(player for player in range(4) if hands[player])


def default_seat_rotations() -> tuple[tuple[int, int, int, int], ...]:
    return (
        (0, 1, 2, 3),
        (3, 0, 1, 2),
        (2, 3, 0, 1),
        (1, 2, 3, 0),
    )


def choose_full_game_agent_move(
    state: GameState,
    hands: Mapping[int, set[Card]],
    player: int,
    agent: FullGameAgent,
    rng: random.Random,
    deck: frozenset[Card] | None = None,
    policy_cache: InformationLimitedPolicyCache | None = None,
    rollout_cache: RolloutTranspositionCache | None = None,
) -> Card | None:
    legal = state.legal_moves(hands[player])
    if not legal:
        return None

    if agent.policy == "random":
        return rng.choice(legal)
    if agent.policy == "greedy_furthest_from_seven":
        return max(legal, key=lambda card: (abs(card.rank - 7), -card_index(card)))
    if agent.policy == "heuristic":
        knowledge = PlayerKnowledge(player, frozenset(hands[player]), deck=deck)
        model = build_opponent_model(state, knowledge)
        return max(
            legal,
            key=lambda card: score_move(state, knowledge, card, model=model, weights=agent.weights).score,
        )
    if agent.policy in SEVEN_RULE_POLICIES:
        knowledge = PlayerKnowledge(player, frozenset(hands[player]), deck=deck)
        return recommend_move_seven_rule(
            state, knowledge, agent.weights, hold_neighbour_sevens=SEVEN_RULE_POLICIES[agent.policy]
        )
    if agent.policy == "information_limited_monte_carlo":
        knowledge = PlayerKnowledge(player, frozenset(hands[player]), deck=deck)
        recommendation = recommend_move_information_limited_monte_carlo(
            state,
            knowledge,
            samples_per_move=agent.samples_per_move,
            max_turns=agent.rollout_max_turns,
            rng=rng,
            weights=agent.weights,
            policy=agent.rollout_policy,
            rationality=agent.rationality,
            policy_cache=policy_cache,
            rollout_cache=rollout_cache,
        )
        return recommendation.card if recommendation is not None else None
    if agent.policy == "perfect_information_counterpart_oracle":
        recommendation = recommend_move_perfect_information_rollout_ev(
            state,
            hands,
            player,
            rollouts_per_move=agent.samples_per_move,
            max_turns=agent.rollout_max_turns,
            rng=rng,
            weights=agent.weights,
            policy=agent.rollout_policy,
            rationality=agent.rationality,
            policy_cache=policy_cache,
            rollout_cache=rollout_cache,
        )
        return recommendation.card if recommendation is not None else None
    if agent.policy == EXACT_LAYOUT_VOTE_POLICY:
        knowledge = PlayerKnowledge(player, frozenset(hands[player]), deck=deck)
        recommendation = recommend_move_exact_layout_vote(
            state, knowledge, layouts=agent.samples_per_move, rng=rng, weights=agent.weights
        )
        return recommendation.card if recommendation is not None else None
    if agent.policy == "oracle_greedy":
        return choose_oracle_move(state, hands, player, agent.weights)

    raise ValueError(f"unknown full-game agent policy: {agent.policy}")


def trace_mc_heuristic_decision(
    state: GameState,
    hands: Mapping[int, set[Card]],
    player: int,
    agent: FullGameAgent,
    rng: random.Random,
    deck: frozenset[Card],
    policy_cache: InformationLimitedPolicyCache,
    rollout_cache: RolloutTranspositionCache,
    turn: int,
    deal_index: int | None = None,
    rotation_index: int | None = None,
) -> tuple[Card | None, DecisionTrace | None]:
    legal = state.legal_moves(hands[player])
    if not legal:
        return None, None

    knowledge = PlayerKnowledge(player, frozenset(hands[player]), deck=deck)
    model = build_opponent_model(state, knowledge)
    heuristic_scores = tuple(
        score_move(state, knowledge, card, model=model, weights=agent.weights)
        for card in legal
    )
    heuristic_best = max(heuristic_scores, key=lambda scored: scored.score)
    mc_result = recommend_move_information_limited_monte_carlo(
        state,
        knowledge,
        samples_per_move=agent.samples_per_move,
        max_turns=agent.rollout_max_turns,
        rng=rng,
        weights=agent.weights,
        policy=agent.rollout_policy,
        rationality=agent.rationality,
        policy_cache=policy_cache,
        rollout_cache=rollout_cache,
    )
    if mc_result is None:
        return None, None

    trace = DecisionTrace(
        deal_index=deal_index,
        rotation_index=rotation_index,
        turn=turn,
        player=player,
        agent_name=agent.name,
        legal_moves=tuple(legal),
        hand=tuple(sorted(hands[player])),
        table_key=table_key_of(state),
        heuristic_card=heuristic_best.card,
        heuristic_score=heuristic_best.score,
        heuristic_reasons=heuristic_best.reasons,
        mc_card=mc_result.card,
        mc_score=mc_result.score,
        mc_win_rate=mc_result.win_rate,
        mc_average_finish_margin=mc_result.average_finish_margin,
        mc_samples=mc_result.samples,
        disagreed=mc_result.card != heuristic_best.card,
    )
    return mc_result.card, trace


def simulate_full_game_to_completion(
    hands: Mapping[int, set[Card]],
    agents_by_seat: Sequence[FullGameAgent],
    rng: random.Random | None = None,
    max_turns: int = 1000,
    deck: frozenset[Card] | None = None,
    trace_mc_heuristic: bool = False,
    deal_index: int | None = None,
    rotation_index: int | None = None,
) -> FullGameResult:
    if len(agents_by_seat) != 4:
        raise ValueError("full-game simulation requires exactly four agents")

    rng = rng or random.Random()
    current_hands = {player: set(hand) for player, hand in hands.items()}
    deck = deck or frozenset(card for hand in current_hands.values() for card in hand)
    state = initial_state_for_hands(current_hands)
    finish_order: list[int] = []
    first_out_counts: tuple[int, int, int, int] | None = None
    decision_traces: list[DecisionTrace] = []
    decision_counts = [FullGameDecisionCounts() for _ in range(4)]
    policy_cache: InformationLimitedPolicyCache = BoundedInformationLimitedPolicyCache()
    rollout_cache: RolloutTranspositionCache = BoundedRolloutTranspositionCache()

    for turns_played in range(max_turns + 1):
        active = unfinished_players(current_hands)
        if len(active) <= 1:
            if active:
                finish_order.append(active[0])
            return finalized_full_game_result(
                agents_by_seat,
                finish_order,
                hand_count_tuple(current_hands),
                turns_played,
                timed_out=False,
                decision_traces=decision_traces,
                decision_counts_by_seat=decision_counts,
                first_out_hand_counts=first_out_counts,
            )
        if turns_played == max_turns:
            break

        player = next_active_player(state.current_player, current_hands)
        if player is None:
            break
        if player != state.current_player:
            state = replace_current_player(state, player)

        agent = agents_by_seat[player]
        legal = state.legal_moves(current_hands[player])
        immediate_win = bool(legal and len(current_hands[player]) == 1)
        sampled_mc = (
            agent.policy in ("information_limited_monte_carlo", EXACT_LAYOUT_VOTE_POLICY)
            and len(legal) > 1
            and not immediate_win
        )
        counts = decision_counts[player]
        decision_counts[player] = FullGameDecisionCounts(
            turns=counts.turns + 1,
            forced_passes=counts.forced_passes + (not legal),
            single_legal_moves=counts.single_legal_moves + (len(legal) == 1),
            immediate_wins=counts.immediate_wins + immediate_win,
            sampled_mc_decisions=counts.sampled_mc_decisions + sampled_mc,
        )
        if (
            trace_mc_heuristic
            and agent.name == DEFAULT_MONTE_CARLO_AGENT_NAME
            and agent.policy == "information_limited_monte_carlo"
        ):
            card, trace = trace_mc_heuristic_decision(
                state,
                current_hands,
                player,
                agent,
                rng,
                deck,
                policy_cache,
                rollout_cache,
                turns_played,
                deal_index,
                rotation_index,
            )
            if trace is not None:
                decision_traces.append(trace)
        else:
            card = choose_full_game_agent_move(
                state,
                current_hands,
                player,
                agent,
                rng,
                deck,
                policy_cache,
                rollout_cache,
            )
        state, current_hands = apply_known_play(state, current_hands, player, card)
        if not current_hands[player] and player not in finish_order:
            finish_order.append(player)
            if first_out_counts is None:
                first_out_counts = hand_count_tuple(current_hands)

        next_player = next_active_player(state.current_player, current_hands)
        if next_player is not None and next_player != state.current_player:
            state = replace_current_player(state, next_player)

    counts = hand_count_tuple(current_hands)
    timed_out_order = finish_order + sorted(
        (player for player in range(4) if player not in finish_order),
        key=lambda player: (counts[player], player),
    )
    return finalized_full_game_result(
        agents_by_seat,
        timed_out_order,
        counts,
        max_turns,
        timed_out=True,
        decision_traces=decision_traces,
        decision_counts_by_seat=decision_counts,
        first_out_hand_counts=first_out_counts,
    )


def finalized_full_game_result(
    agents_by_seat: Sequence[FullGameAgent],
    finish_order: Sequence[int],
    final_hand_counts: tuple[int, int, int, int],
    turns_played: int,
    timed_out: bool,
    decision_traces: Sequence[DecisionTrace] = (),
    decision_counts_by_seat: Sequence[FullGameDecisionCounts] = (FullGameDecisionCounts(),) * 4,
    first_out_hand_counts: tuple[int, int, int, int] | None = None,
) -> FullGameResult:
    ranks = [4, 4, 4, 4]
    seen: set[int] = set()
    normalized_order: list[int] = []
    for player in finish_order:
        if player not in seen:
            normalized_order.append(player)
            seen.add(player)
    for player in range(4):
        if player not in seen:
            normalized_order.append(player)

    for rank, player in enumerate(normalized_order, start=1):
        ranks[player] = rank

    completed_traces = tuple(
        replace(
            trace,
            final_rank=ranks[trace.player],
            final_cards_left=final_hand_counts[trace.player],
            winner=normalized_order[0] if normalized_order else None,
        )
        for trace in decision_traces
    )

    return FullGameResult(
        agent_names_by_seat=tuple(agent.name for agent in agents_by_seat),  # type: ignore[arg-type]
        winner=normalized_order[0] if normalized_order else None,
        finish_order=tuple(normalized_order),
        ranks_by_seat=tuple(ranks),  # type: ignore[arg-type]
        final_hand_counts=final_hand_counts,
        turns_played=turns_played,
        timed_out=timed_out,
        decision_traces=completed_traces,
        decision_counts_by_seat=tuple(decision_counts_by_seat),  # type: ignore[arg-type]
        first_out_hand_counts=first_out_hand_counts or final_hand_counts,
    )


DuplicateDealGameJob = tuple[
    int,
    int,
    int,
    dict[int, set[Card]],
    tuple[int, int, int, int],
    tuple[FullGameAgent, ...],
    int,
    frozenset[Card],
    bool,
]


def make_duplicate_deal_game_jobs(
    agents: Sequence[FullGameAgent],
    dealt_hands: Sequence[dict[int, set[Card]]],
    rotations: Sequence[Sequence[int]],
    rng: random.Random,
    max_turns: int,
    deck: frozenset[Card],
    trace_mc_heuristic: bool = False,
) -> tuple[DuplicateDealGameJob, ...]:
    jobs: list[DuplicateDealGameJob] = []
    for deal_index, deal in enumerate(dealt_hands, start=1):
        for rotation_index, rotation in enumerate(rotations, start=1):
            jobs.append(
                (
                    deal_index,
                    rotation_index,
                    rng.getrandbits(63),
                    deal,
                    tuple(rotation),  # type: ignore[arg-type]
                    tuple(agents),
                    max_turns,
                    deck,
                    trace_mc_heuristic,
                )
            )
    return tuple(jobs)


def simulate_duplicate_deal_game_job(job: DuplicateDealGameJob) -> DuplicateDealGameResult:
    deal_index, rotation_index, seed, deal, rotation, agents, max_turns, deck, trace_mc_heuristic = job
    agents_by_seat = tuple(agents[index] for index in rotation)
    result = simulate_full_game_to_completion(
        deal,
        agents_by_seat,
        random.Random(seed),
        max_turns=max_turns,
        deck=deck,
        trace_mc_heuristic=trace_mc_heuristic,
        deal_index=deal_index,
        rotation_index=rotation_index,
    )
    return DuplicateDealGameResult(deal_index, rotation_index, seed, result)


def standard_error(values: Sequence[float]) -> float:
    count = len(values)
    if count <= 1:
        return 0.0
    average = sum(values) / count
    variance = sum((value - average) ** 2 for value in values) / (count - 1)
    return sqrt(variance / count)


def evaluate_duplicate_deal_seat_rotation(
    agents: Sequence[FullGameAgent],
    deals: int,
    rng: random.Random | None = None,
    max_turns: int = 1000,
    rotations: Sequence[Sequence[int]] | None = None,
    primary_agent_name: str = DEFAULT_MONTE_CARLO_AGENT_NAME,
    progress_callback: Callable[[int, int, int, int, FullGameResult], None] | None = None,
    cards_per_suit: int = 13,
    workers: int = 1,
    trace_mc_heuristic: bool = False,
) -> DuplicateDealEvaluation:
    if len(agents) != 4:
        raise ValueError("duplicate-deal evaluation requires exactly four agents")

    rng = rng or random.Random()
    rotations = tuple(tuple(rotation) for rotation in (rotations or default_seat_rotations()))
    deck = frozenset(reduced_deck(cards_per_suit))
    dealt_hands = [deal_random_hands(rng, cards_per_suit=cards_per_suit) for _ in range(deals)]
    jobs = make_duplicate_deal_game_jobs(
        agents,
        dealt_hands,
        rotations,
        rng,
        max_turns,
        deck,
        trace_mc_heuristic,
    )
    game_results: list[DuplicateDealGameResult] = []

    def record(game_result: DuplicateDealGameResult) -> None:
        game_results.append(game_result)
        if progress_callback is not None:
            progress_callback(
                len(game_results),
                len(jobs),
                game_result.deal_index,
                game_result.rotation_index,
                game_result.result,
            )

    if workers <= 1:
        for job in jobs:
            record(simulate_duplicate_deal_game_job(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(simulate_duplicate_deal_game_job, job) for job in jobs]
            for future in as_completed(futures):
                record(future.result())

    game_results.sort(key=lambda game_result: (game_result.deal_index, game_result.rotation_index))

    return summarize_duplicate_deal_results(
        [game_result.result for game_result in game_results],
        deals=deals,
        rotations_per_deal=len(rotations),
        primary_agent_name=primary_agent_name,
        game_results=game_results,
    )


DECISION_COUNT_FIELDS = ("turns", "forced_passes", "single_legal_moves", "immediate_wins", "sampled_mc_decisions")


def seat_outcomes(result: FullGameResult, seat: int) -> dict[str, float]:
    first_out_counts = result.first_out_hand_counts or result.final_hand_counts
    return {
        "win": 1.0 if result.winner == seat else 0.0,
        "cards_at_first_out": float(first_out_counts[seat]),
        "rank": float(result.ranks_by_seat[seat]),
        "cards_left": float(result.final_hand_counts[seat]),
    }


def paired_advantage(metric: str, primary: Mapping[str, float], baseline: Mapping[str, float]) -> float:
    if metric == "first_out_win":
        return primary["win"] - baseline["win"]
    return baseline[metric] - primary[metric]


def summarize_duplicate_deal_results(
    results: Sequence[FullGameResult],
    deals: int,
    rotations_per_deal: int,
    primary_agent_name: str = DEFAULT_MONTE_CARLO_AGENT_NAME,
    game_results: Sequence[DuplicateDealGameResult] = (),
) -> DuplicateDealEvaluation:
    outcomes_by_agent: dict[str, dict[str, list[float]]] = {}
    decision_totals: dict[str, dict[str, int]] = {}
    paired: dict[tuple[str, str], list[float]] = {}

    for result in results:
        seat_values = [seat_outcomes(result, seat) for seat in range(4)]
        for seat, agent_name in enumerate(result.agent_names_by_seat):
            agent_outcomes = outcomes_by_agent.setdefault(agent_name, {})
            for name, value in seat_values[seat].items():
                agent_outcomes.setdefault(name, []).append(value)
            totals = decision_totals.setdefault(agent_name, dict.fromkeys(DECISION_COUNT_FIELDS, 0))
            for name in DECISION_COUNT_FIELDS:
                totals[name] += getattr(result.decision_counts_by_seat[seat], name)

        if primary_agent_name not in result.agent_names_by_seat:
            continue
        primary = seat_values[result.agent_names_by_seat.index(primary_agent_name)]
        for seat, agent_name in enumerate(result.agent_names_by_seat):
            if agent_name == primary_agent_name:
                continue
            for metric in PAIRED_METRICS:
                paired.setdefault((metric, agent_name), []).append(
                    paired_advantage(metric, primary, seat_values[seat])
                )

    summaries = []
    for agent_name, outcomes in sorted(outcomes_by_agent.items()):
        seats = len(outcomes["win"])
        summaries.append(
            AgentFullGameSummary(
                agent_name=agent_name,
                seats_played=seats,
                win_rate=sum(outcomes["win"]) / seats,
                win_rate_standard_error=standard_error(outcomes["win"]),
                average_cards_at_first_out=sum(outcomes["cards_at_first_out"]) / seats,
                cards_at_first_out_standard_error=standard_error(outcomes["cards_at_first_out"]),
                average_rank=sum(outcomes["rank"]) / seats,
                rank_standard_error=standard_error(outcomes["rank"]),
                average_cards_left=sum(outcomes["cards_left"]) / seats,
                cards_left_standard_error=standard_error(outcomes["cards_left"]),
                **decision_totals[agent_name],
            )
        )

    comparisons = []
    def comparison_order(item: tuple[tuple[str, str], list[float]]) -> tuple[int, str]:
        (metric, baseline), _ = item
        return PAIRED_METRICS.index(metric), baseline

    for (metric, baseline), differences in sorted(paired.items(), key=comparison_order):
        average = sum(differences) / len(differences)
        error = standard_error(differences)
        comparisons.append(
            PairedComparison(
                primary_agent=primary_agent_name,
                baseline_agent=baseline,
                metric=metric,
                comparisons=len(differences),
                average_advantage=average,
                standard_error=error,
                ci95_low=average - 1.96 * error,
                ci95_high=average + 1.96 * error,
            )
        )

    games = len(results)
    return DuplicateDealEvaluation(
        games=games,
        deals=deals,
        rotations_per_deal=rotations_per_deal,
        agent_summaries=tuple(summaries),
        paired_comparisons=tuple(comparisons),
        average_turns=sum(result.turns_played for result in results) / games if games else 0.0,
        timeout_rate=sum(1 for result in results if result.timed_out) / games if games else 0.0,
        game_results=tuple(game_results),
    )


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


SEVEN_GAP_POLICY = "heuristic_seven_gap"
SEVEN_RULE_POLICY = "heuristic_seven_rule"
SEVEN_PAIR_RULE_POLICY = "heuristic_seven_pair_rule"
SEVEN_RULE_POLICIES = {SEVEN_GAP_POLICY: None, SEVEN_RULE_POLICY: "any", SEVEN_PAIR_RULE_POLICY: "pair"}


def seven_kind(card: Card, hand: Iterable[Card]) -> str | None:
    """For a 7: 'gap' if you hold another card of its suit but not its neighbours, 'neighbour' if
    you hold a 6 or 8 of it, 'lone' otherwise. None for other cards."""
    if card.rank != 7:
        return None
    same = [other.rank for other in hand if other.suit == card.suit and other.rank != 7]
    if not same:
        return "lone"
    return "neighbour" if any(abs(rank - 7) == 1 for rank in same) else "gap"


def recommend_move_seven_rule(
    state: GameState,
    knowledge: PlayerKnowledge,
    weights: StrategyWeights = DEFAULT_WEIGHTS,
    hold_neighbour_sevens: str | None = "any",
) -> Card | None:
    """The heuristic with one rule about sevens, found by comparing it with exact best responses.

    Open a 7 when you hold a card of that suit beyond a gap (others must fill the gap before
    you can play it). hold_neighbour_sevens: None never holds; "any" avoids opening a 7 whose
    neighbour you hold while another card is playable; "pair" does so only when that neighbour
    is your only other card of the suit and the alternative is not a 7. Ties go to the
    heuristic score.
    """
    legal = state.legal_moves(knowledge.hand)
    if not legal:
        return None
    model = build_opponent_model(state, knowledge)
    scores = {card: score_move(state, knowledge, card, model, weights).score for card in legal}
    kinds = {card: seven_kind(card, knowledge.hand) for card in legal}
    candidates = [card for card in legal if kinds[card] == "gap"] or legal
    if hold_neighbour_sevens == "any":
        candidates = [card for card in candidates if kinds[card] != "neighbour"] or candidates
    elif hold_neighbour_sevens == "pair":
        def held_pair(card: Card) -> bool:
            return kinds[card] == "neighbour" and sum(1 for other in knowledge.hand if other.suit == card.suit) == 2
        non_sevens = [card for card in candidates if kinds[card] is None]
        if non_sevens:
            candidates = [card for card in candidates if not held_pair(card)] or candidates
    elif hold_neighbour_sevens is not None:
        raise ValueError(f"unknown hold_neighbour_sevens: {hold_neighbour_sevens}")
    return max(candidates, key=lambda card: scores[card])


def parse_cards(text: str) -> frozenset[Card]:
    """Parse a space-separated card list such as "7H 10S AC"."""
    return frozenset(Card.parse(part) for part in text.split())


def labels(cards: Iterable[Card]) -> str:
    ordered = sorted(cards)
    if not ordered:
        return "[]"
    return "[" + ", ".join(card.label() for card in ordered) + "]"
