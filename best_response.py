"""Exact best response in Sevens against fixed, known opponent policies.

The hero sees only their own hand and the public history. With the three
opponents' policies fixed, the hero's problem is a single-agent decision problem
over the hidden layouts. The belief at any point is every layout consistent with
what the hero has seen, each weighted by the prior (uniform over deals) times the
probability that the opponents' policies would have produced the observed
actions. Expectimax over the hero's information states then gives the exact
best-response probability of going out first, and the same recursion gives the
exact value of any other hero policy, so regret is measured without sampling
noise.

The root belief holds every split of the unseen cards, so this is for reduced
decks (3-5 cards per suit) or, via `BestResponseSolver.from_position`, late
full-deck positions, where the belief is the exact posterior given the history.
"""
from __future__ import annotations

import random
import zlib
from collections import defaultdict
from itertools import combinations
from typing import Callable, Iterable, Mapping, NamedTuple

from seven_hearts import (
    Card,
    MoveEvent,
    FirstOutSolver,
    FullGameAgent,
    GameState,
    PlayerKnowledge,
    Suit,
    build_opponent_model,
    card_bit,
    card_set_mask,
    choose_full_game_agent_move,
    mask_to_cards,
    recommend_move_exact_layout_vote,
    recommend_move_information_limited_monte_carlo,
    reduced_deck,
    score_move,
    trailing_pass_count,
    DEFAULT_WEIGHTS,
    SEVEN_GAP_POLICY,
    SEVEN_RULE_POLICY,
    SEVEN_PAIR_RULE_POLICY,
)

Layout = tuple[int, int, int, int]  # opponents' hand masks by seat; the hero's slot is 0
Belief = Mapping[Layout, float]
Distribution = tuple[tuple[Card | None, float], ...]
OpponentPolicy = Callable[[GameState, int, int], Distribution]
HeroPolicy = Callable[[GameState, int, int, Belief], Distribution]

EXHAUSTIVE = 10**9  # layout budget that makes the sampled agents enumerate every layout
OPPONENT_CACHE_LIMIT = 500_000  # entries; the cache is cleared when full so memory stays bounded


class BudgetExceeded(Exception):
    """An evaluation reached more hero decisions than `BestResponseSolver.node_budget`."""

OPPONENT_AGENTS = {
    "greedy": FullGameAgent("Greedy", "greedy_furthest_from_seven"),
    "heuristic": FullGameAgent("Heuristic", "heuristic"),
    "seven_gap": FullGameAgent("Seven Gap", SEVEN_GAP_POLICY),
    "seven_rule": FullGameAgent("Seven Rule", SEVEN_RULE_POLICY),
    "seven_pair_rule": FullGameAgent("Seven Pair Rule", SEVEN_PAIR_RULE_POLICY),
}


def opponent_policy(name: str, deck: Iterable[Card]) -> OpponentPolicy:
    """The evaluation's fixed agents as action distributions: random, greedy, heuristic."""
    deck = frozenset(deck)
    if name == "random":
        def uniform(state: GameState, player: int, hand_mask: int) -> Distribution:
            legal = mask_to_cards(hand_mask & state.public_legal_mask())
            if not legal:
                return ((None, 1.0),)
            return tuple((card, 1.0 / len(legal)) for card in legal)
        uniform.cheap = True  # type: ignore[attr-defined]
        return uniform
    if name not in OPPONENT_AGENTS:
        raise ValueError(f"unknown opponent policy: {name}")
    agent = OPPONENT_AGENTS[name]
    rng = random.Random(0)  # unused: these agents are deterministic

    def deterministic(state: GameState, player: int, hand_mask: int) -> Distribution:
        hands = {player: set(mask_to_cards(hand_mask))}
        return ((choose_full_game_agent_move(state, hands, player, agent, rng, deck=deck), 1.0),)
    if name == "greedy":
        deterministic.cheap = True  # type: ignore[attr-defined]  # one max over the legal cards
    return deterministic


def hero_policy(name: str, deck: Iterable[Card]) -> HeroPolicy:
    """Hero policies to score against the best response.

    random / greedy / heuristic: the evaluation agents.
    seven_gap / seven_rule / seven_pair_rule: the heuristic plus a sevens rule
        (seven_hearts.recommend_move_seven_rule).
    exact_vote: Exact Vote over every layout consistent with public evidence
        (uniform prior), each solved with omniscient rational play.
    posterior_vote: the same vote, but weighting layouts by the true posterior
        given the opponents' policies. Its gap to the best response is the cost
        of the omniscient continuation alone; exact_vote's further gap to it is
        the cost of the uniform belief.
    mc: Monte Carlo over every consistent layout with heuristic rollouts.
    """
    deck = frozenset(deck)
    if name in ("random", *OPPONENT_AGENTS):
        base = opponent_policy(name, deck)
        return lambda state, player, hand_mask, belief: base(state, player, hand_mask)
    if name.startswith("exact_vote:"):
        return sampled_exact_vote(deck, int(name.split(":")[1]))
    if name == "exact_vote":
        def exact_vote(state: GameState, player: int, hand_mask: int, belief: Belief) -> Distribution:
            knowledge = PlayerKnowledge(player, frozenset(mask_to_cards(hand_mask)), deck=deck)
            return ((recommend_move_exact_layout_vote(state, knowledge, layouts=EXHAUSTIVE).card, 1.0),)
        return exact_vote
    if name == "mc":
        def monte_carlo(state: GameState, player: int, hand_mask: int, belief: Belief) -> Distribution:
            knowledge = PlayerKnowledge(player, frozenset(mask_to_cards(hand_mask)), deck=deck)
            recommendation = recommend_move_information_limited_monte_carlo(
                state, knowledge, samples_per_move=EXHAUSTIVE, rng=random.Random(0)
            )
            return ((recommendation.card, 1.0),)
        return monte_carlo
    if name == "posterior_vote":
        return posterior_vote(deck)
    raise ValueError(f"unknown hero policy: {name}")


def sampled_exact_vote(deck: frozenset[Card], layouts: int) -> HeroPolicy:
    """Exact Vote over `layouts` sampled layouts, as the full-deck agent plays it.

    The sample is seeded from the hero's information state, so the policy is a
    fixed function of what the hero knows and exact evaluation stays well defined.
    """
    def vote(state: GameState, player: int, hand_mask: int, belief: Belief) -> Distribution:
        knowledge = PlayerKnowledge(player, frozenset(mask_to_cards(hand_mask)), deck=deck)
        key = ",".join(f"{event.player}{event.card}" for event in state.history) + f"|{hand_mask}"
        rng = random.Random(zlib.crc32(key.encode()))
        return ((recommend_move_exact_layout_vote(state, knowledge, layouts=layouts, rng=rng).card, 1.0),)
    return vote


def posterior_vote(deck: frozenset[Card]) -> HeroPolicy:
    def vote(state: GameState, player: int, hand_mask: int, belief: Belief) -> Distribution:
        legal = list(mask_to_cards(hand_mask & state.public_legal_mask()))
        if len(legal) == 1:
            return ((legal[0], 1.0),)
        hero_cards = mask_to_cards(hand_mask)
        wins = [0.0] * len(legal)
        for layout, weight in belief.items():
            hands = {seat: mask_to_cards(mask) for seat, mask in enumerate(layout)}
            hands[player] = hero_cards
            for index, won in enumerate(FirstOutSolver(hands).move_wins(state.table, player, legal)):
                if won:
                    wins[index] += weight
        knowledge = PlayerKnowledge(player, frozenset(hero_cards), deck=deck)
        model = build_opponent_model(state, knowledge)
        heuristic = [score_move(state, knowledge, card, model=model, weights=DEFAULT_WEIGHTS).score for card in legal]
        best = max(range(len(legal)), key=lambda index: (wins[index], heuristic[index]))
        return ((legal[best], 1.0),)
    return vote


class BestResponseSolver:
    """Exact first-out-win probability for one hero hand against fixed opponents.

    `value()` is the best response; `value(policy)` is the hero following `policy`.
    Both average over every deal of the unseen cards consistent with the hero's hand.
    """

    def __init__(
        self,
        hero_hand: Iterable[Card],
        opponents: Mapping[int, OpponentPolicy],
        hero: int = 0,
        deck: Iterable[Card] | None = None,
    ) -> None:
        hero_hand = frozenset(hero_hand)
        self.hero = hero
        self.hero_mask = card_set_mask(hero_hand)
        self.deck = frozenset(deck) if deck is not None else frozenset(reduced_deck(len(hero_hand)))
        if not hero_hand <= self.deck or len(self.deck) != 4 * len(hero_hand):
            raise ValueError("hero hand must be a quarter of the deck")
        if set(opponents) != {seat for seat in range(4) if seat != hero}:
            raise ValueError("need exactly one policy for each other seat")
        self.opponents = dict(opponents)
        self.hand_size = len(hero_hand)
        self.roots = self._root_beliefs()
        self._init_search()

    def _init_search(self) -> None:
        self.layouts = sum(len(belief) for _, belief in self.roots)
        self.total_mass = sum(sum(belief.values()) for _, belief in self.roots)  # values are divided by this
        # Remember best-response values by history, for move_values / policy_decisions / the
        # cross-table evaluation. A single value() call never revisits a history, so a caller
        # that only wants values can turn this off to save memory.
        self.remember_best = True
        self.node_budget: int | None = None  # hero decisions allowed per value() call
        self._budget_left = -1  # -1: unlimited; value() sets it from node_budget
        self._opponent_cache: dict[tuple, Distribution] = {}
        self.decision_nodes = 0
        self.opponent_queries = 0
        # Best-response move at each hero information state, keyed by public history:
        # the hero's hand is fixed, so the history determines everything the hero knows.
        self.best_moves: dict[tuple, Card] = {}
        self._best_values: dict[tuple, float] = {}  # (player, history) -> best-response value

    @classmethod
    def from_position(
        cls,
        state: GameState,
        hero: int,
        hero_hand: Iterable[Card],
        opponents: Mapping[int, OpponentPolicy],
        deck: Iterable[Card],
    ) -> "BestResponseSolver":
        """Best response from a mid-game position, given its full public history.

        `hero_hand` is the hero's current cards. The root belief is every split of
        the unseen cards that fits the hand counts, weighted by the probability that
        the opponents' policies would have played the history from it (uniform
        prior over deals). Values are win probabilities given this position.
        """
        solver = cls.__new__(cls)
        solver.hero = hero
        solver.hero_mask = card_set_mask(hero_hand)
        solver.deck = frozenset(deck)
        if set(opponents) != {seat for seat in range(4) if seat != hero}:
            raise ValueError("need exactly one policy for each other seat")
        solver.opponents = dict(opponents)
        solver.hand_size = len(solver.deck) // 4
        solver.roots = [(state, position_posterior(state, hero, solver.hero_mask, solver.opponents, solver.deck))]
        solver._init_search()
        if solver.total_mass <= 0.0:
            raise ValueError("the opponents' policies cannot have produced this history")
        return solver

    def _root_beliefs(self) -> list[tuple[GameState, dict[Layout, float]]]:
        seats = [seat for seat in range(4) if seat != self.hero]
        unseen = sorted(self.deck - frozenset(mask_to_cards(self.hero_mask)), key=card_bit)
        seven_of_hearts = card_bit(Card(Suit.HEARTS, 7))
        by_starter: dict[int, dict[Layout, float]] = defaultdict(dict)
        size = self.hand_size
        for first in combinations(unseen, size):
            rest = [card for card in unseen if card not in first]
            for second in combinations(rest, size):
                third = [card for card in rest if card not in second]
                layout = [0, 0, 0, 0]
                for seat, cards in zip(seats, (first, second, third)):
                    layout[seat] = card_set_mask(cards)
                starter = next(
                    seat for seat in range(4)
                    if (self.hero_mask if seat == self.hero else layout[seat]) & seven_of_hearts
                )
                by_starter[starter][tuple(layout)] = 1.0
        counts = (size,) * 4
        return [
            (GameState(hand_counts=counts, current_player=starter), belief)  # type: ignore[arg-type]
            for starter, belief in sorted(by_starter.items())
        ]

    def value(self, policy: HeroPolicy | None = None) -> float:
        self._budget_left = self.node_budget if self.node_budget is not None else -1
        try:
            total = sum(self._node(state, belief, policy) for state, belief in self.roots)
        finally:
            self._budget_left = -1
        return total / self.total_mass

    def best_response_policy(self) -> HeroPolicy:
        """The best response found by `value()`, as a policy that reads only the public history."""
        if not self.best_moves:
            self.value()
        moves = self.best_moves
        return lambda state, player, hand_mask, belief: ((moves[state.history], 1.0),)

    def move_values(self, state: GameState, belief: Belief) -> tuple[tuple[Card, ...], list[float]]:
        """Best-response value of each legal hero move at a hero decision, unnormalised.

        `belief` must be this solver's own belief at that history (prior times its
        opponents' likelihood), since best-response values are cached by history.
        """
        player = state.current_player
        hand_mask = self.hero_mask & ~state.played_cards_mask()
        legal = mask_to_cards(hand_mask & state.public_legal_mask())
        self.decision_nodes += 1
        mass = sum(belief.values())
        values = [
            mass if hand_mask == card_bit(card) else self._node(state.after_play(player, card), belief, None)  # type: ignore[arg-type]
            for card in legal
        ]
        return legal, values

    def _opponent_distribution(self, state: GameState, player: int, hand_mask: int) -> Distribution:
        policy = self.opponents[player]
        if getattr(policy, "cheap", False):
            return policy(state, player, hand_mask)
        key = (state.history, player, hand_mask)
        cached = self._opponent_cache.get(key)
        if cached is None:
            self.opponent_queries += 1
            cached = policy(state, player, hand_mask)
            if len(self._opponent_cache) >= OPPONENT_CACHE_LIMIT:
                self._opponent_cache.clear()
            self._opponent_cache[key] = cached
        return cached

    def _node(self, state: GameState, belief: dict[Layout, float], policy: HeroPolicy | None) -> float:
        """Unnormalised mass of the belief in which the hero goes out first."""
        if trailing_pass_count(state.history) >= 4:
            raise ValueError("deadlock: nobody can play")
        player = state.current_player
        if player == self.hero:
            hand_mask = self.hero_mask & ~state.played_cards_mask()
            legal = mask_to_cards(hand_mask & state.public_legal_mask())
            if not legal:
                return self._node(state.after_play(player, None), belief, policy)
            if self._budget_left == 0:
                raise BudgetExceeded
            self._budget_left -= 1
            if policy is None:
                key = (player, state.history)
                if not self.remember_best:
                    return max(self.move_values(state, belief)[1])
                if key not in self._best_values:
                    legal, values = self.move_values(state, belief)
                    best = max(range(len(legal)), key=values.__getitem__)
                    self.best_moves[state.history] = legal[best]
                    self._best_values[key] = values[best]
                return self._best_values[key]
            self.decision_nodes += 1
            mass = sum(belief.values())
            return sum(
                probability * (mass if hand_mask == card_bit(card) else
                               self._node(state.after_play(player, card), belief, policy))
                for card, probability in policy(state, player, hand_mask, belief)
            )

        return sum(self._node(child_state, child, policy) for child_state, child in self._opponent_children(state, belief))

    def _opponent_children(
        self, state: GameState, belief: dict[Layout, float]
    ) -> list[tuple[GameState, dict[Layout, float]]]:
        """Split the belief by the opponent's observable action; drops actions that end the game."""
        player = state.current_player
        children: dict[Card | None, dict[Layout, float]] = defaultdict(dict)
        by_hand: dict[int, list[tuple[Layout, float]]] = defaultdict(list)
        for layout, weight in belief.items():
            by_hand[layout[player]].append((layout, weight))
        for hand_mask, members in by_hand.items():
            for card, probability in self._opponent_distribution(state, player, hand_mask):
                child = children[card]
                if card is None:
                    for layout, weight in members:
                        child[layout] = weight * probability
                    continue
                remaining = hand_mask & ~card_bit(card)
                for layout, weight in members:
                    moved = list(layout)
                    moved[player] = remaining
                    child[tuple(moved)] = weight * probability  # type: ignore[index]
        return [
            (state.after_play(player, card), child)
            for card, child in children.items()
            if card is None or state.hand_counts[player] != 1  # type: ignore[index]  # else this opponent goes out first
        ]

    def policy_decisions(self, policy: HeroPolicy) -> list["DecisionRecord"]:
        """Every hero decision with 2+ legal moves reached while following `policy`.

        Each record has the probability of reaching it and the best-response value of
        every legal move. By the performance-difference identity, summing
        reach * (best value - value of the policy's move) over the records gives
        exactly value() - value(policy): it says where the policy's regret comes from.
        """
        records: list[DecisionRecord] = []

        def node(state: GameState, belief: dict[Layout, float]) -> None:
            if trailing_pass_count(state.history) >= 4:
                raise ValueError("deadlock: nobody can play")
            player = state.current_player
            if player != self.hero:
                for child_state, child in self._opponent_children(state, belief):
                    node(child_state, child)
                return
            hand_mask = self.hero_mask & ~state.played_cards_mask()
            legal = mask_to_cards(hand_mask & state.public_legal_mask())
            if not legal:
                node(state.after_play(player, None), belief)
                return
            mass = sum(belief.values())
            choices = policy(state, player, hand_mask, belief)
            if len(legal) > 1:
                legal, values = self.move_values(state, belief)
                records.append(DecisionRecord(
                    state=state,
                    hand_mask=hand_mask,
                    legal=legal,
                    reach=mass / self.total_mass,
                    values=tuple(value / mass for value in values),
                    choices=tuple(choices),
                ))
            for card, probability in choices:
                if hand_mask != card_bit(card) and probability > 0.0:
                    node(state.after_play(player, card), belief)

        for state, belief in self.roots:
            node(state, belief)
        return records


class DecisionRecord(NamedTuple):
    state: GameState
    hand_mask: int  # hero's cards
    legal: tuple[Card, ...]
    reach: float  # probability the hero faces this decision
    values: tuple[float, ...]  # best-response win probability after each legal move, given reaching here
    choices: tuple[tuple[Card | None, float], ...]  # the policy's move distribution

    def loss(self) -> float:
        """Win probability given up here by the policy's move, if play is optimal afterwards."""
        best = max(self.values)
        index = {card: i for i, card in enumerate(self.legal)}
        return sum(probability * (best - self.values[index[card]]) for card, probability in self.choices)


class CrossTableResult(NamedTuple):
    value: float  # first-out win vs the actual table, playing the move ideal against the assumed one
    decision_mass: float  # probability-weighted hero decisions faced
    agree_mass: float  # ... where the assumed-table move is also ideal against the actual table
    contested_mass: float  # ... where the actual table's move values differ at all
    contested_agree_mass: float


def evaluate_assumed_table(assumed: BestResponseSolver, actual: BestResponseSolver) -> CrossTableResult:
    """Play the best response to `assumed`'s opponents against `actual`'s opponents, exactly.

    The hero believes the assumed table: at each decision their belief is the prior
    times the assumed policies' likelihood of the public history, and they play the
    move that is best if the assumed table is right. If the actual table does
    something the assumed one never would, that belief resets to uniform over the
    layouts consistent with the cards seen so far. So the hero's move depends on the
    public history alone.

    Masses are summed over hero decisions weighted by their probability, so
    agree_mass / decision_mass is the chance that a decision the hero faces has the
    assumed-table move among the actual table's best moves.
    """
    if assumed.hero != actual.hero or assumed.hero_mask != actual.hero_mask or assumed.deck != actual.deck:
        raise ValueError("both solvers need the same hero, hand and deck")
    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    hero = actual.hero

    def node(state: GameState, layouts: dict[Layout, tuple[float, float]]) -> float:
        if trailing_pass_count(state.history) >= 4:
            raise ValueError("deadlock: nobody can play")
        player = state.current_player
        if player == hero:
            hand_mask = actual.hero_mask & ~state.played_cards_mask()
            if not hand_mask & state.public_legal_mask():
                return node(state.after_play(player, None), layouts)
            assumed_belief = {layout: wa for layout, (wa, _) in layouts.items() if wa > 0.0}
            actual_belief = {layout: wb for layout, (_, wb) in layouts.items() if wb > 0.0}
            legal, assumed_values = assumed.move_values(state, assumed_belief)
            choice = max(range(len(legal)), key=assumed_values.__getitem__)
            _, actual_values = actual.move_values(state, actual_belief)
            mass = sum(actual_belief.values())
            best = max(actual_values)
            tolerance = 1e-12 * max(mass, 1e-300)
            agrees = actual_values[choice] >= best - tolerance
            contested = best - min(actual_values) > tolerance
            totals[1] += mass
            totals[2] += mass * agrees
            totals[3] += mass * contested
            totals[4] += mass * (agrees and contested)
            if hand_mask == card_bit(legal[choice]):
                return mass
            return node(state.after_play(player, legal[choice]), layouts)

        by_hand: dict[int, list[tuple[Layout, tuple[float, float]]]] = defaultdict(list)
        for layout, weights in layouts.items():
            by_hand[layout[player]].append((layout, weights))
        dists = {}
        actions: set[Card | None] = set()
        for hand_mask, members in by_hand.items():
            # Layouts with zero weight stay at zero whatever a policy does, so only
            # ask a policy about hands that still carry weight under it.
            dist_a = dist_b = {}
            if any(wa > 0.0 for _, (wa, _) in members):
                dist_a = dict(assumed._opponent_distribution(state, player, hand_mask))
            if any(wb > 0.0 for _, (_, wb) in members):
                dist_b = dict(actual._opponent_distribution(state, player, hand_mask))
            dists[hand_mask] = (dist_a, dist_b)
            actions.update(card for card, probability in dist_b.items() if probability > 0.0)
        total = 0.0
        public_legal = state.public_legal_mask()
        for card in actions:
            if card is not None and state.hand_counts[player] == 1:  # type: ignore[index]
                continue  # this opponent goes out first
            bit = 0 if card is None else card_bit(card)
            child: dict[Layout, tuple[float, float]] = {}
            for hand_mask, members in by_hand.items():
                legal = hand_mask & public_legal
                if (card is None and legal) or (card is not None and not legal & bit):
                    continue  # the cards rule this layout out
                dist_a, dist_b = dists[hand_mask]
                pa, pb = dist_a.get(card, 0.0), dist_b.get(card, 0.0)
                for layout, (wa, wb) in members:
                    if card is not None:
                        moved = list(layout)
                        moved[player] = hand_mask & ~bit
                        layout = tuple(moved)  # type: ignore[assignment]
                    child[layout] = (wa * pa, wb * pb)
            if not any(wa > 0.0 for wa, _ in child.values()):
                child = {layout: (1.0, wb) for layout, (_, wb) in child.items()}
            total += node(state.after_play(player, card), child)
        return total

    for (state, belief_b), (_, belief_a) in zip(actual.roots, assumed.roots):
        start = {layout: (belief_a.get(layout, 0.0), weight) for layout, weight in belief_b.items()}
        if not any(wa > 0.0 for wa, _ in start.values()):
            start = {layout: (1.0, wb) for layout, (_, wb) in start.items()}
        totals[0] += node(state, start)
    totals[0] /= actual.total_mass
    return CrossTableResult(*totals)


def replay_states(state: GameState, deck_size: int) -> list[GameState]:
    """The state before each event of `state.history`, rebuilt from the deal."""
    history = state.history
    if not history:
        return []
    counts = [count for count in state.hand_counts]  # type: ignore[union-attr]
    for event in history:
        counts[event.player] += event.card is not None
    if sum(counts) != deck_size:
        raise ValueError("hand counts and history do not add up to the deck")
    current = GameState(hand_counts=tuple(counts), current_player=history[0].player)  # type: ignore[arg-type]
    states = []
    for event in history:
        states.append(current)
        current = current.after_play(event.player, event.card)
    if current.table != state.table or current.hand_counts != state.hand_counts:
        raise ValueError("history does not reproduce the position")
    return states


def history_likelihood(
    policy: OpponentPolicy, states: list[GameState], history: tuple[MoveEvent, ...], seat: int, hand_mask: int
) -> float:
    """Probability that `policy` at `seat`, holding `hand_mask` now, played its part of the history."""
    probability = 1.0
    hand = hand_mask
    for index in range(len(history) - 1, -1, -1):  # backwards, so the hand at each event is known
        event = history[index]
        if event.player != seat:
            continue
        if event.card is not None:
            hand |= card_bit(event.card)
        probability *= dict(policy(states[index], seat, hand)).get(event.card, 0.0)
        if probability == 0.0:
            return 0.0
    return probability


def position_posterior(
    state: GameState, hero: int, hero_mask: int, opponents: Mapping[int, OpponentPolicy], deck: frozenset[Card]
) -> dict[Layout, float]:
    """Every opponent layout that fits the position, weighted by the history's likelihood.

    The likelihood factorises over opponents, since each one's actions depend only
    on the public state and their own hand, so each seat's candidate hands are
    scored once.
    """
    states = replay_states(state, len(deck))
    seats = [seat for seat in range(4) if seat != hero]
    unseen = [card for card in mask_to_cards(card_set_mask(deck) & ~hero_mask & ~state.played_cards_mask())]
    counts = [state.hand_counts[seat] for seat in seats]  # type: ignore[index]
    if sum(counts) != len(unseen):  # type: ignore[arg-type]
        raise ValueError("opponent hand counts do not match the unseen cards")
    unseen_mask = card_set_mask(unseen)
    candidates: dict[int, dict[int, float]] = {}
    for seat, count in zip(seats, counts):
        scored = {}
        for hand in combinations(unseen, count):  # type: ignore[arg-type]
            mask = card_set_mask(hand)
            weight = history_likelihood(opponents[seat], states, state.history, seat, mask)
            if weight > 0.0:
                scored[mask] = weight
        candidates[seat] = scored
    # Join the two most constrained seats; the third seat's hand is what is left.
    first, second, third = sorted(seats, key=lambda seat: len(candidates[seat]))
    belief: dict[Layout, float] = {}
    for first_mask, first_weight in candidates[first].items():
        for second_mask, second_weight in candidates[second].items():
            if first_mask & second_mask:
                continue
            third_weight = candidates[third].get(unseen_mask & ~first_mask & ~second_mask, 0.0)
            if third_weight > 0.0:
                layout = [0, 0, 0, 0]
                layout[first], layout[second] = first_mask, second_mask
                layout[third] = unseen_mask & ~first_mask & ~second_mask
                belief[tuple(layout)] = first_weight * second_weight * third_weight  # type: ignore[index]
    return belief
