# Solver Proof Plan

This document describes how the project can establish, test, and explain that a
solver for 7 of Hearts is exact or optimal under clearly stated assumptions.

The central idea is to separate three claims:

1. The rules engine represents the real game correctly.
2. A full-information solver exactly solves the finite game tree.
3. An imperfect-information solver chooses the best expected move over all
   hidden deals consistent with public evidence.

The current project already has useful heuristic, inference, sampling, and
rollout layers. Those are evidence-producing systems. A proof-grade solver needs
an explicit recurrence, a formal objective, and tests/certificates showing that
every legal continuation is evaluated without approximation.

## Implementation Status

The first proof-layer implementation now exists in `seven_hearts.py`.

Implemented:

- `FullInformationState`, an immutable, hashable full-information state for
  exact search
- full-information legal move generation from complete hands and public table
- validated full-information play/pass transitions
- explicit card-conservation checks for declared decks
- `solve_full_information(...)`, a memoized exhaustive rational-play solver
- `ExactSolverResult`, including value vector, move values, best moves, chosen
  move, cache statistics, terminal counts, and deadlock counts
- deterministic rational tie-breaking
- fixed-policy exact solving through `solve_full_information_against_policy(...)`
- built-in fixed-policy helpers for lowest-card and highest-card play
- human-readable certificate formatting for exact full-information results
- `enumerate_hidden_deals(...)`, an exhaustive hidden-deal enumerator with an
  optional `max_deals` guard
- `evaluate_move_exact_imperfect_information(...)`, exact expected value for a
  candidate move when hidden deals are exhaustively enumerated
- `recommend_move_exact_imperfect_information(...)`, exact or explicitly
  non-exhaustive hidden-information recommendation reporting
- shared hidden-deal enumeration across all candidate moves in exact
  imperfect-information recommendation
- shared exact continuation cache across candidate moves and hidden deals in an
  exact imperfect-information recommendation
- human-readable certificate formatting for exact imperfect-information results
- `proof_demo.py`, a command-line proof harness that prints exact certificates
- proof-layer tests for immediate wins, forced passes, rational avoidance of
  opening a next-player win, deterministic ties, brute-force cross-checking,
  fixed-policy solving, certificate output, hidden-deal enumeration, exact EV,
  and non-exhaustive limits

Current verification:

```text
py run_tests.py
56 tests passed

py proof_demo.py

py -m py_compile seven_hearts.py test_seven_hearts.py run_tests.py proof_demo.py
```

Important current limitations:

- exact full-information search is suitable for small or late-game states first;
  full 52-card initial deals may be too large without further optimization
- deadlocks are counted and given neutral value; valid normal games should not
  hit them before first-out
- exact imperfect-information evaluation is proof-grade only when
  `exhaustive=True`
- when `max_deals` truncates hidden-deal enumeration, the result is diagnostic
  rather than proof-grade; because truncation is deterministic, no statistical
  standard error is reported for truncated exact-enumeration results
- hidden-deal enumeration currently uses the existing public constraint model
  and a uniform prior over enumerated deals

Highest-priority proof work after the initial implementation:

1. Add randomized tiny-state brute-force cross-checking across many generated
   reachable reduced games.
2. Add conservation-walk fuzz tests across random simulated games.
3. Add regression certificate snapshots for several hand-authored proof
   positions.
4. Add pass-chain canonicalization as a proof-preserving optimization.
5. Consider bitmask state representation for deeper exact search.

## Formal Model

This section gives the mathematical structure that the implementation should
mirror.

### Basic Sets

Let:

```text
N = {0, 1, 2, 3}
```

be the set of players.

Let:

```text
Suits = {H, S, D, C}
Ranks = {1, 2, ..., 13}
```

where rank `1` is Ace, rank `7` is Seven, and rank `13` is King.

The deck is:

```text
D = Suits x Ranks
```

with `|D| = 52`.

A card is written:

```text
c = (suit, rank) in D
```

### Table Representation

For each suit `q`, the table state is either unopened or an interval containing
rank 7:

```text
table(q) = unopened
```

or:

```text
table(q) = [l_q, h_q]
where 1 <= l_q <= 7 <= h_q <= 13
```

The cards on the table for an opened suit are:

```text
{(q, r) : l_q <= r <= h_q}
```

For an unopened suit, no cards of that suit are on the table.

### Full-Information State Space

A full-information state is:

```text
s = (H_0, H_1, H_2, H_3, table, p, w, k)
```

where:

- `H_i subset D` is player `i`'s hand
- `table` is the table representation above
- `p in N` is the current player
- `w in N union {None}` is the first-out winner, if known
- `k` is the number of consecutive passes since the last play

The state must satisfy card conservation:

```text
H_0, H_1, H_2, H_3, and table_cards(table) are pairwise disjoint
H_0 union H_1 union H_2 union H_3 union table_cards(table) = D
```

for complete full-deal states. For reduced test games, replace `D` with the
declared reduced deck.

### Public Legal Cards

Define `PublicLegal(table)` as the set of cards that may be played to the table
regardless of who holds them.

If the table is empty:

```text
PublicLegal(table) = {(H, 7)}
```

Otherwise, for each suit `q`:

If `q` is unopened:

```text
(q, 7) in PublicLegal(table)
```

If `q` is opened as `[l_q, h_q]`, then:

```text
(q, l_q - 1) in PublicLegal(table) if l_q > 1
(q, h_q + 1) in PublicLegal(table) if h_q < 13
```

No other cards are public-legal.

### Action Function

For a non-terminal state `s`, the current player's legal card actions are:

```text
A_card(s) = H_p intersect PublicLegal(table)
```

where `p` is the current player.

The full legal action set is:

```text
A(s) = A_card(s)                  if A_card(s) is non-empty
A(s) = {pass}                     otherwise
```

For terminal states:

```text
A(s) = empty
```

### Transition Function

The transition function:

```text
T: S x A -> S
```

maps a state and legal action to the unique successor state.

If action `a` is a card `(q, r)`:

```text
H'_p = H_p \ {a}
H'_j = H_j for j != p
table' = expand(table, a)
p' = (p + 1) mod 4
k' = 0
```

If `H'_p` is empty:

```text
w' = p
```

otherwise:

```text
w' = None
```

If action `a` is `pass`:

```text
H'_j = H_j for all j
table' = table
p' = (p + 1) mod 4
w' = w
k' = k + 1
```

`T(s, a)` is undefined for illegal actions.

### Terminal Utility

A terminal state has `w != None`.

The first-out utility vector is:

```text
U(s) = e_w
```

where `e_w` is the length-4 vector with:

```text
e_w[w] = 1
e_w[j] = 0 for j != w
```

Example:

```text
if player 2 wins, U(s) = (0, 0, 1, 0)
```

Optional secondary utilities, such as finish margin or remaining-card count, may
be added only if they are documented as part of the objective or tie-break
system.

### Full-Information Value Function

The exact full-information value function is:

```text
V: S -> R^4
```

For terminal states:

```text
V(s) = U(s)
```

For non-terminal forced-pass states:

```text
V(s) = V(T(s, pass))
```

For non-terminal states with legal card actions, let `p` be the current player.
The acting player chooses the action that maximizes their own component of the
successor value:

```text
Best(s) = argmax_{a in A_card(s)} V_p(T(s, a))
```

If `Best(s)` contains one action:

```text
V(s) = V(T(s, a*))
```

where `a*` is that action.

If `Best(s)` contains multiple actions, apply the documented deterministic
tie-break function:

```text
tau(s, Best(s)) -> a*
```

and return:

```text
V(s) = V(T(s, a*))
```

The resulting policy is:

```text
pi*(s) = a*
```

for non-terminal states.

### Tie-Break Function

The tie-break function:

```text
tau: S x 2^A -> A
```

must be deterministic.

The implemented rational tie-breaker is:

1. maximize the acting player's first-out value
2. minimize the next player's first-out value
3. minimize the strongest opponent's first-out value
4. use deterministic card order

Optional secondary utilities, such as finish-margin proxies, may be added later
only if they are formalized as part of the objective and implemented in both the
solver and its certificates.

Without `tau`, rational play may be underdetermined in multiplayer states where
several actions are equally good for the acting player.

### Fixed-Policy Variant

For testing or opponent-model analysis, define a fixed policy:

```text
pi: S -> A
```

The value under that policy is:

```text
V^pi(s) = U(s)                         if terminal
V^pi(s) = V^pi(T(s, pass))             if forced pass
V^pi(s) = V^pi(T(s, pi(s)))            otherwise
```

For a solver choosing only the current move against a fixed future policy, the
best action for player `i` is:

```text
argmax_{a in A_card(s)} V^pi_i(T(s, a))
```

This is often easier to test than full rational play.

### Public State and Hidden Deals

Let `P` be a public state from the solver's perspective. It contains public
evidence plus the solver's private hand.

Define:

```text
B(P) = {h : h is a complete hidden deal consistent with P}
```

Each `h in B(P)` assigns every unseen card to exactly one opponent and satisfies
all public constraints:

- known hand counts
- pass information
- played-card ownership
- exclusion of the solver's hand
- exclusion of table cards

Let:

```text
Pr(h | P)
```

be the hidden-deal prior. The first exact version should use the uniform prior:

```text
Pr(h | P) = 1 / |B(P)|
```

for all `h in B(P)`.

### Materialization

Materialization converts a public state and hidden deal into a full-information
state:

```text
M(P, h) = s
```

The materialized state must have:

- the solver's known hand
- opponent hands from `h`
- the public table
- the public current player
- no extra hidden information beyond `h`

### Imperfect-Information Expected Value

For solver player `i` and candidate legal action `a`, define:

```text
EV_i(P, a) =
  sum_{h in B(P)} Pr(h | P) * V_i(T(M(P, h), a))
```

where `V` is the full-information value function under the declared continuation
model.

The exact imperfect-information optimal policy is:

```text
Pi*(P) = argmax_{a in A_card(M(P, h), i)} EV_i(P, a)
```

The legal action set must be the same for all `h in B(P)` because the solver's
own hand and the public table are known. If it is not the same, materialization
or public-state construction is inconsistent.

If multiple actions tie, apply a documented deterministic public-state
tie-breaker:

```text
Tau(P, tied_actions) -> a*
```

### Sampling Approximation

Monte Carlo replaces the exact expectation with:

```text
EV_hat_i(P, a) =
  (1 / n) * sum_{m=1}^{n} V_i(T(M(P, h_m), a))
```

where each `h_m` is sampled from `Pr(h | P)`.

When hidden deals are sampled uniformly at random, this estimator can support
confidence intervals and standard error:

```text
SE(EV_hat) = sample_standard_deviation / sqrt(n)
```

but it is not an exact proof unless the sampled set exhausts `B(P)`.
Deterministic prefix truncation of hidden-deal enumeration is not random
sampling and must not report a statistical standard error.

## Definitions

### Game

7 of Hearts is a finite four-player game using a standard 52-card deck.

Each player starts with 13 cards. The player holding `7H` starts and must play
`7H` as the first move. Cards are then played to suit runs. A suit is unopened
until its 7 has been played. Once opened, the only legal cards in that suit are
the rank directly below the current low end or directly above the current high
end.

On each turn:

- if the current player has at least one legal card, they must play exactly one
  legal card
- if the current player has no legal card, they pass
- turn then advances clockwise

The strategic objective for this project is first-out: a player wins when they
are the first player to empty their hand.

### Full-Information State

A full-information state contains all information needed to continue the game
deterministically:

```text
hands: tuple[frozenset[Card], frozenset[Card], frozenset[Card], frozenset[Card]]
table: suit intervals for H, S, D, C
current_player: int in {0, 1, 2, 3}
winner: optional int
consecutive_passes: int
```

The exact representation can differ in code, but it must be canonical and
hashable for memoization. Two states that are strategically identical must have
the same cache key.

The state must not include heuristic scores, explanations, sampled deals, or any
other non-rule metadata.

### Public / Imperfect-Information State

A public state contains only information available to the solver:

```text
own_hand
cards_on_table
current_player
hand_counts, if known
pass history
public move history
solver_player
```

The public state induces a belief set: every complete hidden deal that is
consistent with public evidence.

### Legal Move

A legal move is a card that:

- is in the current player's hand
- is legal according to the public table
- respects the opening rule

Passing is legal only when the current player has no legal card.

### Terminal State

For the project objective, a terminal state occurs as soon as any player has
zero cards. That player is the first-out winner.

The solver does not need to simulate the rest of the game after the first player
empties their hand unless a later objective requires finishing order.

## Solution Concepts

The word "optimal" is only meaningful after choosing a solution concept.

### Fixed-Policy Optimality

The simplest proof target:

```text
The solver chooses the move with the highest value assuming every future
opponent move is generated by a fixed, explicitly defined policy.
```

This is useful for comparing against known bots, heuristic opponents, or a
specific rollout policy.

### Full-Information Rational Optimality

The stronger full-information target:

```text
Each player chooses the legal move that maximizes their own chance or utility
of being first out, assuming all future players do the same.
```

Because this is a multiplayer general-sum game, ties and non-winning preferences
must be defined explicitly. Without tie-breaking, multiple rational continuations
may exist and the solver's recommendation may be underdetermined.

Recommended value model:

```text
value(state) = tuple[float, float, float, float]
```

Each component is that player's value from the state. For deterministic
full-information first-out play, terminal values are:

```text
winner gets 1.0
all other players get 0.0
```

On a player's turn, the acting player chooses the successor whose value vector
has the highest component for that acting player.

Recommended tie-breakers, in order:

1. maximize the acting player's first-out value
2. minimize the next player's first-out value
3. minimize the strongest opponent's first-out value
4. use deterministic card order for reproducibility

Tie-breakers must be part of the formal solver definition. They are not cosmetic:
they determine the chosen line when several moves are equally rational for the
acting player.

### Imperfect-Information Expected Optimality

The hidden-information target:

```text
The solver chooses the legal move with maximal expected value over all hidden
deals consistent with public evidence.
```

For a public state `P` and candidate move `m`:

```text
EV(P, m) = sum over hidden deals h:
             Pr(h | P) * FullInformationValue(apply(P, h, m))
```

The chosen move is:

```text
argmax_m EV(P, m)
```

This is exact only if the solver enumerates the complete hidden-deal set or
otherwise computes the same expectation without sampling error.

Monte Carlo evaluation is an approximation of this target. It should report
standard error and confidence intervals, but it should not be described as proof
of optimality unless the sample space is exhausted.

## Exact Full-Information Recurrence

The exact solver should mirror the recurrence directly.

```text
solve(state):
  if state is terminal:
    return terminal_value(state)

  legal_moves = legal cards for state.current_player

  if legal_moves is empty:
    return solve(state.after_pass())

  child_results = []
  for move in legal_moves:
    child_state = state.after_play(move)
    child_results.append((move, solve(child_state)))

  return choose_best_result_for_current_player(child_results)
```

This recurrence is exact if:

- `legal_moves` includes every legal card
- `legal_moves` excludes every illegal card
- `after_play` applies exactly the game rules
- `after_pass` applies exactly the game rules
- `choose_best_result_for_current_player` implements the declared objective
- memoization is keyed only by canonical game state
- no heuristic pruning changes the result

Memoization is allowed because it does not approximate. It only reuses the exact
answer for an identical state.

## Termination Argument

Every play removes one card from one hand. There are at most 52 plays before all
cards are gone, and the game terminates earlier when the first player empties
their hand.

Passes do not reduce card count, so the solver must handle them explicitly.

Recommended approach:

- include `consecutive_passes` in the state or transition context
- reset it to 0 after a play
- increment it after a pass
- if all non-terminal players pass consecutively without any play, treat the
  state as invalid or deadlocked and surface a consistency error

In valid games generated from a full deal and the correct `7H` opening rule, a
deadlock should not occur before first-out. If it does, the solver should expose
the state for debugging rather than silently assigning a heuristic value.

## Induction Proof Sketch

The exact full-information solver can be justified by induction over game
progress.

Base case:

If a state is terminal, the winner is already known. Returning the terminal
value vector is correct.

Inductive step:

Assume the solver returns correct values for all successor states after one
legal play. For a non-terminal state where the current player has legal moves,
the rules require the current player to choose exactly one legal card. The solver
evaluates every legal successor and selects the successor that is best according
to the declared rational objective. Therefore the returned value for the current
state is correct.

Pass case:

If the current player has no legal move, the rules force a pass. There is only
one successor state. The value of the current state is therefore exactly the
value of that forced successor.

The pass case requires the termination guard described above.

## Required Engine Invariants

The proof depends on the rules engine being correct. The following invariants
should be tested and, where practical, asserted.

### Card Conservation

At every full-information state:

```text
cards in all hands
+ cards on table
= full 52-card deck
```

No card may appear in more than one place.

### Table Validity

For every opened suit:

- the interval contains rank 7
- the low bound is between Ace and 7
- the high bound is between 7 and King
- every card between low and high is considered on the table

For every unopened suit:

- no card in that suit is on the table
- the only possible opening card is that suit's 7, except before `7H`

### Opening Rule

If the table is empty:

```text
public legal cards = {7H}
```

After `7H` is on the table:

```text
unopened suits may be opened by their 7s
opened suits may grow by adjacent cards
```

### Legal Move Completeness

For each current player:

```text
full_information_legal_moves(state)
= player_hand intersect public_legal_cards(state.table)
```

### Pass Legality

A pass is legal if and only if:

```text
full_information_legal_moves(state) is empty
```

### Transition Correctness

After a play:

- the played card is removed from exactly one player's hand
- the table interval for exactly one suit changes
- no other hand changes
- current player advances by one seat
- consecutive pass count resets
- terminal winner is set if the acting player's hand is now empty

After a pass:

- no hand changes
- no table interval changes
- current player advances by one seat
- consecutive pass count increments
- no winner is created by the pass

## Exact Solver Deliverables

### `FullInformationGameState`

Create or adapt a canonical full-information state object.

Requirements:

- immutable or treated immutably
- hashable
- contains complete hands
- contains table state
- contains current player
- contains terminal winner
- contains pass-cycle tracking
- exposes legal move generation
- exposes validated transitions

### `ExactSolverResult`

The exact solver should return a structured result.

Suggested fields:

```text
value: tuple[float, float, float, float]
best_moves: tuple[Card, ...]
chosen_move: Card | None
states_evaluated: int
cache_hits: int
terminal_states: int
deadlock_states: int
policy_name: str
tie_break_description: str
```

`best_moves` should include all moves tied under the primary objective.
`chosen_move` should apply the deterministic tie-breaker.

### Solver Modes

Start with two full-information modes:

1. `exact_fixed_policy`
2. `exact_rational`

`exact_fixed_policy` is useful for testing and comparison.

`exact_rational` is the proof target for perfect rational play.

### Certificate Output

For any analyzed state, the solver should be able to produce a certificate:

```text
current player
legal moves
value vector for each legal move
best move or tied best moves
tie-breaker used
number of states evaluated
cache size
terminal states reached
deadlock states reached
```

This certificate is the user-facing proof artifact. It shows that the
recommendation came from exhaustive evaluation rather than heuristic preference.

## Required Tests

### Rules Tests

Add focused tests for:

- only `7H` is legal on an empty table
- after `7H`, unopened suits can be opened by their 7s
- opened suits grow only by adjacent ranks
- player cannot play a card they do not hold
- player cannot play a non-adjacent card
- player cannot pass while holding a legal card
- player must pass when holding no legal cards

### Transition Tests

Add focused tests for:

- playing a card removes it from hand
- playing a card expands exactly the correct suit interval
- playing a non-7 in an unopened suit is rejected
- pass does not change hands or table
- turn order advances correctly after play and pass
- winner is set immediately when a hand reaches zero

### Conservation Tests

Add tests that walk random valid games and assert:

- all 52 cards remain accounted for
- no duplicate card exists across hands/table
- table intervals match played cards
- hand counts decrease only on plays

### Exact Search Tests

Add small deterministic positions:

- one-card current hand where the only legal card wins immediately
- current player has one winning move and one non-winning move
- forced pass chain reaches the only player with a legal move
- playing a tempting unlock gives next player an immediate win, so exact solver
  avoids it if another move has better value
- two moves are equal under primary utility and deterministic tie-breaker chooses
  the documented move

### Brute-Force Cross-Check Tests

For tiny reduced-deck variants or artificially small states:

- implement a deliberately simple non-memoized brute-force solver
- compare its result with the memoized exact solver
- run across many generated tiny states

This is one of the strongest implementation checks because it verifies that the
cache and canonicalization are not changing answers.

### Regression Tests Against Known Certificates

Store a few exact-solver certificate snapshots for hand-authored positions.

Each snapshot should include:

- current player
- legal moves
- child value vectors
- chosen move

These protect the proof layer from accidental behavioral drift.

## Imperfect-Information Proof Plan

After the full-information exact solver exists, use it as the value oracle for
hidden-deal analysis.

### Belief Set Construction

Given public state and solver knowledge, enumerate all hidden deals satisfying:

- every unseen card belongs to exactly one opponent
- known hand counts are satisfied exactly
- pass constraints are respected
- public played-card ownership is respected
- own hand and table cards are excluded

This should reuse the existing inference constraints where possible.

### Exact Expected Value

For each legal solver move:

```text
total = 0
for each hidden_deal in belief_set:
  probability = belief_probability(hidden_deal)
  full_state = materialize_full_state(public_state, own_hand, hidden_deal)
  child_state = full_state.after_play(solver_player, move)
  total += probability * exact_full_information_value(child_state)[solver_player]
```

The best move is the one with highest total.

### Hidden-Deal Probability

The first exact version should use a clear prior:

```text
all hidden deals consistent with public evidence are equally likely
```

Later versions can support non-uniform behavioral priors, but those must be
documented because they change the meaning of "optimal."

### Enumeration Limits

Full hidden-deal enumeration can be enormous. The solver should expose:

- hidden deal count
- whether enumeration was exhaustive
- whether it fell back to sampling
- exact value or sampled estimate
- standard error for sampled estimates

Only exhaustive enumeration should be labeled proof-grade.

## Handling Irrational Opponents

The exact rational solver should not claim that irrational opponents always
lose. That is too strong, especially in a multiplayer hidden-information game.

The defensible claim is:

```text
Against rational opponents, the solver's move is optimal under the declared
utility and tie-break rules.

If an opponent deviates, the resulting state is simply a new state. The solver
then re-solves from that state. Under the solver's value model, a deviating
opponent has chosen a move that is no better for their own evaluated position
than the rational move, except in tie cases or when hidden-information luck
changes what can be inferred.
```

This captures the practical idea: opponent mistakes do not invalidate the
solver. They create new states, and the solver remains disciplined.

## Performance Strategy

Exact search can be expensive. Optimization is allowed only if it preserves the
exact result.

Allowed exact optimizations:

- memoization by canonical state
- deterministic move ordering
- transposition tables
- alpha-beta-like pruning only if mathematically valid for the chosen objective
- caching legal public moves by table interval
- canonical card bitmasks for compact hands/table keys

Not proof-grade unless separately justified:

- heuristic pruning
- rollout cutoff values
- Monte Carlo estimates
- approximate opponent policies
- learned evaluation functions

The first implementation should prefer clarity over speed. Once the recurrence
is trusted, optimize representation.

## Suggested Implementation Sequence

1. Document the exact full-information objective and tie-breakers in code.
2. Add canonical full-information state support if the current `GameState` is
   not sufficient.
3. Add full-information legal move and transition tests.
4. Implement the simplest memoized exact solver.
5. Add certificate output.
6. Add tiny-state brute-force cross-check tests.
7. Run exact solver on hand-authored positions.
8. Compare current heuristic recommendations against exact full-information
   results on complete known deals.
9. Add exact hidden-deal enumeration for small public states.
10. Compute exact imperfect-information expected values for small states.
11. Keep Monte Carlo as the scalable approximation path for large states.

## Acceptance Criteria

The project can claim "full-information exact solver" when:

- the recurrence is implemented directly
- all legal moves are exhaustively evaluated
- no heuristic cutoffs affect the result
- memoization keys are canonical and tested
- terminal and pass cases are explicit
- rule/transition/conservation tests pass
- memoized results match brute-force results on tiny states
- certificate output shows child values for each legal move

The project can claim "optimal against rational full-information opponents"
when:

- the utility model is documented
- tie-breakers are documented
- each acting player chooses according to their own value component
- exact solver tests pass
- certificates show the selected move is best under that model

The project can claim "exact expected-optimal under imperfect information" for a
specific position when:

- every hidden deal consistent with public evidence is enumerated
- the hidden-deal prior is documented
- each hidden deal is evaluated using the full-information exact solver
- each candidate move's expected value is computed exactly
- the final certificate reports the hidden deal count and expected value of each
  legal move

The project can claim "Monte Carlo evidence" when:

- hidden deals are sampled from the documented belief distribution
- sample count is reported
- standard error is reported
- rollout policy is documented
- the result is not described as an exact proof

## Long-Term Goal

The long-term proof architecture should look like this:

```text
Rules engine
  -> exact full-information solver
    -> exact hidden-deal expectation for small belief sets
      -> Monte Carlo approximation for large belief sets
        -> heuristic policy tuned and validated against exact/sampled evidence
```

The heuristic solver remains valuable because exact play may be too expensive
for normal interactive use. The proof layer gives it a source of truth to learn
from, regress against, and explain where approximation begins.
