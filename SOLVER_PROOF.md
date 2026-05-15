# Solver Proof Plan

This document describes how the project can build, test, and explain a real-game
7 of Hearts solver: a solver that recommends a move from one player's imperfect-
information view of the table.

Primary product target:

```text
Given the solver's private hand, the public table, turn state, hand counts, and
public play/pass history, choose the move with the highest first-out expected
value over the public belief state, while simulated future players act only from
their own private hand and public evidence.
```

The central architecture is:

```text
rules engine
  -> public-history belief tracking
    -> constrained hidden-deal sampling/enumeration
      -> information-limited rollout/search policies
        -> practical imperfect-information move recommendation
```

The exact full-information solver is a validation oracle, not the final product.
It gives proof-sized answer keys, rule/regression tests, and performance
measurements for reduced or late-game states. Exact hidden-deal EV with full-
information continuations is also an oracle: useful for judging engines on small
belief sets, but not the real-game solution because it lets players see hidden
cards in each materialized deal.

The current project already has useful heuristic, inference, sampling, rollout,
and exact-oracle layers. The proof plan must keep their claims separate:

1. The rules engine represents the real game correctly.
2. The belief model conditions hidden deals on public evidence, especially
   played cards and forced passes.
3. The practical solver maximizes sampled belief-state EV under documented
   information-limited continuation policies.
4. Exact full-information and exact hidden-deal oracle modes provide validation
   certificates on tractable reduced cases.

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
  candidate move when hidden deals are exhaustively enumerated and continuations
  are evaluated with full-information rational play
- `recommend_move_exact_imperfect_information(...)`, exact or explicitly
  non-exhaustive hidden-deal recommendation reporting under the full-
  information-continuation oracle model
- shared hidden-deal enumeration across all candidate moves in exact
  imperfect-information recommendation
- shared exact continuation cache across candidate moves and hidden deals in an
  exact imperfect-information recommendation
- human-readable certificate formatting for exact hidden-deal EV results,
  including the continuation-model caveat, expected value vectors, outcome
  counts, and shared search statistics
- `proof_demo.py`, a command-line proof harness that prints exact certificates
- `proof_benchmark.py`, a command-line benchmark harness for exact full-
  information search counters and throughput
- `proof_eval.py`, a command-line visual engine-quality report comparing
  heuristic and Monte Carlo choices against the exact hidden-deal full-
  information-continuation EV oracle
- pass-chain canonicalization for forced-pass runs in exact full-information
  solving
- declared reduced decks for proof-sized `PlayerKnowledge` hidden-deal tests
- public trailing-pass materialization in `FullInformationState.from_hands(...)`
- defensive legal-move invariance checks for materialized hidden deals
- proof-layer tests for immediate wins, forced passes, rational avoidance of
  opening a next-player win, deterministic ties, brute-force cross-checking,
  fixed-policy solving, certificate output, hidden-deal enumeration, exact EV,
  and non-exhaustive limits
- randomized tiny-state brute-force cross-checking across generated reachable
  reduced games
- conservation-walk fuzz tests across random full-deck simulated games
- exact-solver certificate snapshot regression tests for hand-authored proof
  positions
- exact imperfect-information certificate snapshot regression for a reduced
  belief position with two candidate moves and different expected values
- `proof_demo.py` includes the reduced-belief exact EV certificate so the
  visible proof harness shows a multi-move hidden-information choice
- exact imperfect-information certificate snapshot regression for a multi-suit
  reduced belief position with four candidate moves and 90 hidden deals
- `proof_demo.py` includes the multi-suit exact EV certificate
- full-game-to-completion simulation that keeps playing after first-out until
  only one player remains with cards, reporting final ranks and cards-left
  outcomes
- duplicate-deal, cyclic seat-rotated full-game evaluation with Random,
  GreedyFurthestFromSeven, Heuristic, and belief-state sampled EV agents
- paired card advantage summaries and CSV reports through `full_game_eval.py`
- `full_game_eval.py` progress reporting through `--progress-every`, plus
  `--fast` mode that uses heuristic `Monte Carlo` for quick harness checks without
  Monte Carlo-in-the-loop cost
- deterministic per-game seeds and `--workers` multiprocessing for independent
  duplicate-deal rotation games, with results sorted back into deal/rotation
  order for stable serial/parallel reports
- each `full_game_eval.py` run writes a unique timestamp/PID report directory
  containing `games.csv`, `agent_summary.csv`, `paired_card_advantage.csv`,
  `run_metadata.json`, `run_parameters.csv`, and `agent_parameters.csv`
- detailed per-rotation `games.csv` report output with seeds, seat agents,
  winners, ranks, cards left, turns, and timeout flags; run-level and active
  per-agent hyperparameters are logged alongside the outcome CSVs
- `agent_summary.csv` reports per-agent standard errors and decision-volume
  counts for turns, forced passes, single legal moves, immediate wins, and
  sampled Monte Carlo decisions
- `paired_card_advantage.csv` reports standard errors, 95% confidence intervals,
  and oracle-gap columns when the paired baseline is an oracle agent
- `tune_eval.py`, a random-search tuning harness for heuristic weights and
  Monte Carlo run settings, ranking candidates by paired card advantage on
  shared duplicate-deal evaluations
- `EXPERIMENT_LOG.md`, a lightweight experiment notebook for recording serious
  tuning/evaluation questions, commands, seeds, report paths, results, and
  decisions
- optional MC-vs-heuristic decision tracing with
  `full_game_eval.py --trace-mc-heuristic`; when enabled, each run also writes
  `mc_heuristic_decisions.csv` with one row per comparable `Monte Carlo`
  decision, including legal moves, hand/table context, heuristic choice and
  reasons, Monte Carlo score/win-rate/margin/sample count, disagreement flag,
  and the final rank/cards-left outcome for that seat
- reduced-deck full-game evaluation through `--cards-per-suit`, using a rank
  window centered on 7, so `--cards-per-suit 5` runs ranks 5 through 9 in each
  suit while preserving the normal table/turn rules
- `proof_benchmark.py` now has an opt-in hard tier with thousand-state reduced
  exact-search positions; after the bitmask/cache-key pass, the current
  throughput suggests a 100k-state tier is a realistic next measurement target
  before treating millions of states as the longer-range scale question
- 52-bit card masks for compact card-set operations, table/legal-card masks,
  cached full-information hand masks, and compact exact-solver cache keys
- reusable hidden-deal sampler state for repeated Monte Carlo sampling
- shared hidden-deal samples across legal moves in Monte Carlo recommendation
- information-limited rollout through `rollout_information_limited(...)`, where
  each simulated player chooses from only their own private hand plus public
  table/history/count evidence
- mask-backed information-limited rollout through
  `rollout_information_limited_masks(...)`, including compact hand-mask
  updates, mask-based winner/count checks, mask-native policy choice fast paths,
  and Monte Carlo evaluation over completed hidden-deal masks
- `recommend_move_information_limited_monte_carlo(...)`, comparing legal moves
  over shared hidden-deal samples under a documented information-limited policy
- deterministic heuristic-greedy and softmax information-limited heuristic policies
- hidden-deal EV against the declared deterministic information-limited
  heuristic-greedy policy through `recommend_move_exact_information_limited_policy(...)`;
  this is exact over enumerated hidden deals, while terminal-without-timeout
  continuation remains the next proof-tightening step
- perfect-information counterpart rollout-EV oracle through
  `recommend_move_perfect_information_rollout_ev(...)`, mirroring the practical
  candidate-EV comparison shape while using the true full deal instead of
  hidden-deal sampling
- rollout bitmask/cache efficiency plan completed and folded into this proof
  plan: shared deterministic policy caches, bounded rollout transposition
  caches, mask hand helpers, mask-native policy fast paths, and mask-backed
  information-limited Monte Carlo evaluation are implemented
- hidden-deal enumeration regression tests for pass constraints, known hand
  counts, and impossible constraint/count combinations
- regression coverage that non-exhaustive exact recommendation reuses the same
  truncated hidden-deal set across all candidate moves
- regression coverage that full-information materialization derives trailing
  public pass counts
- regression coverage that proof benchmark tiers expose the intended hard
  positions and that a hard position reaches thousand-state exact-search scale
- regression coverage that proof evaluation reports exact EV regret for
  practical engine choices
- regression coverage for deadlock handling, root forced-pass reporting,
  forced-pass-chain equivalence, played-card ownership during hidden-deal
  enumeration, and vector-aware hidden-deal EV tie-breaking
- regression coverage that information-limited rollout does not expose hidden
  hands to the acting player's move policy, labels the continuation policy, and
  supports both greedy and softmax policy variants
- regression coverage that mask hand helpers match set helpers, mask after-play
  matches set after-play, and mask rollouts match set rollouts for deterministic
  heuristic-greedy and fixed-seed softmax scenarios
- regression coverage for exact reduced-state EV against the deterministic
  information-limited heuristic-greedy policy
- dead/redundant helper cleanup in `seven_hearts.py`: removed the obsolete
  scalar EV tie-breaker, old top-level hidden-deal assignment samplers that were
  superseded by `HiddenDealSampler`, and uncalled convenience/legal-card helpers
  that duplicated the active mask-based rules path

Current verification:

```text
py run_tests.py
112 tests passed

py proof_demo.py

py proof_benchmark.py

py proof_benchmark.py --include-hard

py proof_eval.py

py full_game_eval.py --deals 1 --samples-per-move 2 --rollout-max-turns 40

py full_game_eval.py --deals 2 --fast --progress-every 2 --max-turns 500

py full_game_eval.py --deals 2 --cards-per-suit 5 --samples-per-move 4 --rollout-max-turns 40 --progress-every 2 --max-turns 200

py full_game_eval.py --deals 1 --cards-per-suit 5 --samples-per-move 1 --rollout-max-turns 20 --max-turns 200 --oracle-gap --progress-every 0

py full_game_eval.py --deals 8 --cards-per-suit 8 --samples-per-move 4 --rollout-max-turns 80 --workers 4 --progress-every 4

py full_game_eval.py --deals 25 --cards-per-suit 5 --samples-per-move 16 --rollout-max-turns 40 --max-turns 200 --trace-mc-heuristic

py tune_eval.py --mode heuristic --candidates 16 --deals 100 --workers 4

For serious tuning/evaluation runs, add a dated entry to `EXPERIMENT_LOG.md`.

py -m py_compile seven_hearts.py test_seven_hearts.py run_tests.py proof_demo.py proof_benchmark.py proof_eval.py full_game_eval.py tune_eval.py demo.py simulator.py
```

Important current limitations:

- exact full-information search is suitable for reduced, late-game, and
  hand-authored states first. Current hard benchmarks are only at the 4k-state
  scale, but measured throughput suggests the next benchmark tier should push
  toward 100k states and possibly beyond with explicit runtime limits. Full
  52-card initial deals may still be too large without further optimization.
- deadlocks are counted and given neutral value; valid normal games should not
  hit them before first-out
- exact imperfect-information evaluation is proof-grade only when
  `exhaustive=True`, and even then it is exact only for the declared full-
  information-continuation oracle model
- the current hidden-deal EV oracle is not the final true imperfect-information
  optimal solver: after each hidden deal is materialized, all players are solved
  as if they can see the full deal
- when `max_deals` truncates hidden-deal enumeration, the result is diagnostic
  rather than proof-grade; because truncation is deterministic, no statistical
  standard error is reported for truncated exact-enumeration results
- hidden-deal enumeration currently uses the existing public constraint model
  and a uniform prior over enumerated deals
- the current Monte Carlo proof-evaluation report still shows regret on the
  multi-suit reduced-belief scenario, so practical sampled play needs more
  tuning/validation before it can be treated as engine-quality proof
- naive broad searches for harder hidden-belief proof positions can be very
  slow; one ad hoc multi-suit exploratory search timed out after roughly 23
  minutes without finding a useful position. Future position discovery should
  use bounded tooling with explicit limits, progress reporting, and abort
  controls.

Done in the current proof pass:

1. Built the practical imperfect-information solver path: evaluate candidate
   moves over shared hidden-deal samples where every simulated player uses only
   their own private hand and public evidence. The implementation includes
   heuristic-greedy and softmax heuristic policies.
2. Built hidden-deal EV against declared information-limited policies for
   reduced belief states, so the sampled real-game solver has proof-sized
   validation targets. The deterministic heuristic-greedy information-limited
   policy is exact over enumerated hidden deals; removing the remaining bounded
   continuation cutoff is now tracked as a proof-tightening task.
3. Kept the core implementation lean by removing redundant/dead helpers after
   the bitmask and reusable hidden-deal sampler paths became canonical.
4. Moved information-limited Monte Carlo rollout evaluation onto completed
   hand masks and a mask-native rollout path while preserving the set-based
   rollout for parity/debugging.
5. Closed the standalone rollout bitmask workplan as an active implementation
   plan. Remaining set-oriented scoring and opponent-model allocation is now
   tracked as general performance work rather than unfinished bitmask plumbing.

Highest-priority solver/proof work left:

1. Add terminal deterministic information-limited continuations for exact
   reduced-state EV. For deterministic policies such as heuristic-greedy, each
   materialized hidden deal should be advanced until a player empties their hand,
   without an arbitrary `max_turns` cutoff. A full pass cycle with no play should
   still be detected and reported defensively, even though valid full-deck games
   should not deadlock before first-out. Exact information-limited certificates
   should then report terminal outcomes rather than timeout rates.
2. Retire timeout scoring from exact terminal-continuation paths. The
   `monte_carlo_timeout_penalty` remains useful only for bounded rollout modes,
   stochastic softmax diagnostics, or defensive full-game harness limits. It
   should not affect an exact deterministic information-limited EV claim.
3. Validate softmax information-limited rollouts as a first-class practical
   policy option. The default deterministic heuristic-greedy rollout is useful
   for repeatability and proof-sized comparisons, but it can make opponent
   continuations too sharp by always choosing one heuristic-best response.
   Softmax runs should be evaluated as declared policy variants, with
   rationality, seeds, sample counts, rollout limits, timeout rates, and
   uncertainty reported so comparisons remain reproducible and statistically
   valid. Do not silently replace clean deterministic-policy baselines; report
   greedy and softmax as separate Monte Carlo configurations.
4. Add a conservative exact-enumeration path for small public belief spaces
   before falling back to sampled hidden deals. If the complete set of hidden
   deals can be proven to fit within the configured sample budget or an explicit
   enumeration cap, evaluate each deal once and share that same deal set across
   every candidate move. This is not confidence-based early stopping or a
   heuristic skip: it should preserve the same public belief distribution,
   remove duplicate with-replacement samples in low-uncertainty endgames, and
   report whether the deal set was exhaustive or sampled.
5. Treat CFR/perfect-recall and public-belief equilibrium methods as later
   reduced-deck research, not as the required next step for a usable real-game
   solver.
6. Extend harder benchmark discovery beyond the current 4k-state hard case:
   first target 100k-state reduced positions, then test whether larger
   500k/million-state cases are practical with explicit runtime limits and
   progress reporting.
7. Continue reducing remaining allocation in the practical solver, especially
   the `score_move(...)`/`build_opponent_model(...)` path that still
   materializes `PlayerKnowledge` and set-oriented opponent models for
   multi-legal policy decisions.
8. Add deeper hidden-deal enumeration regression tests around multi-pass
   histories.
9. Add more hand-authored imperfect-information snapshots with 3-5 cards per
   opponent and multiple public pass constraints.
10. Grow `proof_eval.py` into a broader engine scoreboard with more exact
   positions, repeated Monte Carlo seeds, and trend snapshots.
11. Tune and validate the practical sampled policies against the exact reduced
   proof scenarios, especially the current multi-suit case where Monte Carlo
   can pick a regretful move at the default sample count.
12. Scale duplicate-deal, seat-rotated full-game evaluation runs, using average
   cards left and paired card advantage as the primary practical metrics.

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

### Hidden-Deal Full-Information-Continuation EV

For solver player `i` and candidate legal action `a`, define:

```text
EV_i(P, a) =
  sum_{h in B(P)} Pr(h | P) * V_i(T(M(P, h), a))
```

where `V` is the full-information value function under the declared continuation
model. In the current implementation, that continuation model is full-
information rational play after each hidden deal is materialized. This is an
exact oracle for evaluating moves against fully revealed continuations; it is
not a true sequential imperfect-information optimum.

The exact hidden-deal oracle policy is:

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

The implementation now computes an expected value vector for every candidate
move and uses the same vector-aware tie-break structure as full-information
rational play.

### Perfect-Information Counterpart Oracle

For practical full-game evaluation, the target oracle should be the
perfect-information counterpart of our belief-state sampled EV agent, not merely
the older greedy full-information structural policy.

Our production agent evaluates candidate moves by sampling plausible hidden
deals from the public belief state, applying each candidate, continuing with a
declared information-limited policy, and averaging outcomes. The corresponding
oracle removes only the hidden-card uncertainty:

```text
given the true full deal:
  for each candidate legal move:
    apply the candidate move in the true full-information state
    evaluate or simulate continuation from that true state
  choose the move with best expected outcome under the declared oracle
  continuation model
```

This oracle is still an upper-bound diagnostic rather than a fair baseline,
because it sees hidden hands. But it is a cleaner comparison than the current
greedy oracle: it asks how much performance is lost because our agent must
reason from a belief state instead of the actual deal.

Implemented path:

1. Keep the current `rollout_oracle(...)` / `choose_oracle_move(...)` path as a
   legacy greedy full-information diagnostic.
2. Use the perfect-information rollout-EV oracle that mirrors the candidate
   comparison shape of `recommend_move_information_limited_monte_carlo(...)`,
   but uses the true full deal directly rather than sampling hidden deals.
3. Use exact full-information solving for this oracle only on reduced states
   where exhaustive search is tractable; use rollout evaluation for full games.
4. Report oracle gap against this counterpart oracle in full-game evaluation
   with `full_game_eval.py --oracle-gap`.

### True Imperfect-Information Target

The final project goal is stronger than the current hidden-deal oracle. A true
imperfect-information solver must evaluate future play where every player acts
from their own private hand plus public evidence, not from the fully materialized
deal.

Near-term bridge targets:

- sampled move EV over shared hidden deals, where every simulated player acts
  from their own information-limited view
- exact EV against declared information-limited policies on reduced belief
  states
- reduced-deck public-belief search where belief updates are part of the state
- CFR or another perfect-recall extensive-form method as later research if an
  equilibrium-style solution concept is needed

Possible future addition:

- **True reduced-deck imperfect-information optimal solver.** For small declared
  reduced decks, build an exact public-belief or perfect-recall solver that
  represents player information sets explicitly, updates beliefs after public
  plays and passes, and computes an optimal or equilibrium-style policy under a
  documented multiplayer solution concept. This would be a research/validation
  layer for proof-sized games, not the expected full-deck production engine.

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

## Validation Oracle: Exact Full-Information Recurrence

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

## Validation Oracle Deliverables

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

## Imperfect-Information Solver Plan

This is the main product path. The full-information exact solver can validate
small positions, but the practical solver must evaluate moves from the solver's
real information state and must not let future players use hidden cards they
could not know.

### Belief Set Construction

Given public state and solver knowledge, enumerate all hidden deals satisfying:

- every unseen card belongs to exactly one opponent
- known hand counts are satisfied exactly
- pass constraints are respected
- public played-card ownership is respected
- own hand and table cards are excluded

This should reuse the existing inference constraints where possible.

### Exact Hidden-Deal Oracle EV

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

This remains a validation oracle when the continuation uses full-information
values. The production imperfect-information solver should instead use
information-limited continuations, where each future player acts from their own
private hand and the public history.

### Practical Information-Limited Rollout EV

For each legal solver move:

```text
sample shared hidden deals from the current public belief
for each candidate move:
  for each sampled hidden deal:
    apply the candidate move in the true sampled world
    continue the game with rollout_information_limited(...)
    every simulated player receives only their own hand + public evidence
  report first-out EV, standard error, timeout rate, and sample count
```

This is the immediate next implementation target because it matches the
real-game use case more closely than full-information continuation search.

For deterministic continuation policies, the proof-sized exact variant should
not use a rollout cutoff. It should run each materialized hidden deal to a
terminal first-out outcome:

```text
for each candidate move:
  for each hidden deal in the exhaustively enumerated belief set:
    apply the candidate move in the true hidden deal
    continue with the declared deterministic information-limited policy
    stop only when a player empties their hand
    detect and report a full-pass-cycle deadlock defensively
  report exact first-out EV and terminal outcome counts
```

For valid full-deck games, a full pass cycle before first-out should be
unreachable because some held card must be public-legal until the game has
advanced. Reduced decks can omit outer cards that the public table would
normally ask for, so exact reduced-deck tooling should still report a deadlock
if one appears rather than looping forever.

Once this terminal deterministic path exists, timeout rate and timeout penalties
are obsolete for exact deterministic information-limited EV. Timeout reporting
remains appropriate for bounded rollout experiments, stochastic softmax
rollouts, and full-game harness safety limits.

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

The solver should not claim that irrational opponents always lose. That is too
strong, especially in a multiplayer hidden-information game.

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

For the real-game imperfect-information solver, the analogous claim is dynamic
optimality under the declared belief model:

```text
At each solver decision point, observe the public history h and private hand,
update the belief state using the actual plays and passes, and choose the move
that maximizes first-out EV under the declared information-limited model.

If an opponent makes a move that differs from the modeled policy, the previous
recommendation is not retroactively guaranteed to be best for that unmodeled
future. Instead, the deviation becomes new public evidence. The solver updates
the belief state and re-solves from the new information state.
```

This lets the project say something stronger than "works only if opponents
follow the script": the solver remains locally EV-optimal after every observed
history under its stated model. Opponent deviations are handled by re-solving,
not by pretending the original plan covered every possible future line.

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

Near-term practical-simulation optimizations:

- reuse one hidden-deal sampler across repeated Monte Carlo samples so
  count-consistent dynamic-programming tables are built once per public state
- compare candidate moves over the same sampled hidden deals to reduce variance
  and avoid repeated sampler setup
- use 52-bit card masks for public legal cards, table cards, hand masks, and
  exact-solver cache keys
- reuse sampled deals across benchmark scenarios when comparing strategy
  variants on the same public position
- cache information-limited policy evaluations by public state, private hand,
  player, and weight set when that policy is deterministic
- keep exact continuation caches shared across candidate moves and hidden deals
  for proof-sized oracle evaluations
- benchmark CPU state throughput before adding GPU-oriented work; the current
  symbolic search and deal sampling are CPU/RAM-bound, while a 1660 Ti 6GB
  becomes useful mainly for later neural approximators or GPU-batched rollouts

Clean FullMC evaluation optimization plan:

The current evaluation question is whether the belief-state Monte Carlo agent
adds marginal value over the one-ply heuristic. Efficiency work for that
experiment must not change the decision policy being evaluated. The clean
`FullMC` agent should run Monte Carlo on every meaningful decision and skip only
non-decisions: forced passes, single legal moves, and immediate winning moves.

Phase 1 focuses on using compute better without changing choices:

- implemented: multiprocessing across independent duplicate-deal rotation games
  through `full_game_eval.py --workers`
- implemented: deterministic per-game seeds, so serial and parallel runs use
  the same random stream per rotation game
- preserved: fixed sample counts and fixed rollout limits for every meaningful
  `FullMC` decision
- implemented: progress reporting plus complete per-run report directories,
  including `games.csv` for per-rotation auditability and metadata/parameter
  CSVs for eval args, Monte Carlo settings, and heuristic weights
- implemented: optional `--trace-mc-heuristic` decision-level audit output,
  which does not change `FullMC` decisions but records where the Monte Carlo
  policy agrees or disagrees with the same-position heuristic policy
- supported: reduced-deck tiers, such as `--cards-per-suit 8`, as explicitly
  labeled clean evaluation tiers before scaling to full-deck games

Phase 2 keeps the same `FullMC` decision semantics while reducing inner-loop
cost:

- profile reduced-deck Monte Carlo runs before rewriting internals
- implemented: immediate-win shortcut: if a candidate move empties the current
  player's hand, choose that terminal move without sampling
- cache hidden-deal sampler construction per decision state
- implemented: cache deterministic heuristic-greedy information-limited rollout policy
  choices by public state, actor hand, hand counts, deck, and weight/policy
  settings inside each Monte Carlo decision
- implemented: promote deterministic heuristic-greedy rollout policy caching from
  per-decision scope to a bounded per-game cache
- implemented: add a rollout transposition cache for deterministic continuation policies,
  keyed by compact public table/history state, current player, hand masks, hand
  counts, deck, weights, policy settings, and remaining rollout budget
- implemented: add mask hand helpers and a mask-native
  `rollout_information_limited_masks(...)` path using compact hand updates,
  mask-based terminal/count checks, and mask-aware transposition cache keys
- implemented: wire information-limited Monte Carlo evaluation to complete
  hidden deals as hand masks and call the mask rollout path
- remaining: reduce repeated `PlayerKnowledge` and opponent-model materialization
  in multi-legal policy scoring
- add a conservative forced-chain win detector only if the win is provable from
  public state plus the actor's known hand; false negatives are acceptable, but
  false positives would invalidate the clean evaluation
- optionally parallelize hidden-deal samples within a decision only after
  game-level multiprocessing is measured

Explicitly out of scope for the clean `FullMC` evaluation:

- adaptive sample budgets, confidence gates, low-impact heuristic skips, timeout
  fallbacks, or any shortcut that allows the heuristic model to override a
  meaningful Monte Carlo decision
- replacing Monte Carlo with exact late-game solving in the primary `FullMC`
  agent; that should be reported as a separate agent variant if added

Potential Phase 3, for a later product agent, is selective/adaptive Monte Carlo:
use heuristic gates, high-impact triggers, adaptive sample budgets, and
fallbacks to spend rollout compute only where it is useful. This is not part of
the clean `FullMC` versus `Heuristic` evaluation because it changes the policy
being compared.

## Suggested Implementation Sequence

Completed foundation:

1. Implement and test the rules engine, legal move generation, transitions, pass
   inference, and card conservation.
2. Build constrained hidden-deal sampling/enumeration from public evidence.
3. Build the exact full-information validation oracle and proof certificates.
4. Build exact hidden-deal full-information-continuation EV for small belief
   sets as a second validation oracle.
5. Add shared sampled deals, reusable sampler state, and bitmask-backed legal
   and exact-search operations.

Immediate product sequence:

1. Implement `rollout_information_limited(...)`.
2. In that rollout, give each simulated player only their own hand plus public
   table/history/count evidence when choosing a move.
3. Use the existing heuristic scorer as the first deterministic
   information-limited policy.
4. Add a terminal deterministic continuation path for exact reduced-state
   information-limited EV. It should follow the declared deterministic policy
   until first-out, detect impossible full-pass-cycle deadlocks defensively, and
   avoid arbitrary turn cutoffs.
5. Remove timeout penalties from exact deterministic information-limited EV
   reporting. Keep timeout fields only for bounded/stochastic rollout modes and
   harness safety limits.
6. Add a softmax policy variant with a rationality parameter, so rollouts can
   test random, noisy, and greedy opponents with the same score function.
7. Add `recommend_move_information_limited_monte_carlo(...)`, comparing legal
   moves over shared hidden deals and reporting EV, standard error, samples,
   timeout rate when the continuation is bounded, and policy name.
8. Build exact reduced-state information-limited EV snapshots to validate
   terminal deterministic continuations and sampled rollout behavior where
   exhaustive evaluation is tractable.
9. Expand benchmark discovery toward 100k-state exact-oracle cases and larger
   sampled imperfect-information scenarios with explicit runtime limits.
10. Treat CFR/public-belief equilibrium methods as later research after the
   belief sampler and information-limited rollout engine are strong.

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

The project can claim "exact hidden-deal EV under full-information
continuation" for a specific position when:

- every hidden deal consistent with public evidence is enumerated
- the hidden-deal prior is documented
- each hidden deal is evaluated using the full-information exact solver
- each candidate move's expected value is computed exactly
- the final certificate reports the hidden deal count and expected value of each
  legal move
- the certificate clearly states that players use full-information rational
  continuation after each materialized deal

The project can claim "exact hidden-deal EV under a deterministic
information-limited policy" for a specific position when:

- every hidden deal consistent with public evidence is enumerated
- the hidden-deal prior is documented
- each simulated player chooses only from their own private hand and public
  evidence under the declared deterministic policy
- each materialized continuation is run to terminal first-out outcome without an
  arbitrary `max_turns` cutoff
- any full-pass-cycle deadlock is reported explicitly as a defensive anomaly
- each candidate move's expected value is computed from terminal outcome counts
- the certificate reports hidden deal count, terminal outcome counts, and the
  declared continuation policy
- timeout penalties do not affect the exact value

The project should not claim "true imperfect-information optimality" until:

- future players' information sets are modeled explicitly
- belief updates are part of the recurrence or equilibrium algorithm
- players do not receive hidden cards they could not know from their own hand
  and public evidence
- the solution concept is documented, such as best response to declared
  information-limited policies, an equilibrium approximation, or another
  explicit multiplayer hidden-information objective

The project may claim "true reduced-deck imperfect-information optimality" for a
specific reduced game only when:

- the reduced deck and public/private information model are declared
- information sets are explicit and no player observes hidden cards outside
  their information set
- public plays and passes update beliefs inside the recurrence or equilibrium
  algorithm
- the multiplayer solution concept is documented
- all reachable information states for that reduced game are solved exactly, or
  any approximation error is explicitly bounded and reported
- certificates distinguish the reduced-deck result from any full-deck practical
  solver claim

The project can claim "practical imperfect-information solver" when:

- hidden deals are sampled or enumerated from the documented public belief
- candidate moves are compared over the same hidden deals when sampling is used
- every simulated player acts only from their own private hand and public
  evidence, under a documented information-limited policy
- the result reports move EV, sample count, uncertainty, the opponent/self
  policy used in continuations, and timeout rate only when the continuation is
  bounded rather than terminal
- exact proof-sized information-limited and full-information oracle positions
  are used as regression checks
- certificates and reports clearly separate exact proof results from sampled
  real-game recommendations

The project can claim "Monte Carlo evidence" when:

- hidden deals are sampled from the documented belief distribution
- sample count is reported
- standard error is reported
- rollout policy is documented
- the result is not described as an exact proof

## Full-Game Evaluation Protocol

The proof layers above validate specific solver claims, but full-game strength
should be evaluated separately. 7 of Hearts is a four-player,
incomplete-information, high-luck, seat-order-dependent game, so raw win rate is
not enough by itself.

The practical evaluation objective is:

```text
On the same deals and seat positions, our agent should finish with fewer cards
left than baseline agents.
```

The primary full-game metric is average cards left. Lower is better. The
equivalent payoff is:

```text
payoff = -cards_left
```

Because this is a direct sign flip, average payoff is optional in reports; the
human-readable headline should be average cards left.

Full-game evaluation should continue play after the first player empties their
hand, at least until all players finish or only one player remains with cards.
This allows true rank and finishing-distribution metrics. The exact solver may
continue to use first-out utility internally; evaluation is allowed to report
the richer terminal ordering and remaining-card outcomes.

### Duplicate-Deal Seat Rotation

The practical benchmark should use duplicate deals and seat rotation:

1. Generate a random full deal.
2. Reuse the exact same card allocation across multiple games.
3. Rotate which agent occupies each seat.
4. Record cards left, payoff, rank, winner, and timeout status.

The first target is 1,000 random deals with four cyclic rotations, for 4,000
games per four-agent table. Larger runs can scale from there after runtime and
confidence intervals are measured.

The four default rotations for agents `A, B, C, D` are:

```text
Game 1: A seat 0, B seat 1, C seat 2, D seat 3
Game 2: D seat 0, A seat 1, B seat 2, C seat 3
Game 3: C seat 0, D seat 1, A seat 2, B seat 3
Game 4: B seat 0, C seat 1, D seat 2, A seat 3
```

All 24 seat permutations can be added later if compute is cheap enough.

### Agents

The initial evaluation agents are:

- `Random`: choose uniformly among legal moves, pass only when forced.
- `GreedyFurthestFromSeven`: choose the legal card with maximum
  `abs(rank - 7)`, using deterministic card order to break ties.
- `Heuristic`: choose the highest one-ply `score_move(...)` result.
- `Monte Carlo`: the belief-state sampled EV agent, using shared hidden deals and a
  documented information-limited continuation policy.
- `Oracle`: a full-information reference that may see sampled true hands during
  continuation. For final evaluation, this should be the perfect-information
  counterpart of `Monte Carlo`: same candidate-EV comparison shape, but with the true
  deal known instead of a sampled belief state. It is an upper-bound diagnostic,
  not a fair production agent.

`Monte Carlo` should not be described as a true optimal imperfect-information solver.
That stronger claim requires explicit information sets, belief updates inside
the recurrence or equilibrium method, and a documented multiplayer hidden-
information solution concept. The current claim is practical and narrower:

```text
belief-state sampled first-out EV under a declared information-limited policy
```

### Full-Game Metrics

Reports should include:

- average cards left, lower is better
- average payoff, higher is better, if useful for formal tables
- win rate
- average rank, lower is better
- finishing distribution
- timeout rate
- 95% confidence intervals

The main comparison statistic is paired card advantage:

```text
paired_card_advantage = baseline_cards_left - ours_cards_left
```

Positive values mean our agent finished with fewer cards than the baseline on
matched deal/seat conditions. Confidence intervals for this statistic should be
computed over the paired differences, not by treating the two agents as
independent samples.

### Initial Benchmark Suite

The first practical benchmark suite should contain:

1. Basic mixed table:

```text
Monte Carlo vs Random vs GreedyFurthestFromSeven vs Heuristic
```

2. Fixed opponent pools:

```text
Monte Carlo vs Random x3
Monte Carlo vs GreedyFurthestFromSeven x3
Monte Carlo vs Heuristic x3
```

3. Strong mixed table:

```text
Monte Carlo vs GreedyFurthestFromSeven vs Heuristic vs the strongest non-oracle rollout baseline
```

4. Paired card advantage against each non-oracle baseline.

5. Oracle-gap analysis, reported separately:

```text
oracle_gap = ours_cards_left - oracle_cards_left
```

Lower oracle gap is better. The oracle should be used to estimate headroom, not
as evidence that normal opponents are being modeled fairly. The older greedy
full-information oracle is useful for regression and debugging, but should not
be the main oracle-gap target once the perfect-information counterpart oracle is
implemented.

### Evaluation Implementation Status

Implemented now:

- full rules engine and legal move generation
- one-ply heuristic recommender
- constrained belief sampling and hidden-deal enumeration
- information-limited rollout and Monte Carlo move recommendation
- exact reduced-state EV against the deterministic information-limited heuristic-greedy
  policy
- legacy greedy full-information oracle rollout with one-ply response penalty
- proof-position reports comparing practical engines against exact
  full-information-continuation oracle values
- full-game evaluator that continues after first-out to final ranks/cards-left
  outcomes
- duplicate-deal, cyclic seat-rotated benchmark harness
- paired card advantage tables with standard errors and 95% confidence intervals
- per-agent full-game summary standard errors and decision-volume counts
- `GreedyFurthestFromSeven` full-game baseline
- perfect-information counterpart oracle for oracle-gap evaluation, with oracle
  gap folded into the paired card advantage report
- random-search tuning reports for heuristic weights and Monte Carlo settings

Not implemented yet:

- true reduced-deck imperfect-information optimal solver with explicit
  information sets and belief updates
- 1,000-deal practical benchmark run

Redundant or legacy under the updated plan:

- average payoff as a separate headline metric is redundant with average cards
  left because `payoff = -cards_left`
- hand-strength bucket analysis is intentionally deferred because the bucket
  definition is subjective and not needed for the core claim
- the current greedy full-information oracle should remain as a regression and
  debugging tool, but it should not be the final oracle-gap reference
- first-out-only complete-game simulation is useful for solver-objective sanity
  checks, but final full-game evaluation should continue beyond first-out

## Long-Term Goal

The long-term proof architecture should look like this:

```text
Rules engine
  -> belief model and constrained hidden-deal sampling/enumeration
    -> information-limited policies using private hand + public evidence
      -> sampled practical imperfect-information move EV for real games
        -> exact information-limited EV on reduced belief states
          -> exact full-information and hidden-deal oracle validation cases
            -> true reduced-deck imperfect-information optimal solver if needed
              -> EV agent tuned and validated against exact/sampled evidence
```

The belief-state sampled EV agent is the central practical solver. The heuristic
policy remains important because it is cheap, inspectable, and useful as a
default information-limited continuation policy, a baseline, and a fallback when
EV rollout is too expensive. The proof layer gives the EV agent and its
continuation policies exact/reduced-deck targets to validate against and helps
explain where approximation begins.
