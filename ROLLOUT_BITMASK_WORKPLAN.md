# Clean Rollout Efficiency Workplan

Status: closed and folded into `SOLVER_PROOF.md`.

The core rollout bitmask/cache work is complete:

- shared deterministic information-limited policy caches
- bounded deterministic rollout transposition caches
- mask hand helpers
- mask-native information-limited rollout
- mask-native policy-choice fast paths
- information-limited Monte Carlo evaluation over completed hidden-deal masks

The remaining set-oriented work, mainly `score_move(...)` and
`build_opponent_model(...)` materialization during multi-legal policy decisions,
is now tracked as general performance work in `SOLVER_PROOF.md`, not as
unfinished bitmask plumbing. This file is retained as an archive of the
completed implementation sequence and invariants.

This is a focused sub-plan for the next clean-safe efficiency work in
information-limited Monte Carlo rollouts. It covers the deeper rollout bitmask
rewrite plus the two highest-value cache optimizations that should preserve the
policy being evaluated.

## Goal

Reduce repeated rollout work and per-rollout allocation while preserving exactly
the same choices, scores, random behavior, and public-state semantics for clean
`FullMC` evaluation.

This plan is in scope for clean evaluation only if every phase is
behavior-preserving. It must not introduce adaptive sample budgets, heuristic
skips, timeout fallbacks, approximate policies, or exact-solver replacement.

Proof-tightening note:

- For deterministic information-limited policies such as heuristic-greedy, the
  exact reduced-state EV path should eventually use a terminal continuation that
  runs each materialized hidden deal until a player empties their hand, with a
  defensive full-pass-cycle deadlock check instead of an arbitrary rollout
  cutoff.
- Once that terminal path exists, `monte_carlo_timeout_penalty` is obsolete for
  exact deterministic information-limited EV. Timeout fields and penalties
  remain relevant only for bounded rollout experiments, stochastic softmax
  diagnostics, and full-game harness safety limits.

## Current State

Already implemented:

- `Card` to compact index/bit helpers: `card_index`, `card_from_index`,
  `card_bit`, `card_set_mask`, `mask_to_cards`, and `maybe_only_card`
- public legal-card masks through `GameState.public_legal_mask()`
- sampler holder constraints using masks in `HiddenDealSampler`
- exact-solver compact hand masks and cache keys
- information-limited heuristic-greedy policy cache keys using actor hand masks and deck
  masks
- immediate-win shortcut before Monte Carlo sampling
- mask hand helpers: `hand_masks_from_hands`, `mask_hand_count_tuple`,
  `first_empty_player_mask`, `mask_after_play`, and `complete_hand_masks`
- mask-native information-limited rollout through
  `rollout_information_limited_masks(...)`
- mask-native policy chooser fast paths through
  `choose_information_limited_move_from_mask(...)`
- information-limited Monte Carlo evaluation now completes sampled hidden deals
  to hand masks and calls the mask rollout path

Still set-heavy:

- the legacy public/debug rollout path `rollout_information_limited(...)` still
  copies all hands as sets
- set-based helpers such as `complete_hands(...)`, `apply_known_play(...)`,
  `first_empty_player(...)`, and `hand_count_tuple(...)` remain for oracle,
  full-game, parity, and debugging paths
- `score_move(...)` and `build_opponent_model(...)` still expect
  `PlayerKnowledge` with `frozenset[Card]`

Current cache status:

- deterministic heuristic-greedy information-limited rollout policy choices are cached
  inside a single Monte Carlo decision
- deterministic heuristic-greedy policy choices are also cached in a bounded
  per-game cache during full-game simulation
- full deterministic rollout continuations are cached by transposition for
  `policy == "heuristic_greedy"` and disabled for softmax

## Non-Negotiable Invariants

- For fixed seed, legal position, sample count, rollout limit, policy, and
  weights, the selected move must remain identical.
- For deterministic heuristic-greedy rollout policy, rollout results must remain
  identical.
- For softmax rollout policy, random draws must occur in the same order and
  only when the old path would have drawn.
- Public history, hand counts, and pass legality must remain unchanged.
- No sampled deal may be accepted or rejected differently.
- No shortcut may choose a move unless that move is already forced by the same
  rules or is a proven immediate win from the actor's known hand.

## Phase 0: Baseline Measurement

Status: superseded by the existing full-game evaluation and tuning report
harnesses. A separate microbenchmark script was not required to complete the
bitmask/cache rollout implementation.

Add a small benchmark script or mode that avoids pytest and reports elapsed time
for a fixed reduced-deck `FullMC` evaluation.

Suggested benchmark:

- use `cards_per_suit=8`
- run one duplicate-deal comparison
- include at least one `information_limited_monte_carlo` agent
- fixed seed
- fixed `samples_per_move` and `rollout_max_turns`
- print total time, decisions, samples, and rollouts if available

Acceptance:

- benchmark is deterministic enough for before/after comparisons
- benchmark does not change production logic

## Track A: Promote Greedy Policy Cache

Status: implemented as a bounded per-game cache, with optional cache arguments
available for direct Monte Carlo recommendation/evaluation calls.

This is likely the lowest-risk high-value cache optimization. It extends the
already implemented per-decision deterministic heuristic-greedy policy cache to a bounded
per-game or per-worker cache.

### A1: Add Optional Shared Policy Cache Plumbing

Status: implemented.

Thread an optional `InformationLimitedPolicyCache` through the full-game
simulation path without changing defaults.

Candidate call path:

- `simulate_full_game_to_completion(...)`
- `choose_full_game_agent_move(...)`
- `recommend_move_information_limited_monte_carlo(...)`
- `evaluate_move_information_limited_monte_carlo_from_deals(...)`
- `rollout_information_limited(...)`
- `choose_information_limited_move(...)`

Implementation notes:

- keep the existing per-decision cache if no shared cache is provided
- use the existing `information_limited_policy_cache_key(...)` unchanged
- use the shared cache only for `policy == "heuristic_greedy"`; legacy
  `"greedy"` inputs are normalized to this name
- do not use this cache for softmax policy

Acceptance:

- fixed-seed full-game decisions are identical before and after
- direct tests show shared-cache and per-decision-cache recommendations match
- `py run_tests.py` passes

### A2: Bound The Shared Cache

Status: implemented with `BoundedInformationLimitedPolicyCache`.

Replace the raw dict with a small bounded LRU-like cache if memory measurements
show unbounded growth.

Implementation options:

- use `collections.OrderedDict`
- keep a simple max-entry count, for example 50,000 or configurable
- evict least-recently used entries on insertion

Acceptance:

- cache hits/misses can be counted in benchmark output or debug metrics
- bounded cache produces the same decisions as unbounded cache
- memory use does not grow without limit during long duplicate-deal evals

## Track B: Deterministic Rollout Transposition Cache

Status: implemented for deterministic heuristic-greedy rollouts and disabled for softmax.

This can avoid replaying identical deterministic continuations. It is more
delicate than Track A because the key must include every input that affects the
rollout result.

Only apply this cache when the rollout policy is deterministic:

- allowed: `policy == "heuristic_greedy"`; legacy `"greedy"` is accepted as an alias
- not allowed: `policy == "softmax"` unless RNG state is included in the key,
  which is probably not worth it

### B1: Define A Complete Rollout Cache Key

Status: implemented as `rollout_transposition_cache_key(...)`.

Proposed key fields:

- canonical table runs
- public history, including passes and played cards
- current player
- hand masks for all four players
- public hand counts
- deck mask
- weights by player or a canonical single `StrategyWeights`
- policy name and rationality
- remaining rollout budget

Implementation notes:

- do not omit history unless proven irrelevant to all public inference used by
  the policy
- include remaining budget because timeout behavior depends on it
- include per-seat weights if rollout weights are passed as a mapping
- use compact integer masks for hands and deck

Acceptance:

- tests demonstrate two identical cache keys produce identical
  `RolloutResult`
- tests demonstrate changing budget, hand masks, current player, or weights
  changes the key
- no cache is used for softmax

### B2: Add Private Cache To The Existing Set Rollout

Status: implemented with `BoundedRolloutTranspositionCache`.

Before combining this with the bitmask rewrite, add the transposition cache to
the existing set-based rollout path behind an optional argument.

Implementation notes:

- check cache at the start of each recursive or iterative continuation state
- because current rollout is iterative, either:
  - cache only whole-call results first, or
  - refactor deterministic heuristic-greedy continuation into a recursive helper that can
    cache suffixes
- whole-call caching is simpler but may have lower hit rate

Acceptance:

- deterministic heuristic-greedy rollout result is identical with cache enabled/disabled
- cache is unused for softmax
- `py run_tests.py` passes

### B3: Share Rollout Cache Across Candidate Moves Within A Decision

Status: implemented for `policy == "heuristic_greedy"`.

Once B2 is correct, pass the same rollout transposition cache across candidate
move evaluations within `recommend_move_information_limited_monte_carlo(...)`.

Acceptance:

- same selected move and score fields on fixed seeds
- benchmark reports cache hit rate and elapsed-time change

## Track C: Rollout Bitmask Rewrite

This track reduces allocation and repeated set/card conversion in
`rollout_information_limited(...)` and its hot callees.

Track C should come after Track A unless profiling shows set churn dominates.
It can run before or after Track B, but each track should be validated
independently before combining them.

## C1: Add Mask-Hand Helpers

Status: implemented.

Introduce small helpers without changing the rollout path yet.

Proposed helpers:

- `hand_masks_from_hands(hands) -> tuple[int, int, int, int]`
- `mask_hand_count_tuple(hand_masks) -> tuple[int, int, int, int]`
- `first_empty_player_mask(hand_masks) -> int | None`
- `mask_after_play(hand_masks, player, card) -> tuple[int, int, int, int]`
- `complete_hand_masks(knowledge, hidden_deal) -> tuple[int, int, int, int]`

Implementation notes:

- use `int.bit_count()` for hand counts
- keep helper return order fixed by player index `0..3`
- do not remove existing set-based helpers

Acceptance:

- direct tests show mask helper results match existing set helper results
- no behavior change in existing rollout code

## C2: Mask-Based Rollout Skeleton

Status: implemented as `rollout_information_limited_masks(...)`.

Add a new private/internal rollout path, for example
`rollout_information_limited_masks(...)`, that mirrors
`rollout_information_limited(...)` but stores hands as a tuple of masks.

Initial version may still materialize actor hands when calling
`choose_information_limited_move(...)`.

Required behavior:

- winner detection uses `first_empty_player_mask(...)`
- hand counts use `mask_hand_count_tuple(...)`
- playing a card updates only the active player's mask
- public state advances with `state.after_play(player, card)`
- pass behavior remains `state.after_play(player, None)`

Acceptance:

- existing set rollout and mask rollout produce identical `RolloutResult` for
  deterministic heuristic-greedy scenarios
- softmax scenarios are tested with fixed seeds and identical results where RNG
  draw order is expected to match
- current public `rollout_information_limited(...)` still defaults to the
  existing path until this phase is proven

## C3: Mask-Aware Policy Choice

Status: implemented as `choose_information_limited_move_from_mask(...)`.

Add a mask-native variant of the policy chooser, for example
`choose_information_limited_move_from_mask(...)`.

Target behavior:

- compute legal moves with `hand_mask & state.public_legal_mask()`
- use `maybe_only_card(...)` for single legal move
- use `mask_to_cards(...)` only for the final legal move list needed by scoring
- reuse the existing information-limited policy cache key directly
- construct `PlayerKnowledge` only after legal/no-legal/single-legal fast paths

Acceptance:

- same chosen card as `choose_information_limited_move(...)` for heuristic-greedy policy
- same softmax distribution path for fixed seed
- no cache-key weakening

## C4: Wire Mask Rollout Into Monte Carlo Evaluation

Status: implemented for information-limited Monte Carlo evaluation.

Change only the information-limited Monte Carlo path to use the mask rollout
after C2 and C3 pass.

Target functions:

- `evaluate_move_information_limited_monte_carlo_from_deals(...)`
- optionally `evaluate_move_information_limited_monte_carlo(...)`

Suggested transition:

1. Convert completed hidden deals to hand masks once per sample.
2. Apply the candidate known play by clearing the actor's card bit and advancing
   public state.
3. Call `rollout_information_limited_masks(...)`.
4. Keep the public set-based function available for tests and debugging.

Acceptance:

- `py run_tests.py` passes
- fixed-seed full-game smoke result matches before/after for a small reduced
  deck
- benchmark from Phase 0 shows no correctness drift and reports speed delta

## C5: Remove Remaining Avoidable Set Churn

Status: folded into general performance work in `SOLVER_PROOF.md`. The
information-limited Monte Carlo path now uses completed hand masks and the mask
rollout. Further reductions in `PlayerKnowledge`, `score_move(...)`, and
`build_opponent_model(...)` materialization are useful but are no longer treated
as unfinished bitmask plumbing.

Only after the mask rollout is the default for information-limited Monte Carlo:

- avoid `complete_hands(...)` in the information-limited MC path
- avoid rebuilding full hand-count tuples from sets inside rollouts
- avoid actor hand materialization before no-legal and single-legal checks
- consider local caching of `state.public_legal_mask()` per rollout state if
  profiling shows it is hot

Acceptance:

- no public API break unless deliberately documented
- same clean-eval decisions on fixed seeds
- benchmark improvement is measurable

## Suggested Execution Order

1. Phase 0 baseline measurement.
2. Track A shared deterministic heuristic-greedy policy cache.
3. Track B rollout transposition cache, starting with whole-call caching.
4. Track C mask rollout rewrite.
5. Re-run baseline after each track and record speedup separately.

This order prioritizes the likely biggest wins first while keeping each change
independently testable.

## Tests To Add

Add tests to `test_seven_hearts.py` and register them in `run_tests.py`.

Suggested focused tests:

- shared policy cache parity with per-decision policy cache
- shared policy cache disabled for softmax
- rollout transposition key changes when any semantic input changes
- rollout transposition cache parity for deterministic heuristic-greedy rollouts
- rollout transposition cache disabled for softmax
- mask helper parity with set helper for hand counts and first-empty player
- mask after-play parity with `hands_after_play(...)`
- mask rollout parity with set rollout for deterministic heuristic-greedy policy
- no-legal pass parity
- immediate winner parity
- fixed-seed softmax parity if the mask path supports softmax directly
- information-limited MC fixed-seed selected move parity before and after mask
  path activation

Do not use pytest as the primary validation command. Use:

```powershell
py run_tests.py
```

Targeted direct checks are also fine:

```powershell
py -c "import test_seven_hearts as t; t.some_test(); print('ok')"
```

## Rollback Plan

Keep the set-based rollout function intact until the mask path is fully
validated. If any parity test fails:

- leave the public path using the existing set implementation
- keep completed helper tests if they are correct
- debug the mask path behind a private function

## Things Not To Do In This Workplan

- no adaptive sample budgets
- no heuristic gates or low-impact skips
- no confidence-based early stopping
- no timeout fallback decisions
- no exact late-game solver substitution in `FullMC`
- no approximate opponent model changes
- no forced-chain win detector unless separately proven conservative
