# 7 of Hearts Solver Project

## Game Rules

7 of Hearts, sometimes called Sevens, is played with a standard 52-card deck by 4 players.

Each player is dealt 13 cards. The objective is to be the first player to get rid of all cards in hand.

The game starts with the player who holds the 7 of hearts. That player must play the 7 of hearts as the opening move.

Cards are played into suit sequences. Each suit begins with its 7. After a suit has been opened, cards in that suit may only be played if they are numerically adjacent to cards already on the table.

For example, if the 7 of hearts is on the table, the legal heart plays are:

- 6 of hearts
- 8 of hearts

If the 6 of hearts is then played, the legal heart plays become:

- 5 of hearts
- 8 of hearts

The same structure applies to every suit. A suit cannot be played at all until its 7 has been played.

Aces are low. Rank order is:

```text
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
```

On a player's turn:

- If the player has at least one legal move, they must play one legal card.
- If the player has no legal move, their turn is skipped.

Play proceeds clockwise. The game continues until all players except one have emptied their hands, but the strategic goal for this project is to choose moves that maximize the chance of being the first player out.

## Strategic Ideas

The game is centered around controlling which future moves become available.

Important strategic patterns include:

- Playing a card may open future moves for yourself.
- Playing a card may also open future moves for opponents.
- Holding a 7 can block an entire suit from being played.
- Holding a bottleneck card, such as a 6 or 8 next to an opened 7, can delay one side of a suit.
- End cards, especially aces and kings, may become liabilities unless the chain toward them is likely to open.

For example, if a player only holds the 7 of spades in spades, they may want to delay playing it because no other spades can be played until that 7 appears. However, if the same player also holds the ace of spades, opening spades earlier may help create enough turns for the lower spade sequence to reach the ace.

## Algorithmic Design Principles

The solver should prefer clean causal logic over special-case strategic moods.

The guiding assumption is that if the game has already been played well, the current position should be evaluated by the same principles throughout the game:

- What cards become playable?
- Who is likely to hold those cards?
- Who gets to act soonest?
- How much future chain control is being released?
- How much runway does the solver need for its own blocked cards?

Endgame situations are important, but they should not automatically create a separate scoring mode. The optimal move should emerge from the same card-specific factors that matter earlier in the game. For example, opening a move for the next opponent is dangerous because of turn order and likely ownership, not because an abstract "endgame urgency" bonus says so.

This means:

- Avoid adding flat score terms that apply equally to every legal move, because they cancel out in the recommendation.
- Avoid panic-style urgency knobs unless they modulate card-specific effects and are supported by testing.
- Prefer simple, inspectable heuristics over complicated weighting that does not match how the game is reasoned about.
- Treat longer released chains as more dangerous because they expose more future cards, but avoid double-counting chain length without evidence.
- Keep hand counts as useful information, especially for hidden-card inference and diagnostics, but do not let them override the direct move logic unless simulation or self-play supports it.

In short, the solver should reason inductively from the current position: if prior moves were chosen well, the next move should be chosen by the same structural principles, not by switching to a different endgame personality.

## Project Scope

The goal of this project is to develop an algorithmic strategic solution for 7 of Hearts, starting with a next-move generator.

The solver should operate under realistic information, not full information.

The solver may know:

- Its own hand.
- Cards already played to the table.
- The current open interval for each suit.
- Whose turn it is.
- Player hand counts, if available.
- Which players have passed on previous turns.
- Publicly observable move history.

The solver should not know:

- The exact cards in opponents' hands.
- The exact order of the remaining unseen cards.

The solver should reason probabilistically about unseen cards using the information available from the current game state and move history.

In this project, unseen cards means cards that are not in the solver's hand and not already on the table. Because the full deck is dealt at the start, every unseen card must be in one of the opponents' hands.

## Initial Solver Direction

The first version should generate all legal moves for the current player and score them with a heuristic.

The heuristic should consider:

- Immediate hand reduction.
- Whether the move unlocks future cards in the solver's own hand.
- Whether the move likely unlocks cards for opponents.
- The value of retaining control over unopened suits and future suit chains.
- Current hand counts, where available, for opponent-holder estimates and diagnostics.
- Information implied by opponent passes.

Later versions may add:

- Hand-count-consistent hidden deal sampling.
- Monte Carlo simulation over possible hidden deals.
- Multi-turn search.
- Self-play for tuning heuristic weights.

## Current Implementation

The first working implementation lives in `seven_hearts.py`.

It currently includes:

- A `Card` model with parsing support for labels such as `7H`, `AS`, and `10D`.
- A `SuitRun` model that represents each suit as a continuous interval growing outward from 7.
- A `GameState` model for public table state, turn state, optional hand counts, and move history.
- Legal move generation by intersecting the public legal cards with the solver's hand.
- Enforcement of the opening rule: the first public legal card is only `7H`.
- Optional turn validation, including rejection of illegal passes when legal moves exist.
- A `PlayerKnowledge` model containing the solver's player index and private hand.
- Calculation of unseen cards as `full_deck - own_hand - cards_on_table`.
- Basic opponent inference from passes.
- Exact hand-count-consistent holder marginals when known counts are available and consistent.
- Constrained hidden-deal sampling for simulation-ready opponent hand assignments.
- Oracle greedy rollouts and Monte Carlo move evaluation over sampled hidden deals.
- Information-limited rollouts and Monte Carlo recommendation over shared
  hidden-deal samples, where each simulated player chooses from only their own
  private hand plus public table/history/count evidence.
- Exact hidden-deal expected value against a declared deterministic
  information-limited greedy policy for reduced proof-sized belief states.
- Exact full-information rational-play solving for small/late-game complete-hand states.
- Exact full-information fixed-policy solving for comparing against declared opponent policies.
- Exhaustive hidden-deal enumeration and exact expected value under full-information rational continuation when the belief set is small enough.
- Complete random-deal simulation and aggregate self-play metrics.
- First-class strategy weight parameters for heuristic and oracle scoring.
- Shared-deal strategy tuning probes that compare candidate weight sets against baseline opponents.
- 52-bit card masks for faster legal-card intersections, compact full-information
  cache keys, and reusable hidden-deal sampling internals.
- Heuristic move scoring and recommendation through `recommend_move`.
- Immutable structured score components alongside human-readable score reasons.
- Structured proof-layer result objects that expose value vectors, move values, best moves, chosen moves, cache statistics, and exhaustive/non-exhaustive status.
- Human-readable proof certificates for full-information and imperfect-information exact recommendations.

Player hand counts are optional. The solver can run with:

```python
GameState(hand_counts=None)
```

or with partial information:

```python
GameState(hand_counts=(4, None, 7, None))
```

Known counts are used as a rough signal in opponent-holder estimates and for diagnostic urgency calculations. Unknown counts are treated neutrally.

## Current Heuristic

The current next-move scorer compares only decision-relevant score components. Earlier versions used a cosmetic base reward for playing a card, but that cancelled out across all legal moves and has been removed.

The scorer adjusts each legal move based on:

- Own cards newly unlocked by the move.
- Public cards newly opened that opponents may hold, weighted by likely holder and turn order.
- Future chain impact for the side or sides of the suit released by the move.
- Time-to-playable pressure for the solver's own cards.

Endgame urgency is not currently added to the move score. Since urgency depends on the state rather than on the specific legal card, adding it directly would cancel out in the move ranking. It remains available as a diagnostic helper, but the active scorer relies on card-specific factors.

This is intentionally a first-pass heuristic. It is designed to be inspectable and tunable before adding deeper search.

Future chain impact is evaluated for every move, not only for playing 7s.

For example:

- Playing `8S` releases the high-side chain `9S, 10S, JS, QS, KS`.
- Playing `10S` only releases `JS, QS, KS`.
- Therefore, playing `8S` gives up more future control than playing `10S`, all else equal.

When a 7 is played, both sides of the suit are evaluated:

- Low side: `6, 5, 4, 3, 2, A`
- High side: `8, 9, 10, J, Q, K`

The future chain impact score considers:

- How many future cards in the released chain are unseen and therefore likely held by opponents.
- How many future cards in the released chain are owned by the solver.
- Whether the solver owns the immediate gate card, such as `6S` or `8S`.
- Whether the solver owns a distant tail card, such as `AS` or `KS`, and that specific tail card's own time-to-playable runway.
- The turn-order-weighted opponent mass in the future released chain.

The future chain danger is intentionally kept simple: opening six future cards is worse than opening four future cards. The model does not currently apply position decay within the chain, because immediate unlock risk and turn order are already handled separately and the simpler count-based shape is easier to inspect.

This means the model now distinguishes cases such as:

- `7S` alone: expensive to open, because it releases a whole suit without much personal benefit.
- `7S + AS`: still risky, but opening the suit gets some credit because the ace needs the low-side chain to progress.
- `7S + 6S + 8S`: much better, because the solver owns both immediate gates and can play at effectively lower cost.
- `8S` versus `10S`: `8S` gives up more future control because it releases a longer chain.

## Current Opponent Model

The opponent model starts by assuming each unseen card could belong to any opponent, then narrows those possibility sets using public history.

It then walks through public move history:

- If an opponent passed, cards that were legal at that moment are removed from that opponent's possible cards.
- If an opponent played a card, that card is removed from the other opponents' possible cards.

The model exposes per-holder marginal estimates through `holder_probability(player, card)` rather than using the earlier misnamed "probability any opponent has this card" idea. Since every unseen card must be held by some opponent, the useful question is which opponent is likely to hold it.

When one or more opponent hand counts are known, the model now attempts exact count-consistent inference with a dynamic program:

- Every hidden card is assigned to exactly one possible opponent holder.
- Known hand counts are treated as quotas that must be exactly satisfied.
- Unknown opponent counts are left unconstrained and absorb any remaining cards.
- Public pass and play constraints remain hard constraints.
- The DP counts all hidden assignments consistent with those constraints and converts those counts into per-card, per-player holder marginals.

The architectural decision here is to improve the hidden-card model before adding multi-turn search. This keeps the active solver deterministic, inspectable, and aligned with the current next-move scope, while making every existing heuristic component consume better probabilities. It also avoids prematurely adding Monte Carlo variance or rollout policy questions before the one-move evaluator has a sounder information model.

If exact count-consistent inference cannot be used, such as when no useful counts are known or the known counts are inconsistent with the public constraints, the model falls back to the older weighted possible-holder estimate.

Unlock risk is turn-order-aware:

- Cards likely held by the next opponent are riskier.
- Cards likely held by later opponents are discounted.
- Multiple newly legal cards are capped per opponent turn so opening `6S` and `8S` is not treated as if both can be played by the same opponent immediately.

Known hand counts are now exact constraints when the DP can satisfy them. The previous rough count weighting remains only as a fallback when exact inference is unavailable.

For every unseen card that has at least one possible holder, holder probabilities are normalized so the per-opponent marginals sum to 1. The model also exposes `consistency_errors()` to flag impossible states, such as a known opponent hand count being larger than that opponent's remaining possible card set.

## Hidden Deal Sampling

The solver now includes constrained hidden-deal sampling through `sample_hidden_deal(...)` and `sample_hidden_deals(...)`.

A sampled hidden deal is a complete assignment of every unseen card to exactly one opponent. The sampler uses the same public information model as holder inference:

- The solver's own hand and already played table cards are excluded.
- Passed legal cards are excluded from the passing opponent's possible hand.
- Cards publicly played by one opponent are excluded from every other opponent.
- Known opponent hand counts are enforced exactly when provided.
- Unknown opponent hand counts remain unconstrained and absorb whatever hidden cards are not assigned to known-count opponents.

The algorithmic decision is to sample from the constrained assignment space directly instead of sampling independent cards from marginal probabilities. Independent marginal sampling would be simpler, but it could create impossible deals where a known-count opponent receives too many cards, too few cards, or a combination of cards that violates pass information. The current sampler walks hidden cards in sorted order and, when known counts exist, uses a dynamic program to count valid completions from each partial assignment. Random choices are weighted by the number of valid completions, so sampled deals remain count-consistent all the way through.

This is intentionally not yet a Monte Carlo player. The sampler creates plausible hidden worlds; a later rollout/search layer will decide how to use those worlds to evaluate candidate moves over multiple turns.

## Oracle Rollouts and Monte Carlo Evaluation

The solver now includes an oracle rollout layer through `rollout_oracle(...)`, `evaluate_move_monte_carlo(...)`, and `recommend_move_monte_carlo(...)`.

The Monte Carlo evaluator works as follows:

- Generate constrained hidden deals from the current public state.
- Apply a candidate legal move for the solver.
- Continue the game forward using complete sampled hands.
- On each simulated turn, the acting player chooses a legal move with an oracle greedy policy.
- Score the candidate by simulated first-out win rate, with average finish margin as a secondary signal and timeouts as a penalty.

`recommend_move_monte_carlo(...)` now compares all legal candidate moves over
the same sampled hidden deals. This reduces sampling variance between moves and
avoids rebuilding count-consistent sampler state for every candidate. Direct
single-move calls to `evaluate_move_monte_carlo(...)` still work; internally
they use the same reusable sampler path.

The card layer now exposes compact 52-bit masks for card sets. Public table
cards, public legal cards, full-information hand masks, exact-solver cache keys,
and hidden-deal sampler holder checks use these masks while the public API still
returns normal `Card` objects for readability.

The important architectural distinction is that the real solver still does not know opponents' hands. Full hand knowledge is used only inside sampled worlds during rollout. The top-level move estimate is still an expectation over plausible hidden deals, not a full-information assumption about the real game.

The rollout policy is called "oracle" because each simulated player can see the complete sampled deal. It is still greedy rather than game-theoretically exhaustive: each simulated turn uses a full-information version of the current structural heuristic, including exact opponent unlock risk, exact future chain impact, and race pressure from actual hand counts.

The oracle move chooser now includes a one-ply response adjustment. For each candidate move, it estimates the strongest immediate move available to the next player and discounts the candidate by part of that response value. This was added because pure greedy oracle play can be too myopic: a move may look locally productive while handing the next player an even stronger response. The one-ply penalty is deliberately shallow and inspectable, avoiding recursive search while improving tactical caution.

This implemented oracle is best understood as the legacy greedy
full-information diagnostic. The target evaluation oracle is stronger and
cleaner: it should be the perfect-information counterpart of the belief-state
sampled EV agent. That oracle should compare candidate moves from the true full
deal using the same candidate-EV shape as `Ours`, rather than only applying a
one-ply greedy structural score. It remains an unfair upper-bound reference
because it sees hidden hands, but it measures headroom more directly.

Monte Carlo scores now include:

- completed sample count
- first-out win rate
- win-rate standard error
- average finish margin
- average rollout length in turns
- timeout rate

The standard error is included so close results can be treated with appropriate skepticism. For example, two moves separated by less than one or two standard errors should be considered statistically close rather than decisively ranked.

This layer creates a useful comparison:

- `recommend_move(...)` gives the current one-move explainable heuristic recommendation.
- `recommend_move_monte_carlo(...)` gives a rollout-backed recommendation over plausible hidden worlds.
- `recommend_move_information_limited_monte_carlo(...)` gives a rollout-backed
  recommendation where continuation players use an information-limited
  heuristic policy instead of seeing the full sampled deal.
- `recommend_move_exact_information_limited_policy(...)` gives exact
  reduced-state EV against the deterministic information-limited greedy policy
  when the hidden-deal set is exhaustively enumerable.

When those recommendations disagree, the position is strategically interesting. The disagreement can show that a move with good immediate structure has poor long-run sampled outcomes, or that a locally scary unlock is acceptable because the sampled continuation favors the solver.

## Exact Proof Layer

The project now includes an initial proof-grade solver layer for positions where exhaustive evaluation is tractable.

The full-information exact solver uses `FullInformationState` and `solve_full_information(...)`.

It assumes:

- all four hands are known
- the table state is known
- the current player is known
- each player rationally maximizes their own first-out value
- ties are resolved by a deterministic documented tie-breaker

The solver recursively evaluates every legal continuation with memoization. It does not use heuristic pruning or rollout cutoffs. The returned `ExactSolverResult` includes:

- the root value vector
- every legal move's child value vector
- all moves tied under the acting player's primary objective
- the deterministic chosen move
- states evaluated
- cache hits
- terminal states reached
- deadlock states reached

The exact layer also includes `solve_full_information_against_policy(...)`, which evaluates the root decision against a declared fixed future policy. This is useful for proving best response to simple bots or regression policies before tackling full rational play.

Exact solver results can be rendered with `format_exact_solver_certificate(...)`, which prints the assumptions, legal moves, value vectors, selected move, tie-break rule, and search statistics.

The hidden-deal oracle layer uses `enumerate_hidden_deals(...)`, `evaluate_move_exact_imperfect_information(...)`, and `recommend_move_exact_imperfect_information(...)`.

It computes exact expected value only when all hidden deals consistent with the public evidence are enumerated, but the continuation is full-information rational play after each hidden deal is materialized. This is useful as a proof oracle for judging practical engines; it is not the final true imperfect-information optimal solver. Candidate moves now share one hidden-deal enumeration and one exact continuation cache, so exact recommendation avoids recomputing the same belief set and common continuations for every legal card. If `max_deals` truncates enumeration, the result is explicitly marked non-exhaustive and should be treated as diagnostic rather than proof. Because truncation is a deterministic prefix of the enumeration, it does not report a statistical standard error.

Exact hidden-deal oracle results can be rendered with `format_exact_imperfect_information_certificate(...)`. Certificates state the full-information continuation model, expected value vectors, outcome counts, and shared exact-search statistics.

`proof_demo.py` prints small proof certificates for:

- rational full-information play
- fixed-policy full-information play
- exact hidden-deal expected value with full-information continuation

`proof_benchmark.py` runs fast exact-search benchmarks by default, and
`proof_benchmark.py --include-hard` adds thousand-state reduced positions for
measuring the current full-information representation's scaling behavior. The
current hard tier is not intended to be the ceiling: after the bitmask/cache-key
pass, 100k-state reduced benchmarks are a reasonable next target, with larger
500k+ cases worth probing under explicit runtime limits.

`proof_eval.py` creates visual engine-quality reports in `proof_reports/`. It
uses the exact hidden-deal full-information-continuation oracle as a benchmark
for proof-sized positions, then plots exact move values, heuristic move scores,
Monte Carlo win rates, and exact EV regret for the practical engines. The CSV
outputs include both scenario-level choices and per-move rows with heuristic
components.

This layer is currently intended for reduced, late-game, or hand-authored proof
scenarios. It should be scaled beyond the present 4k-state hard benchmark toward
100k+ state measurements, but full-game initial positions still require Monte
Carlo or additional exact-search optimization.

## Full-Game Evaluation Plan

The practical full-game evaluation target is duplicate-deal, seat-rotated
average cards left. This is intentionally separate from exact proof validation:
the proof layer certifies small positions and declared policies, while the
full-game evaluator measures practical strength in the high-variance real game.

The primary metric is:

```text
average cards left
```

Lower is better. The equivalent payoff is:

```text
payoff = -cards_left
```

Reports may include payoff for formal comparison, but average cards left should
remain the headline because it is easier to read.

The evaluator should continue play after the first player empties their hand,
at least until all players finish or only one player remains with cards. This
allows true rank and finishing-distribution metrics. Existing first-out EV
search remains valid as the solver's internal objective; full-game evaluation
adds a richer outcome lens.

The benchmark protocol is:

1. Generate a random full deal.
2. Reuse the exact same deal across multiple games.
3. Rotate agent seat assignments.
4. Record cards left, payoff, rank, win/loss, and timeout status.

The first scale target is 1,000 random deals with four cyclic seat rotations,
for 4,000 games per four-agent table. Larger runs can scale up after runtime and
confidence intervals are measured.

Initial agents:

- `Random`: chooses uniformly among legal moves.
- `GreedyFurthestFromSeven`: plays the legal card with maximum
  `abs(rank - 7)`, using deterministic card order for ties.
- `Heuristic`: chooses the highest one-ply `score_move(...)` result.
- `Ours`: the belief-state sampled EV agent using shared hidden deals and a
  documented information-limited continuation policy.
- `Oracle`: full-information upper-bound diagnostic only. For final evaluation,
  this should be the perfect-information counterpart of `Ours`, using the true
  deal directly instead of sampling hidden deals. The current greedy
  full-information oracle remains useful as a legacy diagnostic but should not
  be the main oracle-gap target.

The main comparative statistic is paired card advantage:

```text
paired_card_advantage = baseline_cards_left - ours_cards_left
```

Positive values mean our agent finished with fewer cards than the baseline on
the same deal and seat conditions. Confidence intervals should be computed over
these paired differences.

The initial practical benchmark suite should include:

- `Ours vs Random vs GreedyFurthestFromSeven vs Heuristic`
- `Ours vs Random x3`
- `Ours vs GreedyFurthestFromSeven x3`
- `Ours vs Heuristic x3`
- `Ours vs GreedyFurthestFromSeven vs Heuristic vs strongest non-oracle rollout baseline`
- paired card advantage against every non-oracle baseline
- oracle-gap analysis reported separately

`Ours` should not be described as a true optimal imperfect-information solver.
The current defensible claim is narrower: it is a belief-state sampled first-out
EV agent under a declared information-limited policy. True optimal
imperfect-information play would require explicit information sets, belief
updates inside the recurrence or equilibrium method, and a documented
multiplayer hidden-information solution concept.

Current implementation status:

- Implemented: belief sampling, information-limited rollout, information-limited
  Monte Carlo recommendation, exact reduced-state EV against the deterministic
  information-limited greedy policy, and the legacy greedy full-information
  oracle.
- Not yet implemented: full-game cards-left evaluator, seat-rotated duplicate
  deals, paired card advantage confidence intervals, `GreedyFurthestFromSeven`
  as a full-game baseline, and the perfect-information counterpart oracle.
- Legacy/redundant: average payoff is only a sign-flipped version of average
  cards left, hand-strength buckets are deferred, and the current greedy oracle
  should not be the final oracle-gap reference.

## Benchmark Scenarios

`demo.py` now contains configurable Monte Carlo knobs:

```python
SAMPLES_PER_MOVE = 80
MAX_TURNS = 200
SEED = 7
```

It also runs named benchmark scenarios:

- **Demo Baseline.** The heuristic and oracle Monte Carlo both prefer `7S`, suggesting that the local self-unlock from `8S` survives rollout testing.
- **Bare Seven Versus Heart Progress.** The heuristic prefers heart progress, while oracle Monte Carlo prefers `7S` in the current seed. This is a useful disagreement: the local heuristic sees the bare 7 as a broad unlock risk, while rollouts sometimes value opening the suit path anyway.
- **Controlled Seven With Gates.** The heuristic strongly prefers `7S` because the solver owns both gates, but oracle Monte Carlo prefers `9H` in the current seed. This indicates the rollout layer may be detecting that immediate gate ownership is not always enough if the sampled continuation creates better tempo elsewhere.
- **Tail Runway Pressure.** Both heuristic and oracle Monte Carlo prefer `6H` in the current seed, with `7S` no longer winning after the one-ply response adjustment and the lower sample-count run.

These are not final proofs of strategic truth. They are regression-style positions that make solver disagreements visible and help guide future tuning.

## Complete Game Simulation

The solver now supports complete random-deal oracle self-play through:

- `deal_random_hands(...)`
- `initial_state_for_hands(...)`
- `simulate_complete_game(...)`
- `estimate_complete_game_metrics(...)`

A complete simulation deals all 52 cards evenly, finds the player holding `7H`, starts from an empty table with that player to act, and rolls forward with the same oracle one-ply policy used by Monte Carlo move evaluation. The simulation stops when the first player empties their hand, which matches the project's strategic objective of maximizing first-out probability.

Random deals are generated from a sorted deck before shuffling, so seeded simulation and tuning runs are reproducible across Python processes.

Aggregate complete-game metrics include:

- games simulated
- per-seat win rates
- per-seat win-rate standard errors
- per-seat average finish margins
- average turns until first player out
- timeout rate

The current demo run uses:

```python
COMPLETE_GAME_SAMPLES = 400
SELF_PLAY_TUNING_GAMES = 400
COMPLETE_GAME_MAX_TURNS = 300
```

With `SEED = 7`, the observed complete random-deal oracle self-play estimate was:

```text
games 400, turns 56.3, timeouts 0.0%
P0: win 27.3% +/- 2.2%, margin -1.4
P1: win 24.2% +/- 2.1%, margin -1.6
P2: win 24.8% +/- 2.2%, margin -1.4
P3: win 23.8% +/- 2.1%, margin -1.4
```

This is a sanity check rather than a strategic conclusion. The useful takeaway is that the oracle policy does not show an obvious seat bias in this run and complete games finish without timeouts.

The current demo also runs a small shared-deal tuning probe:

```text
candidate controls P0 against baseline opponents over 400 deals
tuned baseline: win 27.3% +/- 2.2%, margin -1.4, turns 56.3, timeouts 0.0%, score 25.9
gate tempo: win 27.3% +/- 2.2%, margin -1.4, turns 56.3, timeouts 0.0%, score 25.8
cautious unlocks: win 27.0% +/- 2.2%, margin -1.4, turns 56.2, timeouts 0.0%, score 25.6
legacy baseline: win 25.5% +/- 2.2%, margin -1.4, turns 56.4, timeouts 0.0%, score 24.1
```

This is a tuning probe, not a final tuned result. A later 10,000-deal tuning run found a small additional edge for stronger tail-runway credit:

```text
tail stronger: win 25.38% +/- 0.44%, margin -1.395, turns 56.0, timeouts 0.00%, score 23.985
previous tuned: win 25.20% +/- 0.43%, margin -1.403, turns 56.1, timeouts 0.00%, score 23.797
tail modest: win 24.95% +/- 0.43%, margin -1.413, turns 56.1, timeouts 0.00%, score 23.537
gate tempo: win 24.74% +/- 0.43%, margin -1.424, turns 56.0, timeouts 0.00%, score 23.316
cautious unlocks: win 24.56% +/- 0.43%, margin -1.425, turns 56.1, timeouts 0.00%, score 23.135
legacy baseline: win 24.55% +/- 0.43%, margin -1.426, turns 56.1, timeouts 0.00%, score 23.124
```

The current default now uses `tail_base_credit=2.6` and `tail_distance_penalty=0.16`.

## Parameter Tuning Status

The scoring weights are now represented explicitly by `StrategyWeights`.

This keeps the default solver behavior inspectable while making candidate strategies easy to compare. The same weight object is consumed by:

- the public heuristic scorer
- oracle rollout move choice
- Monte Carlo move evaluation
- complete-game simulation
- the shared-deal tuning harness

The project now includes `estimate_strategy_self_play(...)`, which runs multiple candidate `StrategyCandidate` objects over the same random deals. Each candidate controls one configured seat, currently player 0 by default, against baseline-weight oracle opponents. This avoids comparing policies only in all-seats mirror play, where the result is mostly seat and deal variance rather than evidence that a candidate is better.

Simulations currently serve four purposes:

- evaluating move recommendations over sampled hidden deals
- exposing disagreements between local heuristic scoring and rollout outcomes
- providing complete-game sanity checks such as win rate balance, average game length, and timeout rate
- comparing candidate weight sets across shared deals

They do not yet automatically optimize constants such as self-unlock value, opponent unlock risk, future chain impact, tail runway credit, or the one-ply response penalty. The current tuning harness is a measurement layer, not an optimizer.

The next tuning step should be a search layer over candidate weight sets, using the shared-deal harness as the objective evaluator. Until that exists, tuning results should be treated as diagnostics and evidence, not as proof that the weights are optimal.

## Time-To-Playable Logic

The solver now has a `time_to_playable` estimate.

This counts how many chain steps are needed before a card can become playable from the current table state.

Examples:

- If spades is open only at `7S`, then `8S` is 1 step away.
- If spades is open only at `7S`, then `AS` is 6 steps away.
- If a suit is unopened, distance is measured from 7.

This helps avoid treating a distant tail card as automatically valuable. A tail like `AS` or `KS` gets credit only in relation to the runway required to reach it.

Tail-card runway is measured using the tail card itself, not the average distance of all owned cards on that side. For example, owning both `8S` and `KS` should not make the `KS` look close just because `8S` is close.

## Validation

`GameState.public_legal_cards()` now enforces the real opening rule:

```text
Empty table -> only 7H is legal
After 7H -> unopened suits may be opened with their 7s
```

`GameState.after_play(...)` remains usable as a low-level replay helper, but it can validate a turn when a hand is supplied:

```python
state.after_play(player, card, hand=my_hand)
```

Validation rejects:

- playing a card that is not in hand
- playing a card that is not legal
- passing while legal moves are available

## Remaining Modeling Work

The following items are intentionally documented as future work:

- **Practical imperfect-information solver.** The first sampled
  information-limited rollout path now exists: future players act from only
  their own private hand and public evidence under greedy or softmax heuristic
  policies. The deterministic greedy information-limited policy also has an
  exact reduced-state EV path. The current full-information hidden-deal layer
  remains an oracle that reveals each enumerated deal to the continuation
  solver. Near-term work is stronger validation snapshots and additional
  declared information-limited policies.
- **True reduced-deck imperfect-information optimal solver.** As a possible
  future research/validation layer, solve small declared reduced decks with
  explicit information sets, belief updates after public plays and passes, and a
  documented multiplayer solution concept such as exact public-belief search or
  a perfect-recall equilibrium method. This should be treated as proof-sized
  research, not the expected full-deck production engine.
- **Stronger multi-turn search.** The current rollout policy is greedy with oracle hand knowledge plus a one-ply response adjustment. A deeper search layer could compare candidate continuations recursively rather than choosing only the locally best oracle move at each simulated turn.
- **Perfect-information counterpart oracle.** The current `rollout_oracle(...)`
  path is a greedy full-information diagnostic. The evaluation oracle should be
  upgraded to mirror the candidate-EV comparison shape of `Ours` while using the
  true full deal directly, with exact full-information solving used only for
  reduced tractable states and rollout evaluation used for full games.
- **Automated weight search.** Constants are still heuristic. The shared-deal tuning harness now provides an objective surface, but it does not yet generate or optimize candidate weight sets automatically.
- **Endgame policy testing.** Endgame urgency is currently diagnostic only. If self-play shows that late-game move preferences should change, urgency should modulate card-specific components rather than being added as a flat score.
- **Sampler and simulation performance optimization.** Exact count-consistent sampling is principled but can be slow at larger sample counts, especially when many benchmark scenarios are evaluated. The Monte Carlo recommender now reuses sampler state and shared sampled deals across candidate moves, and core card-set operations have a bitmask path. Later work can reuse samples across repeated strategy comparisons, move rollout hands to masks, and cache deterministic information-limited policy evaluations.

## Demo and Tests

`demo.py` contains a small example scenario that prints:

- Legal moves.
- Score for each legal move.
- Reasons behind each score.
- The recommended move.
- Oracle Monte Carlo scores and recommendation.
- Named benchmark scenarios for strategic comparison.
- Complete random-deal oracle self-play metrics.
- Shared-deal candidate strategy tuning metrics.

`test_seven_hearts.py` contains focused tests for rules and inference behavior.

`run_tests.py` is a tiny no-dependency test runner used because `pytest` was slow or unresponsive in the current environment.

Current verification:

```text
py run_tests.py
82 tests passed
```

Current demo command:

```text
py demo.py
```

Current proof benchmark commands:

```text
py proof_benchmark.py
py proof_benchmark.py --include-hard
py proof_eval.py
```
