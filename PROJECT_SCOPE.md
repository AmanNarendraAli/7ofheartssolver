# 7 of Hearts Solver Project

## Purpose

This project builds a practical 7 of Hearts solver: given one player's private
hand and the public game state, recommend a legal move that improves the
player's chance of getting out first.

The solver is meant to operate under realistic information. It knows its own
hand, the table, turn state, public play/pass history, and optional hand counts.
It does not know opponents' exact hands. Opponent cards are modeled through
public evidence, constrained hidden-deal sampling, and reduced proof cases.

Formal solver claims, exact-search semantics, invariants, acceptance criteria,
and full-game evaluation protocol details live in `SOLVER_PROOF.md`. This file
stays high level: what the project is trying to do, what exists, and what is
still planned.

## Game

7 of Hearts, also called Sevens, is played by 4 players with a standard
52-card deck. Each player receives 13 cards. The strategic objective for this
project is to choose moves that maximize first-out chances.

The player holding `7H` starts and must play it. Each suit then grows outward
from its 7. For example, after `7H`, the legal heart plays are `6H` and `8H`.
A suit cannot be played until its 7 has opened it. Aces are low:

```text
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
```

On a turn, a player must play a legal card if one exists. If no legal move
exists, the player passes.

## Strategy

The game is mostly about controlling future availability:

- Playing a card may unlock your own blocked cards.
- Playing a card may also unlock opponents' cards.
- Holding a 7 can block an entire suit.
- Holding a gate card, such as a 6 or 8 next to an opened 7, can control one
  side of a suit.
- Distant tail cards, especially aces and kings, need enough runway to become
  playable.

The solver should prefer causal, inspectable logic over special-case moods. A
move should be evaluated by what it opens, who is likely to benefit, who acts
soonest, how much chain control is released, and how much runway the solver's
own hand needs.

Endgame urgency should not be a flat bonus that applies equally to every move.
If late-game pressure matters, it should modulate card-specific effects and be
validated by evaluation.

## Current Shape

The implementation lives primarily in `seven_hearts.py`.

The project currently has:

- a rules engine for cards, suit runs, public table state, legal moves, passes,
  and turn validation
- a public-information model based on own hand, played cards, hand counts, and
  pass history
- heuristic move scoring through `recommend_move(...)` and `score_move(...)`
- constrained hidden-deal sampling and enumeration
- information-limited rollout and Monte Carlo recommendation over shared hidden
  deals
- mask-backed information-limited Monte Carlo rollouts, including mask hand
  updates, winner/count checks, and mask-native policy choice fast paths
- exact reduced-state EV against declared information-limited policies
- a perfect-information counterpart oracle for oracle-gap evaluation
- exact full-information and fixed-policy proof oracles for small or late-game
  complete-hand states
- proof certificates and benchmark/report scripts for reduced validation cases
- full-game duplicate-deal, seat-rotated evaluation through `full_game_eval.py`
- per-run report directories with outcome CSVs and run/agent parameter metadata
- per-agent standard errors, paired 95% confidence intervals, decision-volume
  counts, and oracle-gap columns in the existing full-game summary CSVs
- random-search tuning harness through `tune_eval.py` for heuristic weights and
  Monte Carlo run settings on shared duplicate-deal evaluations
- experiment notebook in `EXPERIMENT_LOG.md` for preserving run questions,
  commands, seeds, reports, results, and decisions
- optional MC-vs-heuristic decision tracing through
  `full_game_eval.py --trace-mc-heuristic`, written as
  `mc_heuristic_decisions.csv` in the run report directory
- a no-dependency test runner in `run_tests.py`

The main practical agent, `Monte Carlo`, is best described as a belief-state
sampled first-out EV agent under a declared information-limited continuation
policy. It is not claimed to be a true optimal imperfect-information solver.

## Evaluation

There are two evaluation lenses:

- **Proof-sized validation:** exact solvers and certificates on reduced,
  late-game, or hand-authored positions.
- **Practical full-game strength:** duplicate-deal, seat-rotated full games
  against baseline agents.

The headline full-game metric is average cards left, where lower is better.
Reports also track ranks, winner, timeouts, paired card advantage, and run
parameters. The current default comparison table uses:

- `Random`
- `GreedyFurthestFromSeven`
- `Heuristic`
- `Monte Carlo`

`--fast` keeps the agent name `Monte Carlo` but uses the heuristic policy for
cheap smoke tests.

`--trace-mc-heuristic` keeps the evaluation policy unchanged, but when
`Monte Carlo` is using information-limited Monte Carlo it records the
same-position heuristic choice beside the actual Monte Carlo choice. The resulting
`mc_heuristic_decisions.csv` is for diagnosing disagreement rate, MC confidence,
heuristic reasons, and eventual outcome on those decision points.

## Roadmap

Important remaining work:

- run larger practical benchmark suites using the existing confidence-interval
  and decision-volume reporting
- use the perfect-information counterpart oracle for oracle-gap analysis
- continue validating information-limited policies on reduced proof cases
- improve remaining Monte Carlo scoring performance without changing the
  declared clean evaluation policy
- keep true imperfect-information equilibrium methods as future reduced-deck
  research rather than the immediate production engine

## Common Commands

```text
py run_tests.py
py demo.py
py proof_demo.py
py proof_benchmark.py
py proof_benchmark.py --include-hard
py proof_eval.py
py full_game_eval.py
py full_game_eval.py --fast
py full_game_eval.py --deals 2 --cards-per-suit 5 --samples-per-move 4 --rollout-max-turns 40 --progress-every 2 --max-turns 200
py full_game_eval.py --deals 1 --cards-per-suit 5 --samples-per-move 1 --rollout-max-turns 20 --max-turns 200 --oracle-gap --progress-every 0
py full_game_eval.py --deals 25 --cards-per-suit 5 --samples-per-move 16 --rollout-max-turns 40 --max-turns 200 --trace-mc-heuristic
py tune_eval.py --mode heuristic --candidates 16 --deals 100 --workers 4

Before serious tuning/eval runs, add an entry to `EXPERIMENT_LOG.md`.

NEVER use pytest for anything.
```
