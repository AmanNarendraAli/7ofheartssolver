# Experiment Log

This is the project lab notebook for tuning and evaluation runs. Keep entries
short, concrete, and reproducible. The goal is to preserve enough context for
later analysis, paper writing, and sanity checks without turning each run into a
mini report.

## Protocol

For serious runs, record:

- question being tested
- exact command
- code version or commit hash
- role of the run: smoke, train, validation, or locked test
- candidate-generation seed and evaluation seed
- candidate count, deal count, rotations, workers, deck size, and max turns
- primary metric and baseline
- report directory
- result summary with confidence intervals
- decision made from the run

Use separate train, validation, and final-test seeds/deals. Do not use final-test
results to keep tuning.

## Planned Runs

### Heuristic Random Search 001

Question:

```text
Can random-searched heuristic weights beat the current default heuristic on
paired duplicate-deal card advantage?
```

Role:

```text
train / coarse search
```

Recommended command:

```powershell
py tune_eval.py --mode heuristic --candidates 500 --deals 1000 --workers 8 --progress-every 10
```

Primary metric:

```text
paired card advantage vs stock Heuristic
```

Notes:

```text
c0000 is the current default StrategyWeights anchor.
Use candidate_summary.csv for ranking and candidate_parameters.csv to recover weights.
```

Status:

```text
not run
```

## Run Entries

Copy this template for each serious run.

### YYYY-MM-DD Short Run Name

Question:

```text

```

Role:

```text
smoke | train | validation | locked test
```

Command:

```powershell

```

Code version:

```text
commit: 
dirty: 
```

Setup:

```text
mode:
candidates:
deals:
rotations_per_deal:
games:
cards_per_suit:
max_turns:
workers:
candidate_seed:
eval_seed:
primary_metric:
target_baseline:
```

Report directory:

```text

```

Result summary:

```text

```

Decision / next step:

```text

```
