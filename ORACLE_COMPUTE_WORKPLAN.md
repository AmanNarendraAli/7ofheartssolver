# Oracle Compute Workplan

Goal: use a free Oracle Ampere A1 VM as a remote CPU box for serious `7ofhearts` evaluation runs, while keeping runs reproducible and avoiding wasted compute.

## Phase 1: Set Up The VM

1. Create an Oracle Cloud Free Tier account.
2. Create an Ampere A1 Flex compute instance:
   - Image: Ubuntu 22.04 or Ubuntu 24.04
   - Shape: Ampere A1 Flex
   - OCPUs: `4`
   - Memory: `24 GB`
   - Networking: SSH enabled
3. Set an OCI budget alert so billing surprises are visible.
4. SSH into the machine.
5. Install basics:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip tmux
```

## Phase 2: Move The Project

1. Put this repo on GitHub if it is not already there.
2. Clone it on the VM:

```bash
git clone YOUR_REPO_URL 7ofhearts
cd 7ofhearts
```

3. Run tests and a smoke evaluation:

```bash
python3 run_tests.py
python3 full_game_eval.py --deals 2 --workers 1 --samples-per-move 2 --rollout-max-turns 40 --output-dir full_game_reports/smoke
```

## Phase 3: Benchmark The VM

Measure throughput before doing expensive runs. The project is CPU-bound, and `--workers` controls independent worker processes. On a 4 OCPU VM, `4` workers is the expected starting point, but the best value should be measured.

```bash
python3 full_game_eval.py --deals 10 --workers 1 --samples-per-move 8 --rollout-max-turns 80 --output-dir full_game_reports/bench_w1
python3 full_game_eval.py --deals 10 --workers 2 --samples-per-move 8 --rollout-max-turns 80 --output-dir full_game_reports/bench_w2
python3 full_game_eval.py --deals 10 --workers 4 --samples-per-move 8 --rollout-max-turns 80 --output-dir full_game_reports/bench_w4
```

Decision rule: choose the worker count with the best completed-games-per-second. If `4` workers causes memory pressure, cache pressure, or worse throughput, use `2` or `3`. If oversubscription helps, try `6`, but only after the baseline benchmark.

## Phase 4: Calibration Runs

Run enough games to see whether the metrics stabilize.

```bash
python3 full_game_eval.py --deals 100 --workers BEST --samples-per-move 16 --rollout-max-turns 80 --output-dir full_game_reports/calib_s16_d100
```

Inspect:

- `agent_summary.csv`
- `paired_card_advantage.csv`
- standard error
- timeout rate
- average turns

Decision rules:

- If paired-card standard error is large, increase `--deals`.
- If "Ours" appears strategically shallow, increase `--samples-per-move`.
- If games time out or runtime explodes, lower `--rollout-max-turns` or inspect policy behavior.

## Phase 5: Serious Runs

Once calibrated, run multiple seeds instead of trusting one giant run.

```bash
tmux new -s eval
python3 full_game_eval.py --deals 250 --workers BEST --samples-per-move 32 --rollout-max-turns 100 --seed 101 --output-dir full_game_reports/serious_s32_seed101
```

Repeat with seeds such as `102`, `103`, and `104`.

Multiple independent seeds make it easier to tell whether the result is stable across random deals. This is usually more convincing than one monolithic run.

## Phase 6: Pull Reports Back

From the laptop:

```powershell
scp -i path\to\oracle_key -r ubuntu@YOUR_PUBLIC_IP:~/7ofhearts/full_game_reports .\full_game_reports\oracle
```

If the VM uses Oracle Linux instead of Ubuntu, the SSH username may be `opc` instead of `ubuntu`.

## Decision Logic

The main compute question is not "max everything." It is:

1. Find the best `--workers` value for the VM.
2. Spend compute on more `--deals` until standard error is useful.
3. Increase `--samples-per-move` when the goal is to test stronger decision quality.
4. Use multiple seeds for serious claims.

The first serious target is:

```bash
python3 full_game_eval.py --deals 250 --workers BEST --samples-per-move 32 --rollout-max-turns 100 --output-dir full_game_reports/serious_s32_seed101 --seed 101
```

Only run that after the benchmark and calibration runs show that the VM throughput and timeout rate are sane.

