#!/bin/bash
#SBATCH --job-name=7oh-tune
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128        # <-- node core count; collaborator confirms 128
#SBATCH --mem-per-cpu=2000         # MB per core; pure-Python job is light, adjust if needed
#SBATCH --time=1-00:00:00          # D-HH:MM:SS; raise/lower per estimate & partition cap
#SBATCH --output=logs/tune_%j.out
#SBATCH --error=logs/tune_%j.err

# Usage:
#   mkdir -p logs
#   sbatch submit.sh
#   Logs land in logs/tune_<jobid>.out/.err; outputs land under OUTDIR/run_*/.
#   Check status with: squeue -u "$USER" ; after completion: sacct -j <jobid>

set -euo pipefail

# Edit these paths for the cluster account.
SIF="$HOME/containers/tune.sif"
REPO="$HOME/7ofhearts"
OUTDIR="$REPO/tuning_reports"

mkdir -p logs "$OUTDIR"

WORKERS="${SLURM_CPUS_PER_TASK:-128}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "SIF=$SIF"
echo "REPO=$REPO"
echo "OUTDIR=$OUTDIR"
echo "WORKERS=$WORKERS"
echo "started_at=$(date --iso-8601=seconds)"

# Python puts the script's directory on sys.path, so full_game_eval/seven_hearts
# imports resolve when tune_eval.py is run by absolute path.
# If REPO or OUTDIR are on filesystems Apptainer does not auto-mount, keep these
# binds. They are harmless for normal $HOME paths and useful for /scratch paths.
# SIF only needs to be readable by the host apptainer command.
BIND_ARGS=(--bind "$REPO")
case "$OUTDIR/" in
    "$REPO"/*) ;;
    *) BIND_ARGS+=(--bind "$OUTDIR") ;;
esac

apptainer exec "${BIND_ARGS[@]}" "$SIF" python "$REPO/tune_eval.py" \
    --mode heuristic --candidates 500 --deals 1000 \
    --workers "$WORKERS" --progress-every 10 \
    --output-dir "$OUTDIR"

echo "finished_at=$(date --iso-8601=seconds)"
echo "final_output_dir=$OUTDIR"
