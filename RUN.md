# 7 of Hearts CHTC Tuning Runbook

## Build And Transfer

Run these from WSL2 Ubuntu on Windows, in the pulled repo directory that contains `tune.def`:

```sh
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer

apptainer --version
apptainer build tune.sif tune.def

ssh <user>@<cluster-login-host> 'mkdir -p ~/containers'
scp tune.sif <user>@<cluster-login-host>:~/containers/tune.sif
```

Building from `docker://python:3.12-slim` requires internet access in WSL. If the unprivileged build is blocked locally, use `sudo apptainer build tune.sif tune.def`, then `sudo chown "$USER:$USER" tune.sif`.

The image intentionally installs only `numpy`, because `tune_eval.py` needs NumPy for `.npz` output and the rest of the repo code is pulled on the cluster at runtime. It does not copy repository code into the image.

## Cluster Run

On the CHTC login host:

```sh
cd ~/7ofhearts
git pull
apptainer exec ~/containers/tune.sif python -c "import numpy; print(numpy.__version__)"
```

Edit `submit.sh`:

- `SIF`: absolute path to `tune.sif`
- `REPO`: absolute path to the pulled repo
- `OUTDIR`: absolute output path, for example `$REPO/tuning_reports`
- `#SBATCH --cpus-per-task`: core count to request, defaulted to `128`

Submit and monitor:

```sh
mkdir -p logs
sbatch submit.sh
squeue -u "$USER"
sacct -j <jobid>
```

Outputs appear under `OUTDIR/run_*/`:

- `candidate_summary.npz`
- `candidate_parameters.npz`
- `run_metadata.json`

Load the NumPy reports like this:

```python
import numpy as np

summary = np.load("candidate_summary.npz", allow_pickle=True)
print(summary["candidate_id"], summary["score"])

params = np.load("candidate_parameters.npz", allow_pickle=True)
print(params["candidate_id"], params["parameter"], params["value"])
```

After the first full run, use `run_metadata.json["elapsed_seconds"]` to tune `#SBATCH --time`. A previous local run with 8 workers should be roughly 16x slower than a 128-worker run if CPU scaling is ideal, but keep the first cluster time request generous.
