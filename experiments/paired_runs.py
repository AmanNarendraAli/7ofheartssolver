"""Pair two full_game_eval runs on the same deal/seat games.

usage: python paired_runs.py RUN_A AGENT_A RUN_B AGENT_B
Reports AGENT_B minus AGENT_A on games both runs played (same seed, same seat).
"""
import csv, math, sys

def load(run_dir, agent):
    out = {}
    for r in csv.DictReader(open(f"{run_dir}/games.csv")):
        seats = [r[f"seat_{s}_agent"] for s in range(4)]
        seat = seats.index(agent)
        out[(r["deal_index"], r["rotation_index"])] = (
            r["seed"], seat, int(r["winner_seat"]) == seat, int(r[f"seat_{seat}_cards_at_first_out"]), int(r[f"seat_{seat}_rank"])
        )
    return out

def mean_ci(xs):
    n = len(xs); m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)
    return f"{m:+.3f} +/- {1.96 * se:.3f}"

a = load(sys.argv[1], sys.argv[2]); b = load(sys.argv[3], sys.argv[4])
keys = sorted(set(a) & set(b), key=lambda k: (int(k[0]), int(k[1])))
# Hands for deal i depend only on --seed (all deals are dealt before per-game seeds are drawn),
# so runs with different --deals still share deals 1..min(N). Per-game agent seeds differ then.
assert all(a[k][1] == b[k][1] for k in keys), "seats differ"
print(f"{sys.argv[4]} minus {sys.argv[2]} over {len(keys)} paired games")
print("  first_out_win      ", mean_ci([b[k][2] - a[k][2] for k in keys]))
print("  cards_at_first_out ", mean_ci([a[k][3] - b[k][3] for k in keys]), "(positive = fewer cards)")
print("  rank               ", mean_ci([a[k][4] - b[k][4] for k in keys]), "(positive = better rank)")
print(f"  win rates: {sum(a[k][2] for k in keys)/len(keys):.3f} -> {sum(b[k][2] for k in keys)/len(keys):.3f}")
