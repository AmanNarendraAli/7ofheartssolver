"""How often does a vote over B layouts pick a different move than the vote over all layouts?

Reference = vote over every stored layout (exact when the layout set was exhaustive).
For B in 64, 256, 512 the stored layouts are shuffled and split into disjoint blocks;
each block votes with the agent's tie-break (wins, then heuristic score). Regret is in
reference win-rate points: ref_rate(reference pick) - ref_rate(block pick).
"""
import csv, random, sys
from statistics import mean


def parse_masks(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "(":
            j = s.index(")", i)
            out.append(int(s[i + 1:j], 16)); i = j + 1
        else:
            out.append(int(s[i], 16)); i += 1
    return out


def vote(masks, k, heur):
    wins = [sum((m >> i) & 1 for m in masks) for i in range(k)]
    return max(range(k), key=lambda i: (wins[i], heur[i])), wins


def main(path):
    rows = list(csv.DictReader(open(path)))
    rng = random.Random(1)
    stats = {B: {"agree": [], "regret": [], "big": []} for B in (64, 256, 512)}
    heur_agree, heur_regret, contested = [], [], []
    for r in rows:
        k = int(r["legal"]); heur = [float(x) for x in r["heuristic_scores"].split()]
        masks = parse_masks(r["masks"]); n = len(masks)
        assert n == int(r["layouts"])
        ref, wins = vote(masks, k, heur)
        rate = [w / n for w in wins]
        contested.append(len(set(wins)) > 1)
        h = int(r["heuristic_choice"])
        heur_agree.append(h == ref); heur_regret.append(rate[ref] - rate[h])
        shuffled = masks[:]; rng.shuffle(shuffled)
        for B, s in stats.items():
            if n <= B:  # the agent would see every layout too
                blocks = [masks] if int(r["exhaustive"]) else []
            else:
                blocks = [shuffled[i:i + B] for i in range(0, n - B + 1, B)]
            if not blocks:
                s["agree"].append(None); s["regret"].append(None); s["big"].append(None); continue
            picks = [vote(b, k, heur)[0] for b in blocks]
            s["agree"].append(mean(p == ref for p in picks))
            s["regret"].append(mean(rate[ref] - rate[p] for p in picks))
            s["big"].append(mean(rate[ref] - rate[p] > 0.02 for p in picks))
    print(f"{len(rows)} decisions, {sum(int(r['exhaustive']) for r in rows)} exhaustive, "
          f"median layouts {sorted(int(r['layouts']) for r in rows)[len(rows)//2]}, "
          f"{sum(contested)} where the reference win counts differ between moves")
    for label, keep in (("all", lambda i: True), ("contested", lambda i: contested[i])):
        print(f"\n[{label}]")
        for B, s in stats.items():
            idx = [i for i in range(len(rows)) if keep(i) and s["agree"][i] is not None]
            if not idx:
                continue
            print(f"  vote over {B:>3} layouts: same pick {100*mean(s['agree'][i] for i in idx):5.1f}%  "
                  f"mean regret {100*mean(s['regret'][i] for i in idx):.2f} pts  "
                  f"regret >2 pts {100*mean(s['big'][i] for i in idx):5.1f}%  (n={len(idx)})")
        idx = [i for i in range(len(rows)) if keep(i)]
        print(f"  heuristic           : same pick {100*mean(heur_agree[i] for i in idx):5.1f}%  "
              f"mean regret {100*mean(heur_regret[i] for i in idx):.2f} pts")


if __name__ == "__main__":
    main(sys.argv[1])
