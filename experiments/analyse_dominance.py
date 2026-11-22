import csv, sys, math
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    for k in ("hand_size", "legal", "layouts", "exhaustive", "winnable_layouts", "has_dominant", "never_winnable", "heuristic_is_best"):
        r[k] = int(r[k])
    for k in ("best_win_rate", "heuristic_win_rate", "heuristic_regret"):
        r[k] = float(r[k])
    wc = list(map(int, r["win_counts"].split()))
    r["matters"] = int(len(set(wc)) > 1)

def pct(xs):
    return f"{100*sum(xs)/len(xs):5.1f}%" if xs else "   -  "

def report(label, rs):
    n = len(rs)
    if not n:
        return
    winnable = [r for r in rs if not r["never_winnable"]]
    matters = [r for r in rs if r["matters"]]
    print(f"{label:<22} n={n:4d}  never-winnable {pct([r['never_winnable'] for r in rs])}  "
          f"choice-matters {pct([r['matters'] for r in rs])}  "
          f"dominant|winnable {pct([r['has_dominant'] for r in winnable])}  "
          f"dominant|matters {pct([r['has_dominant'] for r in matters])}  "
          f"heur-best {pct([r['heuristic_is_best'] for r in rs])}  "
          f"heur-regret {sum(r['heuristic_regret'] for r in rs)/n:.3f}  "
          f"exhaustive {pct([r['exhaustive'] for r in rs])}")

report("all", rows)
for lo, hi in ((11, 13), (8, 10), (5, 7), (2, 4)):
    report(f"hand size {lo}-{hi}", [r for r in rows if lo <= r["hand_size"] <= hi])
for k in (2, 3, 4):
    report(f"legal = {k}" if k < 4 else "legal >= 4", [r for r in rows if (r["legal"] == k if k < 4 else r["legal"] >= 4)])
m = [r for r in rows if r["matters"]]
nd = [r for r in m if not r["has_dominant"]]
if nd:
    gaps = sorted(r["best_win_rate"] - sorted(map(int, r["win_counts"].split()))[-2] / r["layouts"] for r in nd)
    print(f"\nno-dominant decisions: {len(nd)}; median gap best vs 2nd-best win rate {gaps[len(gaps)//2]:.3f}")
print(f"mean seconds/decision {sum(float(r['seconds']) for r in rows)/len(rows):.2f}")
