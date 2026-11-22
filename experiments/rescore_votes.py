"""Rescore pilot decisions with many more layouts: does the 64-layout vote pick change?

Takes a uniform sample of the dominance pilot's decision points (same games, seeds
10000+), draws up to L layouts for each, solves every layout exactly with
FirstOutSolver, and stores which legal moves win in each layout. analyse_rescore.py
then compares votes on disjoint 64- and 256-layout blocks against the full-L vote.
"""
import csv, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seven_hearts import (
    DEFAULT_WEIGHTS, FirstOutSolver, PlayerKnowledge, build_opponent_model,
    choose_hidden_deals_for_budget, complete_hands, immediate_winning_move, score_move,
)
from dominance_pilot import collect_decisions


def rescore(args):
    (game_seed, turn, state, hands, p, heur_card), L = args
    t = time.perf_counter()
    rng = random.Random(game_seed * 7919 + turn)
    knowledge = PlayerKnowledge(p, hands[p])
    legal = state.legal_moves(hands[p])
    selection = choose_hidden_deals_for_budget(state, knowledge, L, rng)
    masks = []
    for deal in selection.deals:
        wins = FirstOutSolver(complete_hands(knowledge, deal)).move_wins(state.table, p, legal)
        masks.append(sum(1 << i for i, w in enumerate(wins) if w))
    model = build_opponent_model(state, knowledge)
    heuristic = [score_move(state, knowledge, c, model=model, weights=DEFAULT_WEIGHTS).score for c in legal]
    return {
        "game": game_seed, "turn": turn, "player": p, "hand_size": len(hands[p]),
        "legal": len(legal), "layouts": len(masks), "exhaustive": int(selection.exhaustive),
        "heuristic_choice": legal.index(heur_card),
        "heuristic_scores": " ".join(f"{h:.6f}" for h in heuristic),
        "masks": "".join(f"{m:x}" if m < 16 else f"({m:x})" for m in masks),
        "seconds": round(time.perf_counter() - t, 2),
    }


if __name__ == "__main__":
    games, n, L, workers = (int(a) for a in sys.argv[1:5])
    out = sys.argv[5]
    t0 = time.perf_counter()
    points = []
    for g in range(games):
        points.extend(collect_decisions(10_000 + g))
    # The agent never votes when it can go out immediately.
    points = [pt for pt in points
              if immediate_winning_move(PlayerKnowledge(pt[4], pt[3][pt[4]]), pt[2].legal_moves(pt[3][pt[4]])) is None]
    chosen = sorted(random.Random(0).sample(range(len(points)), min(n, len(points))))
    print(f"{len(points)} voting decisions; rescoring {len(chosen)} with up to {L} layouts", flush=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(rescore, [(points[j], L) for j in chosen], chunksize=1)):
            rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(chosen)}] {time.perf_counter()-t0:.0f}s", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"done in {time.perf_counter()-t0:.0f}s -> {out}")
