"""Pilot: at real decision points, is one move best across every hidden-card layout?

Decision points come from games of 4 heuristic players, before first-out.
For each one: draw up to K layouts consistent with the deciding player's
information, solve each layout exactly (perfect information after the move),
and record which legal moves lead to a first-out win for the decider.
"""
import csv, random, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
sys.setrecursionlimit(10000)
from seven_hearts import (
    Card, PlayerKnowledge, SUIT_ORDER, apply_known_play, build_opponent_model,
    choose_hidden_deals_for_budget, complete_hands, deal_random_hands, first_empty_player,
    initial_state_for_hands, score_move, DEFAULT_WEIGHTS,
)

DECODE = [(None, None)] + [(l, h) for l in range(1, 8) for h in range(7, 14)]
def code_of(low, high):
    return 0 if low is None else 1 + (low - 1) * 7 + (high - 7)
SUIT_POS = {s: i for i, s in enumerate(SUIT_ORDER)}

from pi_solve import Solver as _ArraySolver, owner_from_hands

class Solver(_ArraySolver):
    """Exact perfect-information first-out winner; built from the remaining hands only."""
    def __init__(self, hands):
        super().__init__(owner_from_hands(hands))

    def legal_moves(self, codes, p):
        return [(s, nc) for s, nc in self.legal(list(codes), p)]

    def move_wins(self, codes, p, moves):
        out = []
        for s, nc in moves:
            child = list(codes); child[s] = nc
            if self.count(child, p) == 0:
                out.append(True)
            else:
                out.append(self.solve(child, (p + 1) % 4) == p)
        return out

def table_codes(state):
    return tuple(code_of(state.table[s].low, state.table[s].high) for s in SUIT_ORDER)

def heuristic_move(state, hands, player):
    knowledge = PlayerKnowledge(player, frozenset(hands[player]))
    model = build_opponent_model(state, knowledge)
    legal = state.legal_moves(hands[player])
    return max(legal, key=lambda c: score_move(state, knowledge, c, model=model, weights=DEFAULT_WEIGHTS).score)

def collect_decisions(game_seed):
    rng = random.Random(game_seed)
    hands = deal_random_hands(rng)
    state = initial_state_for_hands(hands)
    points = []
    turn = 0
    while first_empty_player(hands) is None:
        p = state.current_player
        legal = state.legal_moves(hands[p])
        card = heuristic_move(state, hands, p) if legal else None
        if len(legal) >= 2:
            points.append((game_seed, turn, state, {q: frozenset(h) for q, h in hands.items()}, p, card))
        state, hands = apply_known_play(state, hands, p, card)
        turn += 1
    return points

def analyse(args):
    (game_seed, turn, state, hands, p, heur_card), K = args
    rng = random.Random(game_seed * 1000 + turn)
    knowledge = PlayerKnowledge(p, hands[p])
    selection = choose_hidden_deals_for_budget(state, knowledge, K, rng)
    codes = table_codes(state)
    legal = state.legal_moves(hands[p])
    t = time.perf_counter()
    win_counts = [0] * len(legal)
    winnable = 0
    layouts = 0
    dominant = [True] * len(legal)
    states = 0
    for deal in selection.deals:
        layout = complete_hands(knowledge, deal)
        sv = Solver(layout)
        moves = sv.legal_moves(codes, p)
        assert [SUIT_ORDER[s] for s, _ in moves] == [c.suit for c in legal] and len(moves) == len(legal)
        wins = sv.move_wins(codes, p, moves)
        states += sv.states
        layouts += 1
        if any(wins):
            winnable += 1
            for i, w in enumerate(wins):
                if not w:
                    dominant[i] = False
        for i, w in enumerate(wins):
            win_counts[i] += w
    true_sv = Solver({q: set(h) for q, h in hands.items()})
    true_wins = true_sv.move_wins(codes, p, true_sv.legal_moves(codes, p))
    best = max(win_counts)
    h_idx = legal.index(heur_card)
    return {
        "game": game_seed, "turn": turn, "player": p, "hand_size": len(hands[p]),
        "legal": len(legal), "layouts": layouts, "exhaustive": int(selection.exhaustive),
        "winnable_layouts": winnable,
        "has_dominant": int(winnable > 0 and any(dominant)),
        "never_winnable": int(winnable == 0),
        "choice_irrelevant": int(len(set(win_counts)) == 1 and (best == 0 or best == layouts) ),
        "best_win_rate": best / layouts,
        "heuristic_win_rate": win_counts[h_idx] / layouts,
        "heuristic_regret": (best - win_counts[h_idx]) / layouts,
        "heuristic_is_best": int(win_counts[h_idx] == best),
        "true_layout_best_is_sample_best": int(true_wins[win_counts.index(best)] or not any(true_wins)),
        "win_counts": " ".join(map(str, win_counts)),
        "states": states, "seconds": round(time.perf_counter() - t, 3),
    }

if __name__ == "__main__":
    games, K, workers, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    t0 = time.perf_counter()
    points = []
    for g in range(games):
        points.extend(collect_decisions(10_000 + g))
    print(f"{len(points)} decision points from {games} games", flush=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(analyse, [(pt, K) for pt in points], chunksize=1)):
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(points)}] {time.perf_counter()-t0:.0f}s", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"done in {time.perf_counter()-t0:.0f}s -> {out}")
