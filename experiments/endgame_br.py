"""Exact best response in real 52-card endgames, and the exact regret of each hero policy there.

Each game is a fresh full-deck deal played by the given policies: hero at seat 0
(--driver, default heuristic) and seats 1-3 the opponents (default random,
greedy, heuristic, as in the full-game eval). At the hero's first turn once the
opponents hold at most --max-hidden cards between them (and nobody is out),
the position becomes an endgame: the belief is the exact posterior over the
unseen cards given the whole history, and every value is the exact first-out
win probability from there. Games where someone goes out first give no endgame.

Endgames whose posterior keeps more than --max-layouts splits, or where an
evaluation reaches more than --node-budget hero decisions, are recorded with
status too_large and not solved (memory and time), so the summary says how many.

usage: endgame_br.py GAMES WORKERS OUT.csv [--max-hidden 12] [--max-layouts 200000]
         [--node-budget 1500000]
         [--opponents random,greedy,heuristic] [--driver heuristic]
         [--policies heuristic,seven_pair_rule,exact_vote,posterior_vote]
         [--seed 0] [--resume]
"""
import argparse, csv, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from math import factorial, sqrt
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(10000)
from best_response import BestResponseSolver, BudgetExceeded, hero_policy, opponent_policy
from seven_hearts import Card, GameState, Suit, card_index, card_set_mask, full_deck, labels

HERO = 0


def table_label(state):
    runs = []
    for suit in Suit:
        run = state.table[suit]
        runs.append(f"{suit.value}:-" if run.low is None else f"{suit.value}:{run.low}-{run.high}")
    return " ".join(runs)


def play_to_endgame(rng, deck, driver, opponents, max_hidden):
    """Deal and play until the endgame position; None if someone goes out first."""
    cards = sorted(deck, key=card_index)
    rng.shuffle(cards)
    hands = {seat: set(cards[13 * seat:13 * (seat + 1)]) for seat in range(4)}
    starter = next(seat for seat in range(4) if Card(Suit.HEARTS, 7) in hands[seat])
    state = GameState(hand_counts=(13,) * 4, current_player=starter)
    policies = {HERO: driver, **opponents}
    while all(hands.values()):
        player = state.current_player
        if player == HERO and sum(len(hands[seat]) for seat in opponents) <= max_hidden:
            return state, hands
        choices = policies[player](state, player, card_set_mask(hands[player]))
        moves, weights = zip(*choices)
        card = rng.choices(moves, weights)[0]
        if card is not None:
            hands[player].discard(card)
        state = state.after_play(player, card)
    return None


def solve_game(job):
    index, seed, max_hidden, max_layouts, node_budget, opponent_names, driver_name, policy_names = job
    deck = frozenset(full_deck())
    rng = random.Random(seed * 1_000_003 + index)
    opponents = {seat: opponent_policy(name, deck) for seat, name in zip((1, 2, 3), opponent_names)}
    played = play_to_endgame(rng, deck, opponent_policy(driver_name, deck), opponents, max_hidden)
    row = {"game": index, "status": "no_endgame"}
    if played is None:
        return row
    state, hands = played
    started = time.perf_counter()
    solver = BestResponseSolver.from_position(state, HERO, hands[HERO], opponents, deck)
    (_, belief), = solver.roots
    true_layout = tuple(card_set_mask(hands[seat]) if seat != HERO else 0 for seat in range(4))
    if belief.get(true_layout, 0.0) <= 0.0:
        raise AssertionError(f"game {index}: the true layout has zero posterior weight")
    counts = [len(hands[seat]) for seat in range(4)]
    hidden = sum(counts[1:])
    consistent = factorial(hidden) // (factorial(counts[1]) * factorial(counts[2]) * factorial(counts[3]))
    posterior_seconds = round(time.perf_counter() - started, 2)
    row.update({
        "turn": len(state.history),
        "hero_hand": labels(sorted(hands[HERO], key=card_index)),
        "hand_counts": " ".join(str(count) for count in counts),
        "table": table_label(state),
        "consistent_layouts": consistent,
        "posterior_layouts": solver.layouts,
        "posterior_seconds": posterior_seconds,
    })
    if solver.layouts > max_layouts:
        row["status"] = "too_large"
        return row
    solver.remember_best = False  # values only: saves memory
    solver.node_budget = node_budget
    started = time.perf_counter()
    try:
        row["best_response"] = solver.value()
        row["br_seconds"] = round(time.perf_counter() - started, 2)
        for name in policy_names:
            started = time.perf_counter()
            row[name] = solver.value(hero_policy(name, deck))
            row[f"{name}_seconds"] = round(time.perf_counter() - started, 2)
    except BudgetExceeded:
        row = {key: row[key] for key in ("game", "turn", "hero_hand", "hand_counts", "table",
                                         "consistent_layouts", "posterior_layouts", "posterior_seconds")}
        row["status"] = "too_large"
        return row
    row["status"] = "solved"
    row["opponent_queries"] = solver.opponent_queries
    return row


def summarise(rows, policy_names, games):
    statuses = Counter(row["status"] for row in rows)
    rows = [row for row in rows if row["status"] == "solved"]
    print(f"\n{len(rows)} endgames solved from {games} games; {statuses['no_endgame']} had someone out first, "
          f"{statuses['too_large']} too large to solve; +/- is 1 SE over solved endgames")
    if not rows:
        return
    br = [float(row["best_response"]) for row in rows]
    print(f"  best response      first-out win {100 * mean(br):6.2f}%  (given the endgame)")
    for name in policy_names:
        regret = [float(row["best_response"]) - float(row[name]) for row in rows]
        se = sqrt(sum((r - mean(regret)) ** 2 for r in regret) / (len(regret) - 1) / len(regret)) if len(regret) > 1 else 0.0
        print(f"  {name:<18} first-out win {100 * mean(float(row[name]) for row in rows):6.2f}%  "
              f"regret {100 * mean(regret):.3f} +/- {100 * se:.3f} pts  "
              f"optimal in {100 * mean(r < 1e-9 for r in regret):.1f}%  worst {100 * max(regret):.2f} pts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("games", type=int)
    parser.add_argument("workers", type=int)
    parser.add_argument("out")
    parser.add_argument("--max-hidden", type=int, default=12)
    parser.add_argument("--max-layouts", type=int, default=200_000)
    parser.add_argument("--node-budget", type=int, default=1_500_000)
    parser.add_argument("--opponents", default="random,greedy,heuristic")
    parser.add_argument("--driver", default="heuristic")
    parser.add_argument("--policies", default="heuristic,seven_pair_rule,exact_vote,posterior_vote")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    opponent_names = tuple(args.opponents.split(","))
    policy_names = tuple(args.policies.split(","))
    assert len(opponent_names) == 3

    jobs = [(i, args.seed, args.max_hidden, args.max_layouts, args.node_budget, opponent_names, args.driver, policy_names) for i in range(args.games)]
    rows = []
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="") as handle:
            rows = list(csv.DictReader(handle))
        done = {int(row["game"]) for row in rows}
        jobs = [job for job in jobs if job[0] not in done]
        print(f"resuming: {len(done)} games already done", flush=True)
    print(f"{len(jobs)} games, max hidden {args.max_hidden}, opponents {opponent_names}, "
          f"driver {args.driver}, policies {policy_names}", flush=True)
    fields = ["game", "status", "turn", "hero_hand", "hand_counts", "table", "consistent_layouts", "posterior_layouts",
              "posterior_seconds", "best_response", "br_seconds"]
    fields += [f for name in policy_names for f in (name, f"{name}_seconds")] + ["opponent_queries"]
    started = time.perf_counter()
    finished = 0
    with open(args.out, "a" if rows else "w", newline="") as handle, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not rows:
            writer.writeheader()
        for future in as_completed([pool.submit(solve_game, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            finished += 1
            if finished % 10 == 0 or finished == len(jobs):
                print(f"[{finished}/{len(jobs)}] {time.perf_counter() - started:.0f}s", flush=True)
    summarise(rows, policy_names, args.games)
    print(f"done in {time.perf_counter() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
