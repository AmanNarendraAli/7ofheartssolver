"""Exact best response vs fixed opponents on reduced decks, and the exact regret of each hero policy.

The hero sits at seat 0; seats 1-3 play the given opponent policies in turn order
(default random, greedy, heuristic: the same cyclic order as the full-game eval
table). For each hero hand, every deal of the unseen cards is enumerated, so all
values are exact first-out-win probabilities given that hand. With --hands all
every possible hero hand is covered and the averages are exact for the deck.

usage: small_deck_br.py CARDS_PER_SUIT HANDS|all WORKERS OUT.csv
         [--opponents random,greedy,heuristic]
         [--policies heuristic,greedy,random,exact_vote,posterior_vote]
         [--seed 0]
"""
import argparse, csv, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from math import comb, sqrt
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(10000)
from best_response import BestResponseSolver, hero_policy, opponent_policy
from seven_hearts import Card, Suit, card_index, labels, reduced_deck

POLICIES = ("heuristic", "greedy", "random", "exact_vote", "posterior_vote", "mc", "seven_gap", "seven_rule", "seven_pair_rule")


def solve_hand(job):
    index, cards_per_suit, hand_indices, opponent_names, policy_names = job
    deck = frozenset(reduced_deck(cards_per_suit))
    by_index = {card_index(card): card for card in deck}
    hand = frozenset(by_index[i] for i in hand_indices)
    opponents = {seat: opponent_policy(name, deck) for seat, name in zip((1, 2, 3), opponent_names)}
    started = time.perf_counter()
    solver = BestResponseSolver(hand, opponents, hero=0, deck=deck)
    row = {
        "hand_id": index,
        "hand": labels(sorted(hand, key=card_index)),
        "holds_7h": int(Card(Suit.HEARTS, 7) in hand),
        "layouts": solver.layouts,
        "best_response": solver.value(),
        "br_decision_nodes": solver.decision_nodes,
        "br_seconds": round(time.perf_counter() - started, 2),
    }
    for name in policy_names:
        started = time.perf_counter()
        row[name] = solver.value(hero_policy(name, deck))
        row[f"{name}_seconds"] = round(time.perf_counter() - started, 2)
    row["opponent_queries"] = solver.opponent_queries
    return row


def hero_hands(cards_per_suit, hands, seed):
    deck = sorted(card_index(card) for card in reduced_deck(cards_per_suit))
    total = comb(len(deck), cards_per_suit)
    if hands == "all" or int(hands) >= total:
        return [tuple(hand) for hand in combinations(deck, cards_per_suit)], True
    rng = random.Random(seed)
    chosen = set()
    while len(chosen) < int(hands):
        chosen.add(tuple(sorted(rng.sample(deck, cards_per_suit))))
    return sorted(chosen), False


def summarise(rows, policy_names, exhaustive):
    br = [row["best_response"] for row in rows]
    print(f"\n{len(rows)} hero hands ({'every hand: exact' if exhaustive else 'sampled: +/- is 1 SE over hands'})")
    print(f"  best response      first-out win {100 * mean(br):6.2f}%")
    for name in policy_names:
        regret = [row["best_response"] - row[name] for row in rows]
        se = sqrt(sum((r - mean(regret)) ** 2 for r in regret) / (len(regret) - 1) / len(regret)) if len(regret) > 1 else 0.0
        optimal = mean(r < 1e-9 for r in regret)
        print(f"  {name:<18} first-out win {100 * mean(row[name] for row in rows):6.2f}%  "
              f"regret {100 * mean(regret):.3f}" + ("" if exhaustive else f" +/- {100 * se:.3f}") +
              f" pts  optimal on {100 * optimal:.1f}% of hands  worst {100 * max(regret):.2f} pts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cards_per_suit", type=int)
    parser.add_argument("hands")
    parser.add_argument("workers", type=int)
    parser.add_argument("out")
    parser.add_argument("--opponents", default="random,greedy,heuristic")
    parser.add_argument("--policies", default="heuristic,greedy,random,exact_vote,posterior_vote")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="keep hands already in OUT and append the rest")
    args = parser.parse_args()
    opponent_names = tuple(args.opponents.split(","))
    policy_names = tuple(args.policies.split(","))
    assert len(opponent_names) == 3 and all(name in POLICIES for name in policy_names)

    hands, exhaustive = hero_hands(args.cards_per_suit, args.hands, args.seed)
    jobs = [(i, args.cards_per_suit, hand, opponent_names, policy_names) for i, hand in enumerate(hands)]
    print(f"{len(jobs)} hero hands, {args.cards_per_suit} cards per suit, opponents {opponent_names}, "
          f"policies {policy_names}", flush=True)
    started = time.perf_counter()
    rows = []
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({key: (value if key == "hand" else float(value)) for key, value in row.items()})
        done = {int(row["hand_id"]) for row in rows}
        jobs = [job for job in jobs if job[0] not in done]
        print(f"resuming: {len(done)} hands already done, {len(jobs)} to go", flush=True)
    with open(args.out, "a" if rows else "w", newline="") as handle, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])) if rows else None
        for future in as_completed([pool.submit(solve_hand, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            new = len(rows) - (len(hands) - len(jobs))
            if new % 10 == 0 or new == len(jobs):
                print(f"[{len(rows)}/{len(hands)}] {time.perf_counter() - started:.0f}s", flush=True)
    rows.sort(key=lambda row: row["hand_id"])
    summarise(rows, policy_names, exhaustive)
    print(f"done in {time.perf_counter() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
