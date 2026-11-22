"""Does the ideal move depend on who you play? Exact best responses against several opponent tables.

For each hero hand (hero at seat 0) and each opponent table, compute the exact best
response. Then, for every ordered pair (assumed A, actual B), play the best response
to A against B exactly (see best_response.evaluate_assumed_table). Cross regret
BR_B - value_B(BR_A) is what you lose by being optimal against the wrong opponents.
The heuristic's exact regret against each table is reported alongside.

usage: cross_table_br.py CARDS_PER_SUIT HANDS|all WORKERS OUT.csv [--tables mixed,heuristic,greedy,random] [--seed 0]
"""
import argparse, csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import sqrt
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(10000)
from best_response import BestResponseSolver, evaluate_assumed_table, hero_policy, opponent_policy
from seven_hearts import Card, Suit, card_index, labels, reduced_deck
from small_deck_br import hero_hands

TABLES = {
    "mixed": ("random", "greedy", "heuristic"),  # the full-game eval table, in seat order
    "heuristic": ("heuristic",) * 3,
    "greedy": ("greedy",) * 3,
    "random": ("random",) * 3,
}


def solve_hand(job):
    index, cards_per_suit, hand_indices, table_names = job
    deck = frozenset(reduced_deck(cards_per_suit))
    by_index = {card_index(card): card for card in deck}
    hand = frozenset(by_index[i] for i in hand_indices)
    started = time.perf_counter()
    solvers = {
        name: BestResponseSolver(
            hand, {seat: opponent_policy(policy, deck) for seat, policy in zip((1, 2, 3), TABLES[name])}, deck=deck
        )
        for name in table_names
    }
    row = {
        "hand_id": index,
        "hand": labels(sorted(hand, key=card_index)),
        "holds_7h": int(Card(Suit.HEARTS, 7) in hand),
        "layouts": next(iter(solvers.values())).layouts,
    }
    heuristic = hero_policy("heuristic", deck)
    for name, solver in solvers.items():
        row[f"br_{name}"] = solver.value()
        row[f"heuristic_{name}"] = solver.value(heuristic)
    for assumed in table_names:
        for actual in table_names:
            if assumed == actual:
                continue
            result = evaluate_assumed_table(solvers[assumed], solvers[actual])
            prefix = f"{assumed}_vs_{actual}"
            row[f"{prefix}_value"] = result.value
            row[f"{prefix}_decision_mass"] = result.decision_mass
            row[f"{prefix}_agree_mass"] = result.agree_mass
            row[f"{prefix}_contested_mass"] = result.contested_mass
            row[f"{prefix}_contested_agree_mass"] = result.contested_agree_mass
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row


def se(values):
    m = mean(values)
    return sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1) / len(values)) if len(values) > 1 else 0.0


def summarise(rows, table_names, exhaustive):
    print(f"\n{len(rows)} hero hands ({'every hand: exact' if exhaustive else 'sampled: +/- is 1 SE over hands'})")
    print("best response and heuristic, first-out win, by actual table:")
    for name in table_names:
        regret = [row[f"br_{name}"] - row[f"heuristic_{name}"] for row in rows]
        print(f"  {name:<10} BR {100 * mean(row[f'br_{name}'] for row in rows):6.2f}%   heuristic regret "
              f"{100 * mean(regret):.3f}" + ("" if exhaustive else f" +/- {100 * se(regret):.3f}") +
              f" pts, optimal on {100 * mean(r < 1e-9 for r in regret):.1f}% of hands")
    print("cross regret (pts) = BR_actual - value of playing BR_assumed against actual; rows assumed, columns actual:")
    print("  " + " " * 10 + "".join(f"{name:>12}" for name in table_names))
    for assumed in table_names:
        cells = []
        for actual in table_names:
            if assumed == actual:
                cells.append(f"{'-':>12}")
                continue
            regret = [row[f"br_{actual}"] - row[f"{assumed}_vs_{actual}_value"] for row in rows]
            cells.append(f"{100 * mean(regret):>12.3f}")
        print(f"  {assumed:<10}" + "".join(cells))
    print("share of decisions faced (probability-weighted) where the assumed-table move is also ideal vs actual:")
    print("  " + " " * 10 + "".join(f"{name:>12}" for name in table_names))
    for assumed in table_names:
        cells = []
        for actual in table_names:
            if assumed == actual:
                cells.append(f"{'-':>12}")
                continue
            prefix = f"{assumed}_vs_{actual}"
            agree = sum(row[f"{prefix}_agree_mass"] / row["layouts"] for row in rows)
            faced = sum(row[f"{prefix}_decision_mass"] / row["layouts"] for row in rows)
            cells.append(f"{100 * agree / faced:>11.2f}%")
        print(f"  {assumed:<10}" + "".join(cells))
    print("same, only decisions where the actual table's move values differ at all:")
    for assumed in table_names:
        cells = []
        for actual in table_names:
            if assumed == actual:
                cells.append(f"{'-':>12}")
                continue
            prefix = f"{assumed}_vs_{actual}"
            agree = sum(row[f"{prefix}_contested_agree_mass"] / row["layouts"] for row in rows)
            faced = sum(row[f"{prefix}_contested_mass"] / row["layouts"] for row in rows)
            cells.append(f"{100 * agree / faced:>11.2f}%" if faced else f"{'n/a':>12}")
        print(f"  {assumed:<10}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cards_per_suit", type=int)
    parser.add_argument("hands")
    parser.add_argument("workers", type=int)
    parser.add_argument("out")
    parser.add_argument("--tables", default=",".join(TABLES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    table_names = tuple(args.tables.split(","))
    assert all(name in TABLES for name in table_names) and len(table_names) >= 2

    hands, exhaustive = hero_hands(args.cards_per_suit, args.hands, args.seed)
    jobs = [(i, args.cards_per_suit, hand, table_names) for i, hand in enumerate(hands)]
    print(f"{len(jobs)} hero hands, {args.cards_per_suit} cards per suit, tables {table_names}", flush=True)
    started = time.perf_counter()
    rows = []
    with open(args.out, "w", newline="") as handle, ProcessPoolExecutor(args.workers) as pool:
        writer = None
        for future in as_completed([pool.submit(solve_hand, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if len(rows) % 10 == 0 or len(rows) == len(jobs):
                print(f"[{len(rows)}/{len(jobs)}] {time.perf_counter() - started:.0f}s", flush=True)
    rows.sort(key=lambda row: row["hand_id"])
    summarise(rows, table_names, exhaustive)
    print(f"done in {time.perf_counter() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
