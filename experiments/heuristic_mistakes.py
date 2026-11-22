"""Where does the heuristic lose to the exact best response? One row per hero decision.

Hero at seat 0 against the given table, on hero hands drawn like small_deck_br.py
(same seed gives the same hands). Every decision with 2+ legal moves that the
heuristic reaches is written with its reach probability and the best-response
value of every legal move. contribution = reach * loss sums to the heuristic's
regret for that hand (performance-difference identity), so rows can be grouped
by any feature to see where the regret comes from.

usage: heuristic_mistakes.py CARDS_PER_SUIT HANDS|all WORKERS OUT.csv
         [--opponents random,greedy,heuristic] [--policy heuristic] [--seed 0] [--resume]
"""
import argparse, csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(10000)
from best_response import BestResponseSolver, hero_policy, opponent_policy
from seven_hearts import SUIT_ORDER, card_index, labels, mask_to_cards, reduced_deck
from small_deck_br import hero_hands

TOLERANCE = 1e-12


def table_label(state):
    runs = []
    for suit in SUIT_ORDER:
        run = state.table[suit]
        runs.append(f"{suit.value}:-" if run.low is None else f"{suit.value}:{run.low}-{run.high}")
    return " ".join(runs)


def solve_hand(job):
    index, cards_per_suit, hand_indices, opponent_names, policy_name = job
    deck = frozenset(reduced_deck(cards_per_suit))
    by_index = {card_index(card): card for card in deck}
    hand = frozenset(by_index[i] for i in hand_indices)
    opponents = {seat: opponent_policy(name, deck) for seat, name in zip((1, 2, 3), opponent_names)}
    solver = BestResponseSolver(hand, opponents, hero=0, deck=deck)
    best = solver.value()
    policy = hero_policy(policy_name, deck)
    records = solver.policy_decisions(policy)
    rows = []
    for record in records:
        top = max(record.values)
        chosen = max(record.choices, key=lambda choice: choice[1])[0]
        state = record.state
        rows.append({
            "hand_id": index,
            "start_hand": labels(sorted(hand, key=card_index)),
            "turn": len(state.history),
            "hero_hand": labels(mask_to_cards(record.hand_mask)),
            "hand_counts": " ".join(str(count) for count in state.hand_counts),
            "table": table_label(state),
            "legal": labels(record.legal),
            "values": " ".join(f"{value:.6f}" for value in record.values),
            "chosen": str(chosen),
            "best": labels(card for card, value in zip(record.legal, record.values) if value >= top - TOLERANCE),
            "contested": int(top - min(record.values) > TOLERANCE),
            "reach": record.reach,
            "loss": record.loss(),
            "contribution": record.reach * record.loss(),
            "hand_best_response": best,
        })
    return index, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cards_per_suit", type=int)
    parser.add_argument("hands")
    parser.add_argument("workers", type=int)
    parser.add_argument("out")
    parser.add_argument("--opponents", default="random,greedy,heuristic")
    parser.add_argument("--policy", default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    opponent_names = tuple(args.opponents.split(","))

    hands, _ = hero_hands(args.cards_per_suit, args.hands, args.seed)
    jobs = [(i, args.cards_per_suit, hand, opponent_names, args.policy) for i, hand in enumerate(hands)]
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="") as handle:
            done = {int(row["hand_id"]) for row in csv.DictReader(handle)}
        # A hand with no 2+ legal decisions writes no rows, so it would rerun; that is cheap.
        jobs = [job for job in jobs if job[0] not in done]
    print(f"{len(jobs)} hero hands to solve ({len(done)} already done), policy {args.policy}, "
          f"opponents {opponent_names}", flush=True)
    started = time.perf_counter()
    finished = 0
    mode = "a" if done else "w"
    with open(args.out, mode, newline="") as handle, ProcessPoolExecutor(args.workers) as pool:
        writer = None
        for future in as_completed([pool.submit(solve_hand, job) for job in jobs]):
            _, rows = future.result()
            finished += 1
            if rows:
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    if not done:
                        writer.writeheader()
                writer.writerows(rows)
                handle.flush()
            if finished % 10 == 0 or finished == len(jobs):
                print(f"[{finished}/{len(jobs)}] {time.perf_counter() - started:.0f}s", flush=True)
    print(f"done in {time.perf_counter() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
