"""Group the heuristic's regret by features of the position and of the moves.

Input: heuristic_mistakes.py output. Regret shares are of the total contribution,
which equals the summed regret over hands, so they say where the regret comes from.

usage: analyse_mistakes.py MISTAKES.csv [--top 20]
"""
import argparse, csv, sys, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seven_hearts import Card, Suit

TOLERANCE = 1e-12


def cards(text):
    text = text.strip("[]")
    return [Card.parse(part) for part in text.split(", ")] if text else []


def parse_table(text):
    table = {}
    for part in text.split():
        suit, run = part.split(":")
        table[Suit(suit)] = None if run == "-" else tuple(int(x) for x in run.split("-"))
    return table


def move_kind(card, hand, table):
    """Describe a legal move from the hero's point of view."""
    run = table[card.suit]
    if card.rank == 7:
        same = [c.rank for c in hand if c.suit == card.suit and c.rank != 7]
        if not same:
            return "seven (lone)"
        if any(abs(rank - 7) == 1 for rank in same):
            return "seven (own neighbour)"
        return "seven (gap to own card)"
    step = -1 if card.rank < 7 else 1
    nxt = card.rank + step
    own_next = any(c.suit == card.suit and c.rank == nxt for c in hand)
    own_beyond = sum(1 for c in hand if c.suit == card.suit and (c.rank - card.rank) * step > 0)
    if own_next:
        return "continues own run"
    if own_beyond:
        return "opens gap to own card"
    return "end of own cards (frees next for others)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.path)))
    hands = {row["hand_id"] for row in rows}
    total = sum(float(row["contribution"]) for row in rows)
    mistakes = [row for row in rows if float(row["loss"]) > TOLERANCE]
    contested = [row for row in rows if row["contested"] == "1"]
    print(f"{len(hands)} hands, {len(rows)} decisions reached (2+ legal), {len(contested)} contested, "
          f"{len(mistakes)} heuristic mistakes")
    print(f"mean regret per hand {100 * total / len(hands):.3f} pts")
    reach_contested = sum(float(r["reach"]) for r in contested)
    reach_mistake = sum(float(r["reach"]) for r in mistakes)
    print(f"heuristic picks a best move on {100 * (1 - reach_mistake / reach_contested):.1f}% of contested decisions "
          f"(reach-weighted)")

    def share(key_fn, title):
        groups = defaultdict(lambda: [0.0, 0.0, 0.0])  # contribution, reach of mistakes, reach of contested
        for row in contested:
            key = key_fn(row)
            groups[key][2] += float(row["reach"])
            if float(row["loss"]) > TOLERANCE:
                groups[key][0] += float(row["contribution"])
                groups[key][1] += float(row["reach"])
        print(f"\n{title}")
        print(f"  {'':<52}{'regret share':>13}{'mistake rate':>14}{'avg loss':>10}")
        for key, (contribution, reach_m, reach_c) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
            rate = reach_m / reach_c if reach_c else 0.0
            avg = contribution / reach_m if reach_m else 0.0
            print(f"  {str(key):<52}{100 * contribution / total:>12.1f}%{100 * rate:>13.1f}%{100 * avg:>9.2f}")

    def hero_left(row):
        return f"hero holds {len(cards(row['hero_hand']))}"

    def tightest(row):
        counts = [int(x) for x in row["hand_counts"].split()]
        return f"fewest cards among opponents: {min(counts[1:])}"

    def chosen_vs_best(row):
        hand, table = cards(row["hero_hand"]), parse_table(row["table"])
        legal = cards(row["legal"])
        values = [float(v) for v in row["values"].split()]
        chosen = Card.parse(row["chosen"])
        top = max(values)
        best = [card for card, value in zip(legal, values) if value >= top - TOLERANCE]
        if chosen in best:
            return "chosen is best"
        best_kinds = sorted({move_kind(card, hand, table) for card in best})
        return f"{move_kind(chosen, hand, table)} -> {' / '.join(best_kinds)}"

    def chosen_kind(row):
        return move_kind(Card.parse(row["chosen"]), cards(row["hero_hand"]), parse_table(row["table"]))

    share(hero_left, "by hero's cards left")
    share(tightest, "by fewest cards among opponents")
    share(chosen_kind, "by kind of move the heuristic played")
    share(chosen_vs_best, "heuristic's move kind -> best move kind (mistakes only carry regret)")

    print(f"\ntop {args.top} mistakes by contribution:")
    for row in sorted(mistakes, key=lambda r: -float(r["contribution"]))[: args.top]:
        print(f"  hand {row['hand_id']:>3} turn {row['turn']:>2} hero {row['hero_hand']:<22} table {row['table']:<28} "
              f"counts {row['hand_counts']}  played {row['chosen']:<4} best {row['best']:<14} "
              f"loss {100 * float(row['loss']):5.2f} reach {100 * float(row['reach']):5.2f}%  "
              f"share {100 * float(row['contribution']) / total:4.1f}%")


if __name__ == "__main__":
    main()
