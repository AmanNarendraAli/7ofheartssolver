"""Prototype: exact perfect-information first-out solver for one deal.

State = (suit codes, player). Hands are implied by the deal + table.
Value = first-out winner under rational play with the repo's tie-breaker.
"""
import sys, time
sys.setrecursionlimit(10000)

UNKNOWN = 255

def code_of(low, high):
    return 0 if low is None else 1 + (low - 1) * 7 + (high - 7)

DECODE = [(None, None)] + [(l, h) for l in range(1, 8) for h in range(7, 14)]

class Solver:
    def __init__(self, owner):
        # owner[suit][rank] -> player or -1 (absent); suits in card-index order C,D,H,S
        self.owner = owner
        self.initial = [sum(1 for s in range(4) for r in range(1, 14) if owner[s][r] == p) for p in range(4)]
        # inside[s][code][p] = cards of p already played in suit s at that code
        self.inside = []
        for s in range(4):
            rows = []
            for code, (l, h) in enumerate(DECODE):
                c = [0, 0, 0, 0]
                if l is not None:
                    for r in range(l, h + 1):
                        o = owner[s][r]
                        if o >= 0:
                            c[o] += 1
                rows.append(c)
            self.inside.append(rows)
        # moves[s][code] -> list of (owner, rank, next_code), in rank order (low side first)
        self.moves = []
        for s in range(4):
            rows = []
            for code, (l, h) in enumerate(DECODE):
                mv = []
                if l is None:
                    if owner[s][7] >= 0:
                        mv.append((owner[s][7], 7, code_of(7, 7)))
                else:
                    if l > 1 and owner[s][l - 1] >= 0:
                        mv.append((owner[s][l - 1], l - 1, code_of(l - 1, h)))
                    if h < 13 and owner[s][h + 1] >= 0:
                        mv.append((owner[s][h + 1], h + 1, code_of(l, h + 1)))
                rows.append(mv)
            self.moves.append(rows)
        self.memo = bytearray([UNKNOWN]) * (50 ** 4 * 4)
        self.states = 0

    def count(self, codes, p):
        ins = self.inside
        return self.initial[p] - ins[0][codes[0]][p] - ins[1][codes[1]][p] - ins[2][codes[2]][p] - ins[3][codes[3]][p]

    def legal(self, codes, p):
        out = []
        if codes == [0, 0, 0, 0] or codes == (0, 0, 0, 0):
            if self.owner[2][7] == p:
                out.append((2, code_of(7, 7)))
            return out
        for s in range(4):
            for o, r, nc in self.moves[s][codes[s]]:
                if o == p:
                    out.append((s, nc))
        return out  # card-index order: suit major, then low side before high side (lower rank first)

    def solve(self, codes, p):
        # skip forced passes
        for _ in range(4):
            legal = self.legal(codes, p)
            if legal:
                break
            p = (p + 1) % 4
        else:
            raise RuntimeError("deadlock")
        key = ((codes[0] * 50 + codes[1]) * 50 + codes[2]) * 50 + codes[3]
        key = key * 4 + p
        v = self.memo[key]
        if v != UNKNOWN:
            return v
        self.states += 1
        nxt = (p + 1) % 4
        best_other = None
        result = None
        for s, nc in legal:
            child = list(codes)
            child[s] = nc
            if self.count(child, p) == 0:
                result = p
                break
            w = self.solve(child, nxt)
            if w == p:
                result = p
                break
            if w != nxt and best_other is None:
                best_other = w
        if result is None:
            result = best_other if best_other is not None else nxt
        self.memo[key] = result
        return result

    def solve_root(self):
        holder = self.owner[2][7]
        return self.solve([0, 0, 0, 0], holder)

def owner_from_hands(hands):
    owner = [[-1] * 14 for _ in range(4)]
    suit_idx = {"C": 0, "D": 1, "H": 2, "S": 3}
    for p, hand in hands.items():
        for c in hand:
            owner[suit_idx[c.suit.value]][c.rank] = p
    return owner

if __name__ == "__main__":
    import random
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
    from seven_hearts import deal_random_hands
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for seed in range(n):
        hands = deal_random_hands(random.Random(seed))
        sv = Solver(owner_from_hands(hands))
        t = time.perf_counter()
        w = sv.solve_root()
        print(f"seed={seed} winner=P{w} states={sv.states} time={time.perf_counter()-t:.2f}s", flush=True)
