"""Do independently trained self-play players pick the same moves, and does it matter?

usage:
  selfplay_compare.py swap GAMES NET...           one seat of A among three of B, every pair
  selfplay_compare.py moves OUT.csv GAMES POS REPS NET... [--ceiling SUFFIX] [--tables NAME...]
      Positions from each player's own self-play (all four seats that player). At each
      sampled choice position, every legal move is played out REPS times at that player's
      table (everyone continues with that player's policy), with common random numbers
      across the moves. The hidden deal is the real one, which is a draw from the true
      posterior given the history, so averages over positions are unbiased values per
      information state. Writes one row per (table, position, move): its value and
      which players (and the heuristic) would pick it.
"""
import argparse, csv, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfplay as sp
from selfplay_eval import play
from best_response import opponent_policy
from seven_hearts import CARD_BY_INDEX, GameState, card_index, card_set_mask, full_deck


def gumbel(shape, generator):
    return -torch.log(-torch.log(torch.rand(shape, generator=generator).clamp(1e-9, 1 - 1e-7)))


@torch.no_grad()
def collect(net, games, seed):
    """Self-play of `net` at all seats; every state with its turn, plus the full move history."""
    gen = torch.Generator().manual_seed(seed)
    t = sp.Tables(games, "cpu", gen)
    raws, masks, cards_l, played, movers = [], [], [], [], []
    while bool((t.winner < 0).any()):
        legal = t.public_legal()
        mask, cards = t.action_mask(legal)
        raw = t.raw()
        logits = sp.masked_logits(net(*sp.features(raw, False)), mask) + gumbel((games, 12), gen)
        card = torch.where(mask.any(1), cards.gather(1, logits.argmax(1, keepdim=True)).squeeze(1), torch.full((games,), -1))
        active = t.winner < 0
        raws.append(raw); masks.append(mask & active.unsqueeze(1)); cards_l.append(cards)
        played.append(torch.where(active, card, torch.full_like(card, -2))); movers.append(t.current.clone())
        t.step(legal, card)
    return raws, masks, cards_l, torch.stack(played, 1), torch.stack(movers, 1)


def engine_state(played_row, mover_row, turn):
    state = GameState(hand_counts=(13,) * 4, current_player=int(mover_row[0]))
    for k in range(turn):
        c = int(played_row[k])
        state = state.after_play(int(mover_row[k]), CARD_BY_INDEX[c] if c >= 0 else None)
    return state


@torch.no_grad()
def values(net, raw, turn, mask, cards, reps, generator):
    """V[p, m]: mover's win rate after playing move m at position p, then everyone plays `net`."""
    P = mask.shape[0]
    pos, move = mask.nonzero(as_tuple=True)                     # one block per legal (position, move)
    rows = pos.repeat_interleave(reps)
    rep = torch.arange(reps).repeat(len(pos))
    first = move.repeat_interleave(reps)
    t = sp.Tables(len(rows), "cpu", owner=raw[0][rows].long())
    t.played_at = raw[1][rows].long(); t.refused = raw[2][rows].clone(); t.passed = raw[3][rows].clone()
    t.current = raw[4][rows].long(); t.counts = raw[5][rows].long(); t.turn = turn[rows].unsqueeze(1)
    hero = t.current.clone()
    step = 0
    while bool((t.winner < 0).any()):
        legal = t.public_legal()
        m, c = t.action_mask(legal)
        if step == 0:
            action = first
        else:
            noise = gumbel((P, reps, 12), generator)[rows, rep]  # shared across the moves of one position
            action = (sp.masked_logits(net(*sp.features(t.raw(), False)), m) + noise).argmax(1)
        card = torch.where(m.any(1), c.gather(1, action.unsqueeze(1)).squeeze(1), torch.full_like(action, -1))
        t.step(legal, card)
        step += 1
    win = (t.winner == hero).float().view(len(pos), reps).mean(1)
    V = torch.full((P, 12), float("nan"))
    V[pos, move] = win
    return V


def swap(args):
    nets = {os.path.basename(p): sp.load_actor(p) for p in args.nets}
    names = list(nets)
    print("hero \\ table " + " ".join(f"{n:>16}" for n in names))
    for a in names:
        row = []
        for b in names:
            seats = [[("hero" if s == g % 4 else "table") for s in range(4)] for g in range(args.games)]
            wins = play(args.games, {"hero": nets[a], "table": nets[b]}, seats, args.seed)
            share = sum(w == g % 4 for g, w in enumerate(wins)) / args.games
            row.append(f"{100 * share:6.2f}+/-{100 * (share * (1 - share) / args.games) ** .5:.2f}")
        print(f"{a:>13} " + " ".join(f"{x:>16}" for x in row), flush=True)


def moves(args):
    paths = list(args.nets)
    extra = [p.replace(".pt", args.ceiling + ".pt") for p in paths] if args.ceiling else []
    nets = {os.path.basename(p): sp.load_actor(p) for p in paths + extra}
    heuristic = opponent_policy("heuristic", full_deck())
    deck = full_deck()
    gen = torch.Generator().manual_seed(args.seed)
    out = open(args.out, "w", newline="")
    writer = csv.writer(out)
    writer.writerow(["table", "position", "turn", "legal", "move", "value"] + [f"p_{n}" for n in nets] +
                    [f"top_{n}" for n in nets] + ["top_heuristic"])
    for ti, path in enumerate(paths):
        table = os.path.basename(path)
        if args.tables and table not in args.tables:
            continue
        start = time.time()
        raws, masks, cards_l, played, movers = collect(nets[table], args.games, args.seed + 100 * ti)
        T = len(raws)
        choice = torch.stack([m.sum(1) >= 2 for m in masks], 1)  # games x turns
        g_idx, t_idx = choice.nonzero(as_tuple=True)
        pick = torch.randperm(len(g_idx), generator=gen)[:args.positions]
        g_idx, t_idx = g_idx[pick], t_idx[pick]
        raw = tuple(torch.stack([raws[t][i][g] for g, t in zip(g_idx.tolist(), t_idx.tolist())]) for i in range(6))
        mask = torch.stack([masks[t][g] for g, t in zip(g_idx.tolist(), t_idx.tolist())])
        cards = torch.stack([cards_l[t][g] for g, t in zip(g_idx.tolist(), t_idx.tolist())])
        probs = {n: F.softmax(sp.masked_logits(net(*sp.features(raw, False)), mask), 1) for n, net in nets.items()}
        heur = torch.empty(len(g_idx), dtype=torch.long)
        for k, (g, t) in enumerate(zip(g_idx.tolist(), t_idx.tolist())):
            state = engine_state(played[g], movers[g], t)
            p = int(movers[g][t])
            hand = card_set_mask(CARD_BY_INDEX[c] for c in range(52) if int(raw[0][k][c]) == p and int(raw[1][k][c]) < 0)
            best = max(heuristic(state, p, hand), key=lambda mw: mw[1])[0]
            heur[k] = int((cards[k] == card_index(best)).nonzero()[0])
        V = torch.cat([values(nets[table], tuple(r[i:i + args.chunk] for r in raw), t_idx[i:i + args.chunk],
                              mask[i:i + args.chunk], cards[i:i + args.chunk], args.reps, gen)
                       for i in range(0, len(g_idx), args.chunk)])
        for k in range(len(g_idx)):
            tops = [int(probs[n][k].argmax()) for n in nets]
            for m in mask[k].nonzero().squeeze(1).tolist():
                writer.writerow([table, k, int(t_idx[k]), int(mask[k].sum()), m, f"{V[k, m]:.5f}"] +
                                [f"{probs[n][k, m]:.4f}" for n in nets] + [int(x == m) for x in tops] + [int(heur[k] == m)])
        out.flush()
        print(f"{table}: {len(g_idx)} positions, {time.time() - start:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("swap"); a.add_argument("games", type=int); a.add_argument("nets", nargs="+")
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("moves"); b.add_argument("out"); b.add_argument("games", type=int)
    b.add_argument("positions", type=int); b.add_argument("reps", type=int); b.add_argument("nets", nargs="+")
    b.add_argument("--ceiling", default=""); b.add_argument("--seed", type=int, default=0)
    b.add_argument("--chunk", type=int, default=50)
    b.add_argument("--tables", nargs="*", default=[], help="only these players' tables (basenames)")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    torch.manual_seed(args.seed)
    swap(args) if args.command == "swap" else moves(args)


if __name__ == "__main__":
    main()
