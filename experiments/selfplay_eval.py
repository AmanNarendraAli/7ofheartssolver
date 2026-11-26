"""Play trained self-play actors against the engine's agents or each other.

usage: selfplay_eval.py GAMES HERO OPPONENT [--greedy]
HERO / OPPONENT: a .pt file from selfplay.py, or an opponent_policy name (heuristic, random, ...).
One HERO seat against three OPPONENT seats, seat rotating; prints HERO's first-out win share (even = 25%).
"""
import argparse, os, random, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfplay as sp
from best_response import opponent_policy
from seven_hearts import CARD_BY_INDEX, GameState, card_index, card_set_mask, full_deck


@torch.no_grad()
def play(n, agents, seats, seed, greedy=False, device="cpu"):
    """agents: name -> actor network or engine policy; seats[g][seat] -> agent name. Returns winners."""
    generator = torch.Generator(device=device).manual_seed(seed)
    rng = random.Random(seed)
    games = sp.Tables(n, device, generator)
    owner = games.owner.tolist()
    engine = any(not isinstance(agent, torch.nn.Module) for agent in agents.values())
    if engine:
        hands = [[{c for c in range(52) if owner[g][c] == s} for s in range(4)] for g in range(n)]
        states = [GameState(hand_counts=(13,) * 4, current_player=int(games.current[g])) for g in range(n)]
    names = list(agents)
    seat_ids = torch.tensor([[names.index(x) for x in row] for row in seats])
    while bool((games.winner < 0).any()):
        legal = games.public_legal()
        mask, cards = games.action_mask(legal)
        card = torch.full((n,), -1, dtype=torch.long)
        who = seat_ids.gather(1, games.current.unsqueeze(1)).squeeze(1)
        active = games.winner < 0
        raw = games.raw()
        for k, name in enumerate(names):
            idx = (active & (who == k) & mask.any(1)).nonzero().squeeze(1)
            if not len(idx):
                continue
            agent = agents[name]
            if isinstance(agent, torch.nn.Module):
                logits = sp.masked_logits(agent(*sp.features(tuple(r[idx] for r in raw), False)), mask[idx])
                if not greedy:
                    logits = logits - torch.log(-torch.log(torch.rand(logits.shape, generator=generator).clamp(1e-9, 1 - 1e-7)))
                card[idx] = cards[idx].gather(1, logits.argmax(1, keepdim=True)).squeeze(1)
            else:
                for g in idx.tolist():
                    p = int(games.current[g])
                    moves, weights = zip(*agent(states[g], p, card_set_mask(CARD_BY_INDEX[c] for c in hands[g][p])))
                    card[g] = card_index(rng.choices(moves, weights)[0])
        if engine:
            for g in active.nonzero().squeeze(1).tolist():
                c = int(card[g]); p = int(games.current[g])
                if c >= 0:
                    hands[g][p].discard(c)
                states[g] = states[g].after_play(p, CARD_BY_INDEX[c] if c >= 0 else None)
        games.step(legal, card)
    return games.winner.tolist()


def get_agent(name):
    return sp.load_actor(name) if name.endswith(".pt") else opponent_policy(name, full_deck())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("games", type=int); parser.add_argument("hero"); parser.add_argument("opponent")
    parser.add_argument("--greedy", action="store_true"); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    agents = {"hero": get_agent(args.hero), "opponent": get_agent(args.opponent)}
    seats = [["hero" if s == g % 4 else "opponent" for s in range(4)] for g in range(args.games)]
    wins = play(args.games, agents, seats, args.seed, args.greedy)
    share = sum(w == g % 4 for g, w in enumerate(wins)) / args.games
    print(f"{os.path.basename(args.hero)} vs 3 {os.path.basename(args.opponent)}: {100 * share:.2f}% +/- {100 * (share * (1 - share) / args.games) ** .5:.2f}")


if __name__ == "__main__":
    main()
