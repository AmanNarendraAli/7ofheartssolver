"""Full-deck Sevens self-play from scratch: first-out win, four copies of one network.

The deck has 8 arms (each suit, each side of its seven, six cards outward) and 4 sevens,
so there are 12 possible moves. The networks share weights over suits and arms, which
builds in the game's symmetry (any suit relabelling, any low/high flip).

Actor: sees only the acting player's information: own hand, table, hand counts, who
played each card and when, and, per opponent and card, how often they left it unplayed
while it was playable and whether a forced pass shows they do not hold it.
Critic: also sees every hand; predicts all four players' win probabilities. It only
lowers the variance of training and is never used to play.

Training: PPO on TD(lambda) advantages in win probability, with an annealed entropy bonus.

Exploiter mode (--frozen P.pt [--init P.pt]): one seat learns against three fixed copies of
P; its win share above 25% is a lower bound on how exploitable P is.

usage: selfplay.py train OUT.pt [--seed 1] [--hours 3] [--games 8192] [--width 128] [--frozen P.pt]
       (writes OUT.pt, snapshots OUT_<iteration>.pt and OUT.log; --resume continues)
"""
import argparse, os, sys, time
import torch
from torch import nn
import torch.nn.functional as F

SEVEN = [13 * s + 6 for s in range(4)]
ARMS = [[13 * s + 6 + sign * d for d in range(1, 7)] for s in range(4) for sign in (-1, 1)]
ACTION_CARD_SEVEN = SEVEN
HEART_SEVEN = SEVEN[2]
MAX_TURNS = 120
OWN_CHANNELS = 13   # per card: mine, played, played by (4 rel. seats), refused x3, passed x3, when
FULL_CHANNELS = OWN_CHANNELS + 3  # critic: plus held by each opponent
GLOBALS = 4 + 1


class Tables:
    """N games in lockstep. owner: who was dealt each card; played_at: turn it was played or -1."""

    def __init__(self, n, device, generator=None, owner=None):
        self.n, self.device = n, device
        if owner is None:
            order = torch.rand(n, 52, device=device, generator=generator).argsort(1)
            owner = torch.empty(n, 52, dtype=torch.long, device=device)
            owner.scatter_(1, order, torch.arange(52, device=device).div(13, rounding_mode="floor").expand(n, 52))
        self.owner = owner
        self.rows = torch.arange(n, device=device)
        self.played_at = torch.full((n, 52), -1, dtype=torch.long, device=device)
        self.refused = torch.zeros(n, 4, 52, dtype=torch.uint8, device=device)
        self.passed = torch.zeros(n, 4, 52, dtype=torch.uint8, device=device)
        self.current = owner[:, HEART_SEVEN].clone()
        self.counts = torch.full((n, 4), 13, dtype=torch.long, device=device)
        self.turn = 0
        self.winner = torch.full((n,), -1, dtype=torch.long, device=device)
        self.arms = torch.tensor(ARMS, device=device)
        self.sevens = torch.tensor(SEVEN, device=device)

    def public_legal(self):
        """(N, 52) cards playable now by whoever holds them."""
        played = self.played_at >= 0
        legal = torch.zeros_like(played)
        arm_played = played[:, self.arms]                         # N, 8, 6
        length = arm_played.long().sum(2)                         # arms fill outward, so a count is a position
        opened = played[:, self.sevens].repeat_interleave(2, 1)   # N, 8
        can = opened & (length < 6)
        nxt = self.arms.expand(self.n, 8, 6).gather(2, length.clamp(max=5).unsqueeze(2)).squeeze(2)
        legal.scatter_(1, nxt, can)                               # arms never share a card
        any_played = played.any(1, keepdim=True)
        sevens = ~played[:, self.sevens] & any_played
        sevens[:, 2] |= ~any_played[:, 0]
        legal[:, self.sevens] = sevens
        self._next, self._can = nxt, can
        return legal

    def action_mask(self, legal):
        """(N, 12) legal moves of the current player: 8 arms then 4 sevens; and the card of each."""
        mine = (self.owner == self.current.unsqueeze(1)) & (self.played_at < 0)
        own = legal & mine
        cards = torch.cat([self._next, self.sevens.expand(self.n, 4)], 1)
        mask = own.gather(1, cards)
        mask[:, :8] &= self._can
        return mask, cards

    def raw(self):
        """Compact copy of the state, enough to rebuild features."""
        return (self.owner.to(torch.uint8), self.played_at.to(torch.int16), self.refused.clone(), self.passed.clone(),
                self.current.to(torch.uint8), self.counts.to(torch.uint8))

    def step(self, legal, card):
        """`card` (N,): card index, or -1 for a pass; finished games are left unchanged."""
        active = self.winner < 0
        moved = active & (card >= 0)
        passing = active & (card < 0)
        seat = self.current
        # Fixed-shape operations only: on MPS every new shape compiles and keeps a new graph.
        onehot = F.one_hot(card.clamp(min=0), 52).bool() & moved.unsqueeze(1)
        seat_rows = F.one_hot(seat, 4).bool().unsqueeze(2)                      # N, 4, 1
        # everything playable that this player did not play: refused, or proven absent by a pass
        left = (legal & ~onehot & moved.unsqueeze(1)).unsqueeze(1) & seat_rows
        absent = (legal & passing.unsqueeze(1)).unsqueeze(1) & seat_rows
        self.refused = self.refused + left.to(torch.uint8)
        self.passed = self.passed | absent.to(torch.uint8)
        self.played_at = torch.where(onehot, torch.zeros_like(self.played_at) + self.turn, self.played_at)
        self.counts = self.counts - (F.one_hot(seat, 4) * moved.unsqueeze(1))
        won = moved & (self.counts.gather(1, seat.unsqueeze(1)).squeeze(1) == 0)
        self.winner = torch.where(won, seat, self.winner)
        self.current = torch.where(active, (seat + 1) % 4, seat)
        self.turn += 1


def features(raw, full):
    """Feature tensor per card (N, 52, C) and globals (N, 5), relative to the current player."""
    owner, played_at, refused, passed, current, counts = raw
    n = owner.shape[0]
    device = owner.device
    cur = current.long()
    owner = owner.long()
    played = played_at >= 0
    rel = (owner - cur.unsqueeze(1)) % 4
    seats = (cur.unsqueeze(1) + torch.arange(1, 4, device=device)) % 4      # N, 3 opponents in turn order
    rows = torch.arange(n, device=device).unsqueeze(1)
    ref = refused[rows, seats].float() * 0.25                               # N, 3, 52
    pas = passed[rows, seats].float()
    chans = [((rel == 0) & ~played).float(), played.float()]
    chans += [((rel == r) & played).float() for r in range(4)]
    chans += [ref[:, i] * (~played) for i in range(3)]
    chans += [pas[:, i] * (~played) for i in range(3)]
    chans.append(played_at.clamp(min=0).float() / 60.0)
    if full:
        chans += [((rel == r) & ~played).float() for r in (1, 2, 3)]
    cards = torch.stack(chans, 2)
    order = (cur.unsqueeze(1) + torch.arange(4, device=device)) % 4
    turn = played_at.max(1).values.float().clamp(min=0) / 60.0
    globals_ = torch.cat([counts.gather(1, order).float() / 13.0, turn.unsqueeze(1)], 1)
    return cards, globals_


def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, out))


class SevensNet(nn.Module):
    """Weight-shared over suits and arms. Outputs 12 move logits, or 4 win logits for the critic."""

    def __init__(self, full, width=128, blocks=3):
        super().__init__()
        self.full = full
        channels = FULL_CHANNELS if full else OWN_CHANNELS
        self.arm_in = nn.Linear(6 * channels, width)
        self.seven_in = nn.Linear(channels, width)
        self.glob_in = nn.Linear(GLOBALS, width)
        self.arm_blocks = nn.ModuleList(mlp(4 * width, width, width) for _ in range(blocks))
        self.suit_blocks = nn.ModuleList(mlp(3 * width, width, width) for _ in range(blocks))
        self.glob_blocks = nn.ModuleList(mlp(3 * width, width, width) for _ in range(blocks))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3 * blocks))
        if full:
            self.out = mlp(width, width, 4)
        else:
            self.arm_out = mlp(2 * width, width, 1)
            self.suit_out = mlp(2 * width, width, 1)
        self.register_buffer("arms", torch.tensor(ARMS), persistent=False)
        self.register_buffer("sevens", torch.tensor(SEVEN), persistent=False)

    def forward(self, cards, globals_):
        n = cards.shape[0]
        arm = self.arm_in(cards[:, self.arms].reshape(n, 8, -1))     # N, 8, W
        suit = self.seven_in(cards[:, self.sevens])                  # N, 4, W
        glob = self.glob_in(globals_)                                # N, W
        for k, (fa, fs, fg) in enumerate(zip(self.arm_blocks, self.suit_blocks, self.glob_blocks)):
            na, ns, ng = self.norms[3 * k], self.norms[3 * k + 1], self.norms[3 * k + 2]
            pairs = arm.view(n, 4, 2, -1)
            twin = pairs.flip(2).reshape(n, 8, -1)
            g8, g4 = glob.unsqueeze(1).expand(-1, 8, -1), glob.unsqueeze(1).expand(-1, 4, -1)
            arm2 = na(arm + fa(torch.cat([arm, twin, suit.repeat_interleave(2, 1), g8], 2)))
            suit2 = ns(suit + fs(torch.cat([suit, pairs.sum(2), g4], 2)))
            glob = ng(glob + fg(torch.cat([glob, arm2.mean(1), suit2.mean(1)], 1)))
            arm, suit = arm2, suit2
        if self.full:
            return self.out(glob)
        g8, g4 = glob.unsqueeze(1).expand(-1, 8, -1), glob.unsqueeze(1).expand(-1, 4, -1)
        return torch.cat([self.arm_out(torch.cat([arm, g8], 2)).squeeze(2),
                          self.suit_out(torch.cat([suit, g4], 2)).squeeze(2)], 1)


def masked_logits(logits, mask):
    return logits.masked_fill(~mask, -1e9)


@torch.no_grad()
def rollout(actor, critic, n, device, lam, frozen=None):
    """Play n games; returns the flattened decisions and value targets.
    With `frozen`, game g's seat g % 4 is the learner and the other three seats play `frozen`."""
    games = Tables(n, device)
    learner_seat = torch.arange(n, device=device) % 4
    raws, masks, actions, logps, values, actives, currents, learners = [], [], [], [], [], [], [], []
    while bool((games.winner < 0).any()) and games.turn < MAX_TURNS:
        legal = games.public_legal()
        mask, cards = games.action_mask(legal)
        raw = games.raw()
        learner = games.current == learner_seat if frozen is not None else torch.ones_like(games.current, dtype=torch.bool)
        logits = masked_logits(actor(*features(raw, False)), mask)
        if frozen is not None:  # both networks on every row: fixed shapes
            logits = torch.where(learner.unsqueeze(1), logits, masked_logits(frozen(*features(raw, False)), mask))
        rel_win = F.softmax(critic(*features(raw, True)), 1)
        order = (games.current.unsqueeze(1) + torch.arange(4, device=device)) % 4
        win = torch.zeros_like(rel_win).scatter_(1, order, rel_win)                  # by absolute seat
        logp = F.log_softmax(logits, 1)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-9, 1 - 1e-7)))
        action = (logits + gumbel).argmax(1)
        has_move = mask.any(1)
        card = torch.where(has_move, cards.gather(1, action.unsqueeze(1)).squeeze(1), torch.full_like(action, -1))
        raws.append(tuple(r.cpu() for r in raw)); masks.append(mask.cpu()); actions.append(action.cpu())
        logps.append(logp.gather(1, action.unsqueeze(1)).squeeze(1).cpu())
        values.append(win.cpu()); actives.append((games.winner < 0).cpu()); currents.append(games.current.cpu())
        learners.append(learner.cpu())
        games.step(legal, card)
    if bool((games.winner < 0).any()):
        raise RuntimeError("a game did not finish")
    outcome = F.one_hot(games.winner.cpu(), 4).float()
    T = len(values)
    values, actives = torch.stack(values), torch.stack(actives)
    targets = torch.empty_like(values)
    ret = outcome
    for t in range(T - 1, -1, -1):
        last = ~actives[t + 1] if t + 1 < T else torch.ones(n, dtype=torch.bool)
        nxt = outcome if t + 1 == T else values[t + 1]
        ret = torch.where(last.unsqueeze(1), outcome, (1 - lam) * nxt + lam * ret)
        targets[t] = ret
    cur = torch.stack(currents)
    advantage = (targets - values).gather(2, cur.unsqueeze(2)).squeeze(2)
    keep = actives.reshape(-1)
    raw = tuple(torch.stack([r[i] for r in raws]).flatten(0, 1)[keep] for i in range(6))
    mask = torch.stack(masks).flatten(0, 1)[keep]
    flat = {"raw": raw, "mask": mask, "action": torch.stack(actions).reshape(-1)[keep],
            "logp": torch.stack(logps).reshape(-1)[keep], "advantage": advantage.reshape(-1)[keep],
            "target": targets.flatten(0, 1)[keep], "learner": torch.stack(learners).reshape(-1)[keep]}
    return flat, T


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    actor, critic = SevensNet(False, args.width).to(device), SevensNet(True, args.width).to(device)
    opt = torch.optim.Adam([{"params": actor.parameters(), "lr": args.lr}, {"params": critic.parameters(), "lr": args.lr}])
    iteration, spent = 0, 0.0
    frozen = load_actor(args.frozen, device) if args.frozen else None
    if args.init and not (args.resume and os.path.exists(args.out)):
        saved = torch.load(args.init, map_location=device)
        actor.load_state_dict(saved["actor"]); critic.load_state_dict(saved["critic"])
        opt.load_state_dict(saved["opt"])  # a fresh Adam's first steps would knock the trained nets
    if args.resume and os.path.exists(args.out):
        saved = torch.load(args.out, map_location=device)
        actor.load_state_dict(saved["actor"]); critic.load_state_dict(saved["critic"]); opt.load_state_dict(saved["opt"])
        iteration, spent = saved["iteration"], saved["seconds"]
        torch.manual_seed(args.seed * 1_000_003 + iteration)
    log = open(args.out.replace(".pt", ".log"), "a")
    total = args.hours * 3600
    last = time.time()
    while spent < total:
        progress = spent / total
        lr = args.lr * (1.0 - 0.9 * progress)
        entropy_coef = args.entropy * (1 - progress) + args.entropy_end * progress
        for group in opt.param_groups:
            group["lr"] = lr
        data, turns = rollout(actor, critic, args.games, device, args.lam, frozen)
        size = data["mask"].shape[0]
        stats = torch.zeros(5, device=device)
        batches = 0
        for _ in range(args.epochs):
            # The actor learns only where there is a choice; the critic from as many random states.
            choices = ((data["mask"].sum(1) > 1) & data["learner"]).nonzero().squeeze(1)
            choices = choices[torch.randperm(len(choices))]
            anywhere = torch.randperm(size)
            for start in range(0, len(choices) - args.batch + 1, args.batch):  # full batches only: fixed shapes
                idx = anywhere[start:start + args.batch]
                raw = tuple(r[idx].to(device) for r in data["raw"])
                target = data["target"][idx].to(device)
                order = (raw[4].long().unsqueeze(1) + torch.arange(4, device=device)) % 4
                value_logp = F.log_softmax(critic(*features(raw, True)), 1)
                critic_loss = -(target.gather(1, order) * value_logp).sum(1).mean()
                idx = choices[start:start + args.batch]
                raw = tuple(r[idx].to(device) for r in data["raw"])
                mask, action, old_logp, advantage = (
                    data[key][idx].to(device) for key in ("mask", "action", "logp", "advantage"))
                logp_all = F.log_softmax(masked_logits(actor(*features(raw, False)), mask), 1)
                logp = logp_all.gather(1, action.unsqueeze(1)).squeeze(1)
                ratio = (logp - old_logp).clamp(-10, 10).exp()
                adv = advantage / args.adv_scale
                policy_loss = -torch.min(ratio * adv, ratio.clamp(0.9, 1.1) * adv).mean()
                entropy = -(logp_all.exp() * logp_all.masked_fill(~mask, 0.0)).sum(1).mean()
                loss = policy_loss - entropy_coef * entropy + critic_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    stats += torch.stack([policy_loss, entropy, critic_loss, (old_logp - logp).mean(), adv.std()])
                    batches += 1
        data = None
        if device == "mps":
            torch.mps.empty_cache()
            used = torch.mps.driver_allocated_memory() / 2**30
            if used > args.max_gpu_gb:
                raise RuntimeError(f"GPU memory {used:.1f} GB over the limit")
        iteration += 1
        now = time.time(); spent += min(now - last, 600); last = now  # a sleeping laptop does not count
        if iteration % args.save_every == 0 or spent >= total:
            blob = {"actor": actor.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                    "iteration": iteration, "seconds": spent, "args": vars(args)}
            torch.save(blob, args.out + ".tmp"); os.replace(args.out + ".tmp", args.out)
        if iteration % args.snapshot_every == 0:
            torch.save({"actor": actor.state_dict(), "args": vars(args), "iteration": iteration},
                       args.out.replace(".pt", f"_{iteration}.pt"))
        s = (stats / batches).tolist()
        print(f"{iteration} {spent:.0f}s games {iteration * args.games} turns {turns} states {size} entropy {s[1]:.4f} "
              f"critic {s[2]:.4f} kl {s[3]:.5f} adv {s[4]:.3f} gpu {used if device == 'mps' else 0:.2f}GB", file=log, flush=True)


def load_actor(path, device="cpu"):
    saved = torch.load(path, map_location=device)
    actor = SevensNet(False, saved["args"]["width"]).to(device)
    actor.load_state_dict(saved["actor"])
    return actor.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train"])
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lam", type=float, default=0.9)
    parser.add_argument("--adv-scale", type=float, default=0.05)
    parser.add_argument("--entropy", type=float, default=0.1)
    parser.add_argument("--entropy-end", type=float, default=0.005)
    parser.add_argument("--max-gpu-gb", type=float, default=4.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--snapshot-every", type=int, default=200)
    parser.add_argument("--frozen", default="", help="exploiter mode: one learning seat against three copies of this actor")
    parser.add_argument("--init", default="", help="start actor and critic from this checkpoint")
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
