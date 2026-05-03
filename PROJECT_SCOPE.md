# 7 of Hearts Solver Project

## Game Rules

7 of Hearts, sometimes called Sevens, is played with a standard 52-card deck by 4 players.

Each player is dealt 13 cards. The objective is to be the first player to get rid of all cards in hand.

The game starts with the player who holds the 7 of hearts. That player must play the 7 of hearts as the opening move.

Cards are played into suit sequences. Each suit begins with its 7. After a suit has been opened, cards in that suit may only be played if they are numerically adjacent to cards already on the table.

For example, if the 7 of hearts is on the table, the legal heart plays are:

- 6 of hearts
- 8 of hearts

If the 6 of hearts is then played, the legal heart plays become:

- 5 of hearts
- 8 of hearts

The same structure applies to every suit. A suit cannot be played at all until its 7 has been played.

Aces are low. Rank order is:

```text
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
```

On a player's turn:

- If the player has at least one legal move, they must play one legal card.
- If the player has no legal move, their turn is skipped.

Play proceeds clockwise. The game continues until all players except one have emptied their hands, but the strategic goal for this project is to choose moves that maximize the chance of being the first player out.

## Strategic Ideas

The game is centered around controlling which future moves become available.

Important strategic patterns include:

- Playing a card may open future moves for yourself.
- Playing a card may also open future moves for opponents.
- Holding a 7 can block an entire suit from being played.
- Holding a bottleneck card, such as a 6 or 8 next to an opened 7, can delay one side of a suit.
- End cards, especially aces and kings, may become liabilities unless the chain toward them is likely to open.

For example, if a player only holds the 7 of spades in spades, they may want to delay playing it because no other spades can be played until that 7 appears. However, if the same player also holds the ace of spades, opening spades earlier may help create enough turns for the lower spade sequence to reach the ace.

## Algorithmic Design Principles

The solver should prefer clean causal logic over special-case strategic moods.

The guiding assumption is that if the game has already been played well, the current position should be evaluated by the same principles throughout the game:

- What cards become playable?
- Who is likely to hold those cards?
- Who gets to act soonest?
- How much future chain control is being released?
- How much runway does the solver need for its own blocked cards?

Endgame situations are important, but they should not automatically create a separate scoring mode. The optimal move should emerge from the same card-specific factors that matter earlier in the game. For example, opening a move for the next opponent is dangerous because of turn order and likely ownership, not because an abstract "endgame urgency" bonus says so.

This means:

- Avoid adding flat score terms that apply equally to every legal move, because they cancel out in the recommendation.
- Avoid panic-style urgency knobs unless they modulate card-specific effects and are supported by testing.
- Prefer simple, inspectable heuristics over complicated weighting that does not match how the game is reasoned about.
- Treat longer released chains as more dangerous because they expose more future cards, but avoid double-counting chain length without evidence.
- Keep hand counts as useful information, especially for hidden-card inference and diagnostics, but do not let them override the direct move logic unless simulation or self-play supports it.

In short, the solver should reason inductively from the current position: if prior moves were chosen well, the next move should be chosen by the same structural principles, not by switching to a different endgame personality.

## Project Scope

The goal of this project is to develop an algorithmic strategic solution for 7 of Hearts, starting with a next-move generator.

The solver should operate under realistic information, not full information.

The solver may know:

- Its own hand.
- Cards already played to the table.
- The current open interval for each suit.
- Whose turn it is.
- Player hand counts, if available.
- Which players have passed on previous turns.
- Publicly observable move history.

The solver should not know:

- The exact cards in opponents' hands.
- The exact order of the remaining unseen cards.

The solver should reason probabilistically about unseen cards using the information available from the current game state and move history.

In this project, unseen cards means cards that are not in the solver's hand and not already on the table. Because the full deck is dealt at the start, every unseen card must be in one of the opponents' hands.

## Initial Solver Direction

The first version should generate all legal moves for the current player and score them with a heuristic.

The heuristic should consider:

- Immediate hand reduction.
- Whether the move unlocks future cards in the solver's own hand.
- Whether the move likely unlocks cards for opponents.
- The value of retaining control over unopened suits and future suit chains.
- Current hand counts, where available, for opponent-holder estimates and diagnostics.
- Information implied by opponent passes.

Later versions may add:

- Hand-count-consistent hidden deal sampling.
- Monte Carlo simulation over possible hidden deals.
- Multi-turn search.
- Self-play for tuning heuristic weights.

## Current Implementation

The first working implementation lives in `seven_hearts.py`.

It currently includes:

- A `Card` model with parsing support for labels such as `7H`, `AS`, and `10D`.
- A `SuitRun` model that represents each suit as a continuous interval growing outward from 7.
- A `GameState` model for public table state, turn state, optional hand counts, and move history.
- Legal move generation by intersecting the public legal cards with the solver's hand.
- Enforcement of the opening rule: the first public legal card is only `7H`.
- Optional turn validation, including rejection of illegal passes when legal moves exist.
- A `PlayerKnowledge` model containing the solver's player index and private hand.
- Calculation of unseen cards as `full_deck - own_hand - cards_on_table`.
- Basic opponent inference from passes.
- Heuristic move scoring and recommendation through `recommend_move`.
- Immutable structured score components alongside human-readable score reasons.

Player hand counts are optional. The solver can run with:

```python
GameState(hand_counts=None)
```

or with partial information:

```python
GameState(hand_counts=(4, None, 7, None))
```

Known counts are used as a rough signal in opponent-holder estimates and for diagnostic urgency calculations. Unknown counts are treated neutrally.

## Current Heuristic

The current next-move scorer compares only decision-relevant score components. Earlier versions used a cosmetic base reward for playing a card, but that cancelled out across all legal moves and has been removed.

The scorer adjusts each legal move based on:

- Own cards newly unlocked by the move.
- Public cards newly opened that opponents may hold, weighted by likely holder and turn order.
- Future chain impact for the side or sides of the suit released by the move.
- Time-to-playable pressure for the solver's own cards.

Endgame urgency is not currently added to the move score. Since urgency depends on the state rather than on the specific legal card, adding it directly would cancel out in the move ranking. It remains available as a diagnostic helper, but the active scorer relies on card-specific factors.

This is intentionally a first-pass heuristic. It is designed to be inspectable and tunable before adding deeper search.

Future chain impact is evaluated for every move, not only for playing 7s.

For example:

- Playing `8S` releases the high-side chain `9S, 10S, JS, QS, KS`.
- Playing `10S` only releases `JS, QS, KS`.
- Therefore, playing `8S` gives up more future control than playing `10S`, all else equal.

When a 7 is played, both sides of the suit are evaluated:

- Low side: `6, 5, 4, 3, 2, A`
- High side: `8, 9, 10, J, Q, K`

The future chain impact score considers:

- How many future cards in the released chain are unseen and therefore likely held by opponents.
- How many future cards in the released chain are owned by the solver.
- Whether the solver owns the immediate gate card, such as `6S` or `8S`.
- Whether the solver owns a distant tail card, such as `AS` or `KS`, and that specific tail card's own time-to-playable runway.
- The turn-order-weighted opponent mass in the future released chain.

The future chain danger is intentionally kept simple: opening six future cards is worse than opening four future cards. The model does not currently apply position decay within the chain, because immediate unlock risk and turn order are already handled separately and the simpler count-based shape is easier to inspect.

This means the model now distinguishes cases such as:

- `7S` alone: expensive to open, because it releases a whole suit without much personal benefit.
- `7S + AS`: still risky, but opening the suit gets some credit because the ace needs the low-side chain to progress.
- `7S + 6S + 8S`: much better, because the solver owns both immediate gates and can play at effectively lower cost.
- `8S` versus `10S`: `8S` gives up more future control because it releases a longer chain.

## Current Opponent Model

The opponent model starts by assuming each unseen card could belong to any opponent, then narrows those possibility sets using public history.

It then walks through public move history:

- If an opponent passed, cards that were legal at that moment are removed from that opponent's possible cards.
- If an opponent played a card, that card is removed from the other opponents' possible cards.

The model now exposes per-holder marginal estimates through `holder_probability(player, card)` rather than using the earlier misnamed "probability any opponent has this card" idea. Since every unseen card must be held by some opponent, the useful question is which opponent is likely to hold it.

Unlock risk is turn-order-aware:

- Cards likely held by the next opponent are riskier.
- Cards likely held by later opponents are discounted.
- Multiple newly legal cards are capped per opponent turn so opening `6S` and `8S` is not treated as if both can be played by the same opponent immediately.

Known hand counts are used as a rough weighting signal when available, but the model still does not fully enforce exact hidden-hand distributions.

For every unseen card that has at least one possible holder, holder probabilities are normalized so the per-opponent marginals sum to 1. The model also exposes `consistency_errors()` to flag impossible states, such as a known opponent hand count being larger than that opponent's remaining possible card set.

## Time-To-Playable Logic

The solver now has a `time_to_playable` estimate.

This counts how many chain steps are needed before a card can become playable from the current table state.

Examples:

- If spades is open only at `7S`, then `8S` is 1 step away.
- If spades is open only at `7S`, then `AS` is 6 steps away.
- If a suit is unopened, distance is measured from 7.

This helps avoid treating a distant tail card as automatically valuable. A tail like `AS` or `KS` gets credit only in relation to the runway required to reach it.

Tail-card runway is measured using the tail card itself, not the average distance of all owned cards on that side. For example, owning both `8S` and `KS` should not make the `KS` look close just because `8S` is close.

## Validation

`GameState.public_legal_cards()` now enforces the real opening rule:

```text
Empty table -> only 7H is legal
After 7H -> unopened suits may be opened with their 7s
```

`GameState.after_play(...)` remains usable as a low-level replay helper, but it can validate a turn when a hand is supplied:

```python
state.after_play(player, card, hand=my_hand)
```

Validation rejects:

- playing a card that is not in hand
- playing a card that is not legal
- passing while legal moves are available

## Remaining Modeling Work

The following items are intentionally documented as future work:

- **Hand-count-consistent hidden deal sampling.** Current holder probabilities are approximate. A stronger model should sample or enumerate hidden deals where each unseen card is assigned to exactly one opponent and known hand counts are satisfied.
- **Monte Carlo and multi-turn search.** The current solver is still a one-move heuristic. Search over sampled hidden deals would better estimate who benefits from a move over several turns.
- **Self-play and weight tuning.** Constants are still heuristic. A self-play harness should compare strategies across many deals and provide an objective for tuning weights.
- **Endgame policy testing.** Endgame urgency is currently diagnostic only. If self-play shows that late-game move preferences should change, urgency should modulate card-specific components rather than being added as a flat score.

## Demo and Tests

`demo.py` contains a small example scenario that prints:

- Legal moves.
- Score for each legal move.
- Reasons behind each score.
- The recommended move.

`test_seven_hearts.py` contains focused tests for rules and inference behavior.

`run_tests.py` is a tiny no-dependency test runner used because `pytest` was slow or unresponsive in the current environment.

Current verification:

```text
py run_tests.py
23 tests passed
```

Current demo command:

```text
py demo.py
```
