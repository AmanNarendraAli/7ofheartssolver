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
- The value of retaining control over unopened suits or blocked chains.
- Current hand counts and endgame urgency.
- Information implied by opponent passes.

Later versions may add:

- Probabilistic opponent hand modeling.
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
- A `PlayerKnowledge` model containing the solver's player index and private hand.
- Calculation of unseen cards as `full_deck - own_hand - cards_on_table`.
- Basic opponent inference from passes.
- Heuristic move scoring and recommendation through `recommend_move`.

Player hand counts are optional. The solver can run with:

```python
GameState(hand_counts=None)
```

or with partial information:

```python
GameState(hand_counts=(4, None, 7, None))
```

Known counts are used for endgame urgency. Unknown counts are treated neutrally.

## Current Heuristic

The current next-move scorer starts each legal move with a base reward for playing one card.

It then adjusts the score based on:

- Own cards newly unlocked by the move.
- Public cards newly opened that opponents may hold.
- Blocking value lost by opening a suit or giving up a bottleneck.
- Endgame urgency when known hand counts show that the solver or an opponent is close to going out.

This is intentionally a first-pass heuristic. It is designed to be inspectable and tunable before adding deeper search.

## Current Opponent Model

The opponent model starts by assuming each unseen card could belong to any opponent.

It then walks through public move history:

- If an opponent passed, cards that were legal at that moment are removed from that opponent's possible cards.
- If an opponent played a card, that card is removed from the other opponents' possible cards.

This captures important negative information from passes, but it does not yet enforce exact hidden-hand distributions or full hand-count-consistent probabilities.

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
7 tests passed
```
