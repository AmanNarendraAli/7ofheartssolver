# Game Simulation Workplan

## Goal

A visual, browser-based game simulator for 7 of Hearts that wraps the existing
`seven_hearts` engine. The user can sit at any subset of the four seats and play
against perfect-play bots in the empty seats.

Visual style is loosely inspired by pokernow.club: a felt-green oval table with
seats around the perimeter, the four suit chains laid out in the middle, and
the active human player's hand at the bottom of the screen.

## Requirements (from the user)

1. Toggle the number of bot opponents between 0 and 4. Any non-bot seat is
   playable by the human at the keyboard (hot seat).
2. Bots play "perfectly": they use oracle simulation with the tuned strategy
   weights — i.e. `choose_oracle_move` with `DEFAULT_WEIGHTS`, which has full
   hand information inside the simulator and the one-ply response adjustment
   already baked in.
3. The human gets a move recommendation each turn but is free to play any
   legal move (or pass when there are no legal moves).

## Architecture

Single-process Flask app with one in-memory game at a time.

- `simulator.py` — Flask routes + a `GameSession` wrapper around the engine.
- `templates/index.html` — Single page UI.
- `static/style.css` — Felt-table styling, card visuals.
- `static/app.js` — Fetches state, renders the table, dispatches moves,
  drives bot turns on a timer.

The browser polls a snapshot endpoint, renders, and (when it's a bot's turn
and the game isn't over) calls a bot-step endpoint after a short delay so the
user can watch the action.

## API

- `POST /api/new_game` — Body: `{ "bot_seats": [0,2,3], "seed": 7 }`. Deals 13
  cards each, finds the holder of `7H`, returns the snapshot.
- `GET  /api/state` — Returns the current snapshot.
- `POST /api/play` — Body: `{ "player": 0, "card": "6H" }` or
  `{ "player": 0, "card": null }` to pass. Validates against the engine's rules.
- `POST /api/bot_step` — Advances one move if the current player is a bot.

### Snapshot shape

```json
{
  "table": { "H": {"low": 7, "high": 9}, "S": null, "D": null, "C": null },
  "current_player": 2,
  "seats": [
    { "index": 0, "is_bot": false, "card_count": 12, "passed_last": false }
  ],
  "active_player": 2,
  "active_is_bot": true,
  "active_hand": null,
  "active_legal_moves": [],
  "active_must_pass": false,
  "recommendation": null,
  "winner": null,
  "log": ["P3 played 7H", "P0 played 6H"]
}
```

`active_hand`, `active_legal_moves`, and `recommendation` are only populated
when the active player is human, so private information stays private.

## Engine glue

- Bot moves: `choose_oracle_move(state, hands, player, DEFAULT_WEIGHTS)`. This
  is full-information perfect play inside the simulator (the user explicitly
  asked for "oracle simulation with optimized strategy"). It already has the
  one-ply response penalty applied.
- Human recommendation: `recommend_move(state, knowledge, DEFAULT_WEIGHTS)`.
  The heuristic is fast enough to compute every turn and exposes its score
  reasons for display.
- Move application: `apply_known_play(state, hands, player, card_or_none)`,
  which updates both the public state and the full-information hands dict.
- Game end: stop at first-out (matches the project's strategic objective);
  show "P<x> won" and offer a New Game button.

## Visuals

- Oval green felt with the four seats positioned bottom / left / top / right
  for seats 0 / 1 / 2 / 3.
- Each seat shows the player label, a Bot/Human badge, the card count, and an
  optional "passed" badge after a pass move. The active seat gets a glow.
- The middle of the table holds four suit rows. Each row is a horizontal
  strip with rank slots A–K. Played ranks are filled in as cards; empty
  slots are dim. Hearts/Diamonds are red; Clubs/Spades are black.
- The active human's hand is rendered face-up at the bottom. Cards that are
  in the legal-moves set are clickable and visually elevated; the
  recommended card gets a green border. Other cards are dimmed but still
  clickable (the user is free to attempt a move; the server will reject
  illegal plays with a toast).
- A Pass button appears when the legal-moves set is empty.
- A move log scrolls on the right.

## Implementation order

1. `simulator.py` with `GameSession`, snapshot builder, and Flask routes.
2. `templates/index.html` with placeholders and the script include.
3. `static/style.css` with the table layout and card design.
4. `static/app.js` with fetch + render + click handlers + bot tick.
5. Manual smoke test: launch, start a 1-human-vs-3-bots game, click through.

## Out of scope

- Networked multiplayer / multiple concurrent games.
- Persisted state across restarts.
- Animation polish beyond a basic fade and the bot-step delay.
- Showing the Monte Carlo recommendation alongside the heuristic
  (the heuristic is fast; Monte Carlo per-turn would lag the UI).
