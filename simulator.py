"""Flask-based visual game simulator for 7 of Hearts.

Run with: ``py simulator.py`` and open http://127.0.0.1:5000/.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import Iterable

from flask import Flask, jsonify, render_template, request

from seven_hearts import (
    Card,
    DEFAULT_WEIGHTS,
    GameState,
    PlayerKnowledge,
    Suit,
    apply_known_play,
    choose_oracle_move,
    deal_random_hands,
    first_empty_player,
    initial_state_for_hands,
    recommend_move,
)


SUIT_ORDER = (Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS)


@dataclass
class GameSession:
    lock: threading.Lock
    hands: dict[int, set[Card]] | None = None
    state: GameState | None = None
    seats_are_bot: list[bool] | None = None
    winner: int | None = None
    move_log: list[str] | None = None
    last_pass: list[bool] | None = None

    def new_game(self, bot_seats: Iterable[int], seed: int | None) -> dict:
        with self.lock:
            rng = random.Random(seed) if seed is not None else random.Random()
            self.hands = deal_random_hands(rng)
            self.state = initial_state_for_hands(self.hands)
            self.seats_are_bot = [seat in set(bot_seats) for seat in range(4)]
            self.winner = None
            self.move_log = [f"P{self.state.current_player} holds 7H and acts first."]
            self.last_pass = [False, False, False, False]
            return self._snapshot_locked()

    def play(self, player: int, card_label: str | None) -> dict:
        with self.lock:
            self._require_active()
            assert self.state is not None and self.hands is not None
            if self.state.current_player != player:
                raise ValueError(
                    f"not P{player}'s turn (current player is P{self.state.current_player})"
                )

            hand = self.hands[player]
            card = Card.parse(card_label) if card_label else None
            self.state.validate_turn(hand, card)
            assert self.last_pass is not None and self.move_log is not None

            self.state, self.hands = apply_known_play(self.state, self.hands, player, card)
            if card is None:
                self.last_pass[player] = True
                self.move_log.append(f"P{player} passed.")
            else:
                self.last_pass[player] = False
                self.move_log.append(f"P{player} played {card.label()}.")

            self._check_winner_locked()
            return self._snapshot_locked()

    def bot_step(self) -> dict:
        with self.lock:
            self._require_active()
            assert self.state is not None and self.hands is not None
            assert self.seats_are_bot is not None

            if self.winner is not None:
                return self._snapshot_locked()

            player = self.state.current_player
            if not self.seats_are_bot[player]:
                return self._snapshot_locked()

            legal = self.state.legal_moves(self.hands[player])
            if not legal:
                card = None
            else:
                card = choose_oracle_move(
                    self.state, self.hands, player, DEFAULT_WEIGHTS
                )

            assert self.move_log is not None and self.last_pass is not None
            self.state, self.hands = apply_known_play(self.state, self.hands, player, card)
            if card is None:
                self.last_pass[player] = True
                self.move_log.append(f"P{player} (bot) passed.")
            else:
                self.last_pass[player] = False
                self.move_log.append(f"P{player} (bot) played {card.label()}.")

            self._check_winner_locked()
            return self._snapshot_locked()

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot_locked()

    def _check_winner_locked(self) -> None:
        assert self.hands is not None
        if self.winner is not None:
            return
        first_out = first_empty_player(self.hands)
        if first_out is not None:
            self.winner = first_out
            assert self.move_log is not None
            self.move_log.append(f"P{first_out} is out — first-out winner.")

    def _require_active(self) -> None:
        if self.state is None or self.hands is None or self.seats_are_bot is None:
            raise ValueError("no active game; call /api/new_game first")

    def _snapshot_locked(self) -> dict:
        if self.state is None or self.hands is None or self.seats_are_bot is None:
            return {"active": False}

        assert self.last_pass is not None and self.move_log is not None
        active_player = self.state.current_player
        active_is_bot = self.seats_are_bot[active_player]

        legal_moves: list[str] = []
        active_hand: list[str] | None = None
        recommendation: dict | None = None
        active_must_pass = False

        if self.winner is None and not active_is_bot:
            hand = sorted(self.hands[active_player])
            active_hand = [card.label() for card in hand]
            legal = self.state.legal_moves(self.hands[active_player])
            legal_moves = [card.label() for card in legal]
            active_must_pass = len(legal) == 0
            if not active_must_pass:
                knowledge = PlayerKnowledge(
                    player=active_player,
                    hand=frozenset(self.hands[active_player]),
                )
                rec = recommend_move(self.state, knowledge, DEFAULT_WEIGHTS)
                if rec is not None:
                    recommendation = {
                        "card": rec.card.label(),
                        "score": round(rec.score, 2),
                        "reasons": list(rec.reasons),
                    }

        seats = [
            {
                "index": seat,
                "is_bot": self.seats_are_bot[seat],
                "card_count": len(self.hands[seat]),
                "passed_last": self.last_pass[seat],
                "is_active": seat == active_player and self.winner is None,
                "is_winner": seat == self.winner,
            }
            for seat in range(4)
        ]

        table = {}
        for suit in SUIT_ORDER:
            run = self.state.table[suit]
            if run.is_open:
                table[suit.value] = {"low": run.low, "high": run.high}
            else:
                table[suit.value] = None

        return {
            "active": True,
            "table": table,
            "current_player": active_player,
            "active_is_bot": active_is_bot,
            "active_hand": active_hand,
            "active_legal_moves": legal_moves,
            "active_must_pass": active_must_pass,
            "recommendation": recommendation,
            "seats": seats,
            "winner": self.winner,
            "log": list(self.move_log[-30:]),
        }


SESSION = GameSession(lock=threading.Lock())


app = Flask(__name__)


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/new_game")
def api_new_game():
    payload = request.get_json(silent=True) or {}
    bot_seats = payload.get("bot_seats", [1, 2, 3])
    seed = payload.get("seed")
    try:
        bot_seats = [int(s) for s in bot_seats]
    except (TypeError, ValueError):
        return jsonify({"error": "bot_seats must be a list of seat indices 0..3"}), 400
    if any(seat < 0 or seat > 3 for seat in bot_seats):
        return jsonify({"error": "seat indices must be 0..3"}), 400
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return jsonify({"error": "seed must be an integer"}), 400
    snapshot = SESSION.new_game(bot_seats, seed)
    return jsonify(snapshot)


@app.get("/api/state")
def api_state():
    return jsonify(SESSION.snapshot())


@app.post("/api/play")
def api_play():
    payload = request.get_json(silent=True) or {}
    if "player" not in payload:
        return jsonify({"error": "missing 'player'"}), 400
    try:
        player = int(payload["player"])
    except (TypeError, ValueError):
        return jsonify({"error": "player must be an integer 0..3"}), 400
    card = payload.get("card")
    try:
        snapshot = SESSION.play(player, card)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(snapshot)


@app.post("/api/bot_step")
def api_bot_step():
    try:
        snapshot = SESSION.bot_step()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(snapshot)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
