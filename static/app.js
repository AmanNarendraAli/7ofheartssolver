"use strict";

const RANKS = [
  { rank: 1, label: "A" },
  { rank: 2, label: "2" },
  { rank: 3, label: "3" },
  { rank: 4, label: "4" },
  { rank: 5, label: "5" },
  { rank: 6, label: "6" },
  { rank: 7, label: "7" },
  { rank: 8, label: "8" },
  { rank: 9, label: "9" },
  { rank: 10, label: "10" },
  { rank: 11, label: "J" },
  { rank: 12, label: "Q" },
  { rank: 13, label: "K" },
];

const SUITS = ["H", "S", "D", "C"];

const SUIT_GLYPH = {
  H: "♥",
  D: "♦",
  S: "♠",
  C: "♣",
};

const SUIT_NAME = {
  H: "Hearts",
  D: "Diamonds",
  S: "Spades",
  C: "Clubs",
};

const SEAT_LABELS = {
  0: "Seat 0 (bottom)",
  1: "Seat 1 (left)",
  2: "Seat 2 (top)",
  3: "Seat 3 (right)",
};

const BOT_DELAY_MS = 850;

let lastSnapshot = null;
let botTickHandle = null;
let toastHandle = null;

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("start-btn").addEventListener("click", onStart);
  document.getElementById("pass-btn").addEventListener("click", onPass);
  document.getElementById("new-game-btn").addEventListener("click", showSetup);
  document.getElementById("log-toggle").addEventListener("click", onToggleLog);
});

function onToggleLog() {
  const area = document.getElementById("log-area");
  const btn = document.getElementById("log-toggle");
  const collapsed = area.classList.toggle("collapsed");
  btn.setAttribute("aria-expanded", String(!collapsed));
}

function showSetup() {
  if (botTickHandle) {
    clearTimeout(botTickHandle);
    botTickHandle = null;
  }
  document.getElementById("setup-overlay").classList.remove("hidden");
  document.getElementById("game-area").classList.add("hidden");
  document.getElementById("winner-banner").classList.add("hidden");
}

async function onStart() {
  const botSeats = [];
  document
    .querySelectorAll(".seat-toggle input[type=checkbox]")
    .forEach((box) => {
      if (box.checked) {
        botSeats.push(parseInt(box.dataset.seat, 10));
      }
    });

  const seedValue = document.getElementById("seed-input").value.trim();
  const body = { bot_seats: botSeats };
  if (seedValue !== "") {
    body.seed = parseInt(seedValue, 10);
  }

  const snapshot = await postJSON("/api/new_game", body);
  if (snapshot.error) {
    showToast(snapshot.error);
    return;
  }

  document.getElementById("setup-overlay").classList.add("hidden");
  document.getElementById("game-area").classList.remove("hidden");
  applySnapshot(snapshot);
}

async function onPass() {
  if (!lastSnapshot || lastSnapshot.active_is_bot) return;
  const player = lastSnapshot.current_player;
  const snapshot = await postJSON("/api/play", { player, card: null });
  if (snapshot.error) {
    showToast(snapshot.error);
    return;
  }
  applySnapshot(snapshot);
}

async function onCardClick(cardLabel) {
  if (!lastSnapshot || lastSnapshot.active_is_bot) return;
  const player = lastSnapshot.current_player;
  const snapshot = await postJSON("/api/play", { player, card: cardLabel });
  if (snapshot.error) {
    showToast(snapshot.error);
    return;
  }
  applySnapshot(snapshot);
}

async function botStep() {
  botTickHandle = null;
  if (!lastSnapshot || !lastSnapshot.active_is_bot || lastSnapshot.winner !== null) {
    return;
  }
  const snapshot = await postJSON("/api/bot_step", {});
  if (snapshot.error) {
    showToast(snapshot.error);
    return;
  }
  applySnapshot(snapshot);
}

function applySnapshot(snapshot) {
  if (!snapshot || !snapshot.active) return;
  lastSnapshot = snapshot;
  renderSeats(snapshot);
  renderBoard(snapshot);
  renderHand(snapshot);
  renderRecommendation(snapshot);
  renderLog(snapshot);
  renderWinner(snapshot);

  if (botTickHandle) {
    clearTimeout(botTickHandle);
    botTickHandle = null;
  }
  if (snapshot.winner === null && snapshot.active_is_bot) {
    botTickHandle = setTimeout(botStep, BOT_DELAY_MS);
  }
}

function renderSeats(snapshot) {
  snapshot.seats.forEach((seat) => {
    const el = document.getElementById(`seat-${seat.index}`);
    el.classList.toggle("active", seat.is_active);
    el.classList.toggle("winner", seat.is_winner);

    const role = seat.is_bot ? "Bot" : "Human";
    const badges = [];
    badges.push(
      `<span class="badge ${seat.is_bot ? "bot" : "human"}">${role}</span>`,
    );
    if (seat.is_winner) {
      badges.push('<span class="badge winner">Winner</span>');
    } else if (seat.passed_last) {
      badges.push('<span class="badge passed">Passed</span>');
    }

    el.innerHTML = `
      <div class="avatar">P${seat.index}</div>
      <div class="info">
        <div class="name">${SEAT_LABELS[seat.index]}</div>
        <div class="meta">${badges.join(" ")}</div>
      </div>
      ${cardStackHTML(seat.card_count)}
    `;
  });
}

function cardStackHTML(count) {
  if (count <= 0) return '<div class="card-stack"></div>';
  return `
    <div class="card-stack">
      <div class="stack-card"></div>
      <div class="stack-card"></div>
      <div class="stack-card"></div>
      <div class="count">${count}</div>
    </div>
  `;
}

function renderBoard(snapshot) {
  SUITS.forEach((suit) => {
    const row = document.querySelector(`.suit-row[data-suit="${suit}"] .rank-row`);
    row.innerHTML = "";
    const run = snapshot.table[suit];
    RANKS.forEach(({ rank, label }) => {
      const slot = document.createElement("div");
      slot.className = "rank-slot";
      if (rank === 7) slot.classList.add("seven");
      const isFilled = run && rank >= run.low && rank <= run.high;
      if (isFilled) {
        slot.classList.add("filled");
        if (suit === "H" || suit === "D") slot.classList.add("red");
      }
      slot.innerHTML = `
        <span class="slot-rank">${label}</span>
        <span class="slot-suit">${SUIT_GLYPH[suit]}</span>
      `;
      row.appendChild(slot);
    });
  });
}

function renderHand(snapshot) {
  const handEl = document.getElementById("hand");
  const passBtn = document.getElementById("pass-btn");
  const activeLabel = document.getElementById("active-label");
  handEl.innerHTML = "";

  if (snapshot.winner !== null) {
    activeLabel.textContent = `Game over — P${snapshot.winner} won.`;
    passBtn.disabled = true;
    return;
  }

  if (snapshot.active_is_bot) {
    activeLabel.textContent = `Bot P${snapshot.current_player} is thinking…`;
    passBtn.disabled = true;
    return;
  }

  activeLabel.textContent = `Your turn — P${snapshot.current_player}`;
  const hand = snapshot.active_hand || [];
  const legalSet = new Set(snapshot.active_legal_moves || []);
  const recommended = snapshot.recommendation ? snapshot.recommendation.card : null;

  hand.forEach((cardLabel) => {
    const card = parseCardLabel(cardLabel);
    const div = document.createElement("div");
    const isLegal = legalSet.has(cardLabel);
    div.className = "card";
    if (card.suit === "H" || card.suit === "D") div.classList.add("red");
    div.classList.add(isLegal ? "legal" : "illegal");
    if (cardLabel === recommended) div.classList.add("recommended");

    const glyph = SUIT_GLYPH[card.suit];
    div.innerHTML = `
      <span class="corner top-left">
        <span class="corner-rank">${card.rankLabel}</span>
        <span class="corner-suit">${glyph}</span>
      </span>
      <span class="center-suit">${glyph}</span>
      <span class="corner bottom-right">
        <span class="corner-rank">${card.rankLabel}</span>
        <span class="corner-suit">${glyph}</span>
      </span>
    `;
    div.title = isLegal
      ? `Play ${card.rankLabel} of ${SUIT_NAME[card.suit]}`
      : `Not currently legal`;
    div.addEventListener("click", () => {
      if (isLegal) onCardClick(cardLabel);
      else showToast("That card is not currently legal.");
    });
    handEl.appendChild(div);
  });

  passBtn.disabled = !snapshot.active_must_pass;
}

function renderRecommendation(snapshot) {
  const el = document.getElementById("recommendation");
  if (snapshot.winner !== null || snapshot.active_is_bot || !snapshot.recommendation) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }

  const rec = snapshot.recommendation;
  const card = parseCardLabel(rec.card);
  const reasons = (rec.reasons || []).map((r) => `<li>${escapeHtml(r)}</li>`).join("");
  el.innerHTML = `
    <div>Recommended: <strong>${card.rankLabel}${SUIT_GLYPH[card.suit]}</strong>
    (heuristic score ${rec.score.toFixed(2)})</div>
    ${reasons ? `<ul style="margin: 4px 0 0 20px; padding: 0;">${reasons}</ul>` : ""}
  `;
  el.classList.remove("hidden");
}

function renderLog(snapshot) {
  const log = document.getElementById("log");
  log.innerHTML = "";
  (snapshot.log || []).forEach((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    log.appendChild(li);
  });
  log.scrollTop = log.scrollHeight;
}

function renderWinner(snapshot) {
  const banner = document.getElementById("winner-banner");
  if (snapshot.winner !== null) {
    banner.textContent = `P${snapshot.winner} is out — first-out winner!`;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

function parseCardLabel(label) {
  const suit = label.slice(-1);
  const rankText = label.slice(0, -1);
  return { suit, rankLabel: rankText };
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      if (data.error) message = data.error;
    } catch (_) { /* ignore */ }
    return { error: message };
  }
  return response.json();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  if (toastHandle) clearTimeout(toastHandle);
  toastHandle = setTimeout(() => toast.classList.add("hidden"), 3000);
}
