"use strict";

const RANK_LABELS = { 1: "A", 11: "J", 12: "Q", 13: "K" };
const RANKS = Array.from({ length: 13 }, (_, i) => ({
  rank: i + 1,
  label: RANK_LABELS[i + 1] || String(i + 1),
}));

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

const SEAT_LABELS = ["Seat 0 (bottom)", "Seat 1 (left)", "Seat 2 (top)", "Seat 3 (right)"];

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

function cancelBotTick() {
  if (botTickHandle) {
    clearTimeout(botTickHandle);
    botTickHandle = null;
  }
}

function showSetup() {
  cancelBotTick();
  document.getElementById("setup-overlay").classList.remove("hidden");
  document.getElementById("game-area").classList.add("hidden");
  document.getElementById("winner-banner").classList.add("hidden");
}

async function onStart() {
  const botSeats = Array.from(document.querySelectorAll(".seat-toggle input[type=checkbox]:checked"))
    .map((box) => parseInt(box.dataset.seat, 10));

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

// Pass when cardLabel is null.
async function submitPlay(cardLabel) {
  if (!lastSnapshot || lastSnapshot.active_is_bot) return;
  const player = lastSnapshot.current_player;
  applyResponse(await postJSON("/api/play", { player, card: cardLabel }));
}

function onPass() {
  return submitPlay(null);
}

async function botStep() {
  botTickHandle = null;
  if (!lastSnapshot || !lastSnapshot.active_is_bot || lastSnapshot.winner !== null) {
    return;
  }
  applyResponse(await postJSON("/api/bot_step", {}));
}

function applyResponse(snapshot) {
  if (snapshot.error) showToast(snapshot.error);
  else applySnapshot(snapshot);
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

  cancelBotTick();
  if (snapshot.winner === null && snapshot.active_is_bot) {
    botTickHandle = setTimeout(botStep, BOT_DELAY_MS);
  }
}

function renderSeats(snapshot) {
  snapshot.seats.forEach((seat) => {
    const el = document.getElementById(`seat-${seat.index}`);
    el.classList.toggle("active", seat.is_active);
    el.classList.toggle("winner", seat.is_winner);

    const badges = [
      seat.is_bot ? '<span class="badge bot">Bot</span>' : '<span class="badge human">Human</span>',
    ];
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
      if (isLegal) submitPlay(cardLabel);
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
    ${reasons ? `<ul>${reasons}</ul>` : ""}
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
  return { suit: label.slice(-1), rankLabel: label.slice(0, -1) };
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
