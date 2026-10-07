// The page: load a model, check it against PyTorch, write stories with it, and
// show what the model considered for every word.

import { StorySession } from "./generate.js";
import { TinyGPT } from "./model.js";
import { runSelfTest } from "./selftest.js";
import { Tokenizer } from "./tokenizer.js";

// ---------------------------------------------------------------- settings
const REPO_URL = ""; // e.g. "https://github.com/your-name/tiny-lm"; the footer link appears once this is set
const AUTHOR = { name: "Kritika Garg", url: "https://kritikaatech.vercel.app/" };

const MAX_TOKENS = 300; // longest story the page will write
const TOP_K = 40; // only the 40 most likely tokens can ever be picked
const KEEP = 8; // how many candidates the bars show
const SPEED_DELAY = { slow: 260, normal: 70, fast: 0 }; // milliseconds between tokens

// ---------------------------------------------------------------- elements and state
const $ = (id) => document.getElementById(id);
const els = {
  empty: $("empty"), failed: $("failed"), failedText: $("failed-text"), app: $("app"),
  model: $("model"), badge: $("badge"), facts: $("facts"),
  prompt: $("prompt"), primary: $("primary"), story: $("story"),
  boundaries: $("boundaries"), count: $("count"),
  title: $("inspector-title"), text: $("inspector-text"), bars: $("bars"), hint: $("inspector-hint"),
  step: $("step"), back: $("back"),
  temperature: $("temperature"), temperatureValue: $("temperature-value"),
  howModel: $("how-model"), howCheck: $("how-check"), credit: $("credit"), repo: $("repo"),
};

let model = null;
let tokenizer = null;
let session = null; // the story being written (see StorySession in generate.js)
let mode = "idle"; // idle | writing | paused | finished
let selected = null; // index of the written token being inspected, or null for "the next word"
let run = 0; // bumped to stop a writing loop that is in flight
let loadTicket = 0; // guards against a slow model load finishing after a newer one
let burst = { start: 0, count: 0 }; // for the tokens-per-second figure

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const temperature = () => Number(els.temperature.value);
const speed = () => document.querySelector('input[name="speed"]:checked').value;
const promptUnchanged = () => session !== null && els.prompt.value === session.prompt;

// Everything that changes the story runs through this queue, one at a time, so
// a click can never land in the middle of another step.
let queue = Promise.resolve();
function enqueue(task) {
  queue = queue.then(task).catch((error) => {
    console.error(error);
    run++;
    setMode(session ? "paused" : "idle");
    els.hint.textContent = `Something went wrong: ${error.message}`;
  });
  return queue;
}

// ---------------------------------------------------------------- wording helpers
function percent(probability) {
  if (probability > 0 && probability < 0.005) return "<1%";
  return `${Math.round(probability * 100)}%`;
}

/** How to show a token to a person: tokens are words, pieces of words, or special marks. */
function describe(id) {
  const text = tokenizer.vocab[id];
  if (id === tokenizer.eotId) return { label: "end of story", special: true };
  if (text === "\n") return { label: "new line", special: true };
  if (text.trim() === "") return { label: "space", special: true };
  if (text.startsWith(" ")) return { label: text.slice(1), special: false };
  if (/^[A-Za-z']/.test(text)) return { label: `…${text}`, special: false }; // joins onto the previous word
  return { label: text, special: false };
}

function quoted(id) {
  const { label, special } = describe(id);
  const bold = document.createElement("b");
  bold.textContent = special ? label : `“${label}”`;
  return bold;
}

/** 0 = a likely pick (no shading) ... 3 = a long shot (darkest shading). About a third of tokens get shaded. */
function unsureLevel(probability) {
  if (probability >= 0.35) return 0;
  if (probability >= 0.15) return 1;
  if (probability >= 0.05) return 2;
  return 3;
}

// ---------------------------------------------------------------- story rendering
/**
 * The pieces of page for one token. A token usually carries the space before
 * its word; that space is left outside the span so shading and outlines hug
 * the word itself.
 */
function tokenNodes(text, className) {
  const span = document.createElement("span");
  span.className = className;
  const wordAfterSpace = text.length > 1 && text.startsWith(" ");
  span.textContent = wordAfterSpace ? text.slice(1) : text;
  return { span, nodes: wordAfterSpace ? [" ", span] : [span] };
}

function writtenToken(record, index) {
  const level = unsureLevel(record.probability);
  const className = `tok written${level ? ` u${level}` : ""}${record.forced ? " forced" : ""}`;
  const { span, nodes } = tokenNodes(record.text, className);
  span.dataset.index = index;
  span.title = record.forced
    ? `You chose this. The model gave it ${percent(record.probability)}.`
    : `The model gave this ${percent(record.probability)}. Click to see the alternatives.`;
  return nodes;
}

function renderStory() {
  els.story.textContent = "";
  if (!session) {
    const placeholder = document.createElement("span");
    placeholder.className = "placeholder";
    placeholder.textContent = "Press “Write a story”. The model continues your opening one token at a time, and you can click any word afterwards to see what else it considered.";
    els.story.append(placeholder);
    return;
  }
  for (const id of session.ids.slice(0, session.promptLength)) {
    if (id === tokenizer.eotId) continue;
    els.story.append(...tokenNodes(tokenizer.vocab[id], "tok prompt").nodes);
  }
  session.tokens.forEach((record, index) => els.story.append(...writtenToken(record, index)));
  const caret = document.createElement("span");
  caret.className = "caret";
  caret.hidden = mode !== "writing";
  els.story.append(caret);
  markSelected();
}

function appendToken(record, index) {
  const atBottom = els.story.scrollHeight - els.story.scrollTop - els.story.clientHeight < 48;
  els.story.lastElementChild.before(...writtenToken(record, index));
  if (atBottom) els.story.scrollTop = els.story.scrollHeight;
}

function markSelected() {
  els.story.querySelector(".tok.selected")?.classList.remove("selected");
  if (selected === null) return;
  const span = els.story.querySelector(`.tok.written[data-index="${selected}"]`);
  span?.classList.add("selected");
  span?.scrollIntoView({ block: "nearest" });
}

// ---------------------------------------------------------------- inspector (the bars)
function setBars(rows, { past = false, clickable = true } = {}) {
  els.bars.classList.toggle("past", past);
  while (els.bars.children.length > rows.length) els.bars.lastElementChild.remove();
  while (els.bars.children.length < rows.length) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "bar-row";
    for (const className of ["bar-label", "bar-track", "bar-value"]) {
      const cell = document.createElement("span");
      cell.className = className;
      row.append(cell);
    }
    row.children[1].append(Object.assign(document.createElement("span"), { className: "bar-fill" }));
    row.children[1].firstChild.style.display = "block";
    els.bars.append(row);
  }
  rows.forEach((item, i) => {
    const row = els.bars.children[i];
    const [label, track, value] = row.children;
    row.className = `bar-row${item.chosen ? " chosen" : ""}${item.other ? " other" : ""}`;
    row.disabled = Boolean(item.other || item.chosen) || !clickable;
    if (item.other) delete row.dataset.id;
    else row.dataset.id = item.id;
    const shown = item.other ? { label: "everything else", special: false } : describe(item.id);
    label.textContent = shown.label;
    label.className = `bar-label${shown.special ? " special" : ""}`;
    track.firstChild.style.width = `${Math.max(0, Math.min(1, item.probability)) * 100}%`;
    value.textContent = percent(item.probability);
    row.setAttribute("aria-label", `${shown.label}, ${percent(item.probability)}${item.chosen ? ", chosen" : ""}`);
  });
}

function withOther(candidates, other) {
  const rows = candidates.map((c) => ({ ...c }));
  if (other >= 0.005) rows.push({ other: true, probability: other });
  return rows;
}

function renderInspector({ smooth = false } = {}) {
  els.bars.classList.toggle("smooth", smooth);
  els.back.hidden = selected === null;
  els.text.textContent = "";
  els.hint.textContent = "";

  // Looking back at a word that was already written.
  if (session && selected !== null && session.tokens[selected]) {
    const record = session.tokens[selected];
    els.title.textContent = "Why this word?";
    if (record.forced) els.text.append("You chose ", quoted(record.id), `. The model gave it ${percent(record.probability)}.`);
    else els.text.append("The model picked ", quoted(record.id), ` with a ${percent(record.probability)} chance.`);
    const rows = withOther(record.candidates, record.other).map((row) => ({ ...row, chosen: row.id === record.id }));
    if (!rows.some((row) => row.chosen)) rows.splice(KEEP, 0, { id: record.id, probability: record.probability, chosen: true });
    setBars(rows, { past: true });
    els.hint.textContent = "Click another option to rewrite the story from this word.";
    return;
  }

  els.title.textContent = "Next word";
  if (!session) {
    els.text.textContent = "The model's guesses for the next token appear here as it writes.";
    setBars([]);
    return;
  }
  if (mode === "finished") {
    els.title.textContent = "The end";
    els.text.textContent = session.finished
      ? "The model chose to end the story here."
      : `The page stops after ${MAX_TOKENS} tokens.`;
    setBars([]);
    els.hint.textContent = "Click any word in the story to see what else it could have been.";
    return;
  }
  const { candidates, other } = session.next({ temperature: temperature(), topK: TOP_K, keep: KEEP });
  els.text.textContent = "The model's guesses for what comes next. One is drawn at random, in proportion to its bar.";
  setBars(withOther(candidates, other));
  els.hint.textContent = "Click a guess to choose it yourself. … marks a piece that joins the previous word.";
}

let inspectorQueued = false;
let inspectorSmooth = false;
function scheduleInspector({ smooth = false } = {}) {
  inspectorSmooth = smooth;
  if (inspectorQueued) return;
  inspectorQueued = true;
  requestAnimationFrame(() => {
    inspectorQueued = false;
    renderInspector({ smooth: inspectorSmooth });
  });
}

// ---------------------------------------------------------------- controls
function updateControls() {
  const ready = model !== null;
  let label = "Write a story";
  if (mode === "writing") label = "Pause";
  else if (mode === "paused" && promptUnchanged()) label = "Continue";
  else if (mode === "finished" && promptUnchanged()) label = "Write another";
  els.primary.textContent = label;
  els.primary.disabled = !ready;
  els.step.disabled = !ready || (mode === "finished" && promptUnchanged());

  const caret = els.story.querySelector(".caret");
  if (caret) caret.hidden = mode !== "writing";

  const written = session ? session.tokens.length : 0;
  let count = written ? `${written} token${written === 1 ? "" : "s"}` : "";
  if (mode === "writing" && speed() === "fast" && burst.count >= 12) {
    const perSecond = burst.count / Math.max((performance.now() - burst.start) / 1000, 0.001);
    count += ` · ${Math.round(perSecond)} per second`;
  }
  els.count.textContent = count;
}

function setMode(next) {
  mode = next;
  updateControls();
}

// ---------------------------------------------------------------- writing
/** Write one token (the model's pick, or `forceId`). Must run inside enqueue(). */
async function stepOnce(forceId) {
  if (!session || session.finished || session.tokens.length >= MAX_TOKENS) return;
  const record = await session.advance({ temperature: temperature(), topK: TOP_K, keep: KEEP, forceId });
  if (record) {
    appendToken(record, session.tokens.length - 1);
    burst.count++;
  }
  if (record === null || session.tokens.length >= MAX_TOKENS) {
    run++;
    setMode("finished");
  } else {
    updateControls();
  }
  scheduleInspector();
}

async function writeLoop(myRun) {
  burst = { start: performance.now(), count: 0 };
  while (run === myRun && mode === "writing") {
    await enqueue(() => (run === myRun && mode === "writing" ? stepOnce() : undefined));
    const delay = SPEED_DELAY[speed()];
    if (delay > 0) await sleep(delay);
    else if (burst.count % 6 === 0) await sleep(0); // let the browser repaint
  }
}

function startWriting() {
  selected = null;
  markSelected();
  setMode("writing");
  scheduleInspector();
  writeLoop(++run);
}

function pauseWriting() {
  run++;
  if (mode === "writing") setMode("paused");
}

/** Begin a new story from the text in the prompt box. */
function newStory({ autoplay }) {
  run++;
  const prompt = els.prompt.value;
  return enqueue(async () => {
    session = await StorySession.create(model, tokenizer, prompt);
    selected = null;
    mode = autoplay ? "writing" : "paused";
    renderStory();
    updateControls();
    renderInspector();
    if (autoplay) writeLoop(++run);
  });
}

function onPrimary() {
  if (!model) return;
  if (mode === "writing") return pauseWriting();
  if (mode === "paused" && promptUnchanged()) return startWriting();
  return newStory({ autoplay: true });
}

function onStep() {
  if (!model) return;
  pauseWriting();
  if (!session || !promptUnchanged()) newStory({ autoplay: false });
  selected = null;
  markSelected();
  enqueue(() => stepOnce());
}

/** The reader clicked one of the bars. */
function pickCandidate(id) {
  if (!session) return;
  pauseWriting();
  const rewindTo = selected;
  enqueue(async () => {
    if (rewindTo !== null) {
      // Rewrite history: drop everything from the selected word on, then put the new choice there.
      await session.rewind(rewindTo);
      selected = null;
      mode = "paused";
      renderStory();
    } else if (mode === "finished") {
      return;
    }
    await stepOnce(id);
    if (mode !== "finished") setMode("paused");
  });
}

function select(index) {
  if (!session || session.tokens.length === 0) return;
  pauseWriting();
  selected = index === null ? null : Math.max(0, Math.min(session.tokens.length - 1, index));
  markSelected();
  renderInspector();
  // On a narrow screen the bars sit below the story: bring them into view.
  if (selected !== null) els.title.closest(".inspector").scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------------------------------------------------------------- loading a model
async function loadModel(name, listing) {
  const ticket = ++loadTicket;
  run++;
  model = null;
  session = null;
  selected = null;
  mode = "idle";
  els.badge.hidden = true;
  renderStory();
  updateControls();
  renderInspector();

  try {
    const loaded = await TinyGPT.load(`models/${name}`);
    const parity = await fetch(`models/${name}/parity.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    if (ticket !== loadTicket) return; // the reader picked another model meanwhile

    const { manifest } = loaded;
    const loadedTokenizer = new Tokenizer(manifest.tokenizer.vocab, manifest.tokenizer.merges);
    const test = parity ? runSelfTest(loaded.model, loadedTokenizer, parity) : null;
    model = loaded.model;
    tokenizer = loadedTokenizer;

    const millions = `${(manifest.parameters / 1e6).toFixed(1)}M`;
    const megabytes = listing.find((item) => item.name === name)?.megabytes;
    els.facts.textContent = `${millions} parameters${megabytes ? `, ${megabytes} MB.` : "."}`;
    els.howModel.textContent =
      `This one has ${millions} parameters in ${manifest.config.n_layer} layers and a vocabulary of ` +
      `${manifest.config.vocab_size.toLocaleString("en")} tokens.`;

    els.badge.hidden = false;
    els.badge.textContent = "";
    if (test?.ok) {
      els.badge.className = "badge pass";
      els.badge.append(Object.assign(document.createElement("strong"), { textContent: "✓ Matches PyTorch" }));
      els.badge.title = "Checked when the page loaded: same token ids, same output scores and same text as the original PyTorch model.";
      els.howCheck.textContent =
        `Checked on load against the original PyTorch model: tokenizer ${test.tokenizerPassed}/${test.tokenizerTotal}, ` +
        `largest score difference ${test.maxLogitDifference.toExponential(1)}, identical text.`;
    } else {
      els.badge.className = "badge";
      els.badge.textContent = test ? "✗ Differs from PyTorch" : "Self-test skipped";
      els.howCheck.textContent = test
        ? "The self-test against PyTorch failed for this model; re-export it with training/export.py."
        : "";
    }
    updateControls();
    renderInspector();
  } catch (error) {
    if (ticket !== loadTicket) return;
    els.app.hidden = true;
    els.failed.hidden = false;
    els.failedText.textContent = error.message;
  }
}

// ---------------------------------------------------------------- start-up
async function start() {
  els.credit.textContent = "";
  els.credit.append("Built by ", Object.assign(document.createElement("a"), { href: AUTHOR.url, textContent: AUTHOR.name }), ".");
  if (REPO_URL) {
    els.repo.href = REPO_URL;
    els.repo.hidden = false;
  }

  let listing = [];
  try {
    const response = await fetch("models/index.json");
    if (response.ok) listing = await response.json();
  } catch {
    // No listing: fall through to the "no model" message.
  }
  if (!Array.isArray(listing) || listing.length === 0) {
    els.empty.hidden = false;
    return;
  }

  for (const item of listing) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = `Model: ${item.name} (${(item.parameters / 1e6).toFixed(1)}M)`;
    els.model.append(option);
  }
  // Start with the model that scored best on held-out stories.
  els.model.value = listing.reduce((best, item) => (item.val_loss < best.val_loss ? item : best)).name;
  els.model.hidden = listing.length < 2;
  els.app.hidden = false;

  els.model.addEventListener("change", () => loadModel(els.model.value, listing));
  els.primary.addEventListener("click", onPrimary);
  els.step.addEventListener("click", onStep);
  els.back.addEventListener("click", () => select(null));
  els.prompt.addEventListener("input", updateControls);
  els.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && model) newStory({ autoplay: true });
  });
  document.querySelectorAll(".chip").forEach((chip) =>
    chip.addEventListener("click", () => {
      els.prompt.value = chip.dataset.prompt;
      if (model) newStory({ autoplay: true });
    }),
  );
  els.temperature.addEventListener("input", () => {
    els.temperatureValue.textContent = temperature().toFixed(2).replace(/0$/, "");
    if (selected === null) scheduleInspector({ smooth: true }); // the bars follow the slider live
  });
  els.boundaries.addEventListener("change", () => els.story.classList.toggle("boundaries", els.boundaries.checked));
  els.bars.addEventListener("click", (event) => {
    const row = event.target.closest("button.bar-row");
    if (row && !row.disabled && row.dataset.id !== undefined) pickCandidate(Number(row.dataset.id));
  });
  els.story.addEventListener("click", (event) => {
    const token = event.target.closest(".tok.written");
    if (token) select(Number(token.dataset.index));
  });
  els.story.addEventListener("keydown", (event) => {
    if (!session || session.tokens.length === 0) return;
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      const last = session.tokens.length - 1;
      if (selected === null) select(event.key === "ArrowLeft" ? last : 0);
      else select(selected + (event.key === "ArrowRight" ? 1 : -1));
    } else if (event.key === "Escape") {
      select(null);
    }
  });

  await loadModel(els.model.value, listing);
}

start();
