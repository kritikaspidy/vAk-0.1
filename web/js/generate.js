// Turning the model's scores into text: sampling one token at a time.

/** A small seeded random number generator (mulberry32), so a given seed always writes the same story. */
export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Index of the largest score: the single most likely next token. */
export function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

/**
 * The probabilities the next token is actually drawn from.
 *   temperature  0 = all the probability goes to the most likely token;
 *                1 = exactly what the model predicts; higher = flatter
 *   topK         only the k most likely tokens keep any probability
 * Returns { ids, probabilities }: the surviving tokens, most likely first,
 * with probabilities that sum to 1.
 */
export function distribution(logits, { temperature = 0.8, topK = 40 } = {}) {
  const k = Math.max(1, Math.min(topK ?? logits.length, logits.length));
  const ids = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, k);
  if (temperature <= 0) return { ids, probabilities: ids.map((_, i) => (i === 0 ? 1 : 0)) };

  const top = logits[ids[0]];
  const weights = ids.map((id) => Math.exp((logits[id] - top) / temperature)); // softmax, shifted for stability
  const total = weights.reduce((a, b) => a + b, 0);
  return { ids, probabilities: weights.map((w) => w / total) };
}

/** Draw one position from a list of probabilities that sum to 1. */
function draw(probabilities, random) {
  let threshold = random();
  for (let i = 0; i < probabilities.length; i++) {
    threshold -= probabilities[i];
    if (threshold <= 0) return i;
  }
  return probabilities.length - 1;
}

/**
 * Choose the next token from the model's scores.
 * Returns { id, candidates }, where candidates lists the top tokens with the
 * probability each had of being picked: [{ id, probability }, ...].
 */
export function sampleToken(logits, { temperature = 0.8, topK = 40, random = Math.random, keep = 10 } = {}) {
  const { ids, probabilities } = distribution(logits, { temperature, topK });
  const candidates = ids.slice(0, keep).map((id, i) => ({ id, probability: probabilities[i] }));
  return { id: ids[draw(probabilities, random)], candidates };
}

const pause = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * One story being written. It owns the model's memory while it is in use and
 * keeps, for every token it wrote, what the alternatives were. That record is
 * what lets the page show "why this word?" and rewind to try another one.
 *
 *   const session = await StorySession.create(model, tokenizer, "Once upon a time");
 *   session.next({ temperature })          -> what could come next, with probabilities
 *   await session.advance({ temperature }) -> write one token (or force one with { forceId })
 *   await session.rewind(5)                -> keep the first 5 written tokens, drop the rest
 */
export class StorySession {
  constructor(model, tokenizer, prompt) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.prompt = prompt;
    let ids = tokenizer.encode(prompt);
    if (ids.length === 0) ids = [tokenizer.eotId]; // empty prompt: start a fresh story
    this.ids = ids.slice(-(model.config.block_size - 1)); // prompt tokens, then everything written
    this.promptLength = this.ids.length;
    this.tokens = []; // one record per written token
    this.logits = null; // the model's scores for the next token
    this.finished = false; // true once the model chose to end the story
  }

  static async create(model, tokenizer, prompt) {
    const session = new StorySession(model, tokenizer, prompt);
    await session.reread();
    return session;
  }

  /**
   * Make the model read the story so far from the start of its memory.
   * The model holds at most block_size tokens; if the story is longer, only
   * the most recent half-window is read, leaving room to keep writing.
   */
  async reread() {
    const blockSize = this.model.config.block_size;
    const visible = this.ids.length < blockSize ? this.ids : this.ids.slice(-Math.floor(blockSize / 2));
    this.model.reset();
    for (let i = 0; i < visible.length; i++) {
      this.logits = this.model.step(visible[i]);
      if (i % 32 === 31) await pause(); // keep the page responsive during a long re-read
    }
  }

  /** The top candidates for the next token and the probability left over for all the others. */
  next({ temperature = 0.8, topK = 40, keep = 8 } = {}) {
    const { ids, probabilities } = distribution(this.logits, { temperature, topK });
    const candidates = ids.slice(0, keep).map((id, i) => ({ id, probability: probabilities[i] }));
    const shown = candidates.reduce((sum, c) => sum + c.probability, 0);
    return { candidates, other: Math.max(0, 1 - shown) };
  }

  /**
   * Write one token: sampled from the model's prediction, or `forceId` if given.
   * Returns the record for the new token, or null if the story ended instead.
   */
  async advance({ temperature = 0.8, topK = 40, random = Math.random, keep = 8, forceId } = {}) {
    if (this.finished) return null;
    const { ids, probabilities } = distribution(this.logits, { temperature, topK });
    const forced = forceId !== undefined;
    const id = forced ? forceId : ids[draw(probabilities, random)];
    const rank = ids.indexOf(id); // 0 = the model's first choice; -1 = outside the top k

    const candidates = ids.slice(0, keep).map((candidate, i) => ({ id: candidate, probability: probabilities[i] }));
    const record = {
      id,
      text: this.tokenizer.vocab[id],
      probability: rank === -1 ? 0 : probabilities[rank],
      rank,
      candidates,
      other: Math.max(0, 1 - candidates.reduce((sum, c) => sum + c.probability, 0)),
      forced,
    };

    if (id === this.tokenizer.eotId) {
      this.finished = true;
      this.ending = record; // kept so the page can still show why the story stopped
      return null;
    }
    this.ids.push(id);
    this.tokens.push(record);
    if (this.model.length >= this.model.config.block_size) await this.reread();
    else this.logits = this.model.step(id);
    return record;
  }

  /** Go back in time: keep only the first `count` written tokens and re-read the story up to there. */
  async rewind(count) {
    this.tokens.length = Math.min(count, this.tokens.length);
    this.ids.length = this.promptLength + this.tokens.length;
    this.finished = false;
    this.ending = undefined;
    await this.reread();
  }
}

/**
 * Write a whole story. An async generator: each `yield` is one new token,
 * { id, text, candidates }. Stops at the end-of-story token or after maxTokens.
 */
export async function* generate(model, tokenizer, prompt, options = {}) {
  const { maxTokens = 250, temperature = 0.8, topK = 40, seed, signal } = options;
  const random = seed === undefined ? Math.random : seededRandom(seed);
  const session = await StorySession.create(model, tokenizer, prompt);
  for (let produced = 0; produced < maxTokens; produced++) {
    if (signal?.aborted) return;
    const record = await session.advance({ temperature, topK, random });
    if (record === null) return;
    yield record;
    if (produced % 4 === 3) await pause(); // let the browser repaint now and then
  }
}
