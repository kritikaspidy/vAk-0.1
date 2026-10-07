// Proof that the JavaScript model computes the same thing as PyTorch.
// training/export.py saves reference answers in parity.json; this compares
// against them. The same check runs in the browser and in the Node tests.

import { argmax } from "./generate.js";

const sameIds = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);

/**
 * Returns {
 *   ok,                    true if every check passed
 *   tokenizerPassed/Total, how many texts produced exactly PyTorch's token ids
 *   maxLogitDifference,    largest gap between our output scores and PyTorch's
 *   greedyMatches,         same text when always picking the most likely token
 *   greedyText,            that text, as written by this JavaScript model
 * }
 */
export function runSelfTest(model, tokenizer, parity, { logitTolerance = 1e-3 } = {}) {
  // 1. Tokenizer: identical ids for every reference text.
  const tokenizerPassed = parity.tokenizer_cases.filter((c) => sameIds(tokenizer.encode(c.text), c.ids)).length;

  // 2. Forward pass: identical scores after reading the reference prompt.
  model.reset();
  let logits;
  for (const id of parity.prompt_ids) logits = model.step(id);
  let maxLogitDifference = 0;
  for (let i = 0; i < logits.length; i++) {
    maxLogitDifference = Math.max(maxLogitDifference, Math.abs(logits[i] - parity.logits[i]));
  }

  // 3. Generation: always taking the most likely token must write the same text.
  const ids = [...parity.prompt_ids];
  while (ids.length < parity.greedy_ids.length && model.length < model.config.block_size) {
    const next = argmax(logits);
    ids.push(next);
    logits = model.step(next);
  }
  const greedyMatches = sameIds(ids, parity.greedy_ids.slice(0, ids.length)) && ids.length === parity.greedy_ids.length;
  model.reset();

  return {
    ok: tokenizerPassed === parity.tokenizer_cases.length && maxLogitDifference <= logitTolerance && greedyMatches,
    tokenizerPassed,
    tokenizerTotal: parity.tokenizer_cases.length,
    maxLogitDifference,
    greedyMatches,
    greedyText: tokenizer.decode(ids),
  };
}
