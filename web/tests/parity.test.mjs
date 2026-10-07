// Run with:  node --test web/tests/parity.test.mjs
//
// Checks every exported model in web/models/ against the PyTorch reference
// answers saved beside it, plus a few direct tests of the sampler.

import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { argmax, generate, sampleToken, seededRandom, StorySession } from "../js/generate.js";
import { VakModel } from "../js/model.js";
import { runSelfTest } from "../js/selftest.js";
import { Tokenizer } from "../js/tokenizer.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "models");
const names = existsSync(MODELS)
  ? readdirSync(MODELS).filter((name) => existsSync(join(MODELS, name, "model.json")))
  : [];

function loadModel(name) {
  const manifest = JSON.parse(readFileSync(join(MODELS, name, "model.json"), "utf8"));
  const bytes = readFileSync(join(MODELS, name, "weights.bin"));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const parity = JSON.parse(readFileSync(join(MODELS, name, "parity.json"), "utf8"));
  return {
    model: new VakModel(manifest, buffer),
    tokenizer: new Tokenizer(manifest.tokenizer.vocab, manifest.tokenizer.merges),
    parity,
  };
}

test("at least one model has been exported", () => {
  assert.ok(names.length > 0, "run: python training/export.py");
});

for (const name of names) {
  test(`${name}: JavaScript matches PyTorch`, () => {
    const { model, tokenizer, parity } = loadModel(name);
    const result = runSelfTest(model, tokenizer, parity);
    assert.equal(result.tokenizerPassed, result.tokenizerTotal, "tokenizer ids differ from Python");
    assert.ok(result.maxLogitDifference <= 1e-3, `logits differ by ${result.maxLogitDifference}`);
    assert.ok(result.greedyMatches, `greedy text differs:\n  js:      ${result.greedyText}\n  pytorch: ${parity.greedy_text}`);
  });

  test(`${name}: generation is repeatable with a seed and survives a full context`, async () => {
    const { model, tokenizer } = loadModel(name);
    const write = async (seed) => {
      let text = "";
      let count = 0;
      // More tokens than the context holds, so the "drop the older half" path runs.
      for await (const token of generate(model, tokenizer, "Once upon a time", { seed, maxTokens: model.config.block_size + 40 })) {
        text += token.text;
        count++;
        assert.ok(token.candidates.length > 0 && token.candidates[0].probability <= 1);
      }
      return { text, count };
    };
    const first = await write(7);
    const again = await write(7);
    assert.equal(first.text, again.text);
    assert.ok(first.count > 0);
  });
}

for (const name of names) {
  test(`${name}: a story session records alternatives, can be steered, and can be rewound`, async () => {
    const { model, tokenizer } = loadModel(name);
    const session = await StorySession.create(model, tokenizer, "Once upon a time");
    const greedy = { temperature: 0 };

    // What could come next: sorted, and the probabilities account for everything.
    const { candidates, other } = session.next({ temperature: 0.8, keep: 8 });
    assert.equal(candidates.length, 8);
    assert.ok(candidates.every((c, i) => i === 0 || c.probability <= candidates[i - 1].probability));
    assert.ok(Math.abs(candidates.reduce((sum, c) => sum + c.probability, 0) + other - 1) < 1e-9);
    assert.equal(session.next(greedy).candidates[0].probability, 1);

    // Write 30 tokens, always taking the most likely one.
    for (let i = 0; i < 30; i++) await session.advance(greedy);
    const original = session.tokens.map((t) => t.id);
    assert.equal(original.length, 30);
    assert.ok(session.tokens.every((t) => t.rank === 0 && t.probability === 1 && !t.forced));

    // Rewind to token 10 and write on: the model must reproduce the same 20 tokens.
    await session.rewind(10);
    assert.equal(session.tokens.length, 10);
    for (let i = 0; i < 20; i++) await session.advance(greedy);
    assert.deepEqual(session.tokens.map((t) => t.id), original);

    // Rewind again, but force the model's second choice: the story must change from there.
    await session.rewind(10);
    const second = session.next({ temperature: 0.8 }).candidates[1].id;
    const forced = await session.advance({ temperature: 0.8, forceId: second });
    assert.ok(forced.forced && forced.id === second && forced.rank === 1);
    assert.notEqual(session.tokens[10].id, original[10]);

    // Forcing the end-of-story token ends the story; rewinding reopens it.
    assert.equal(await session.advance({ forceId: tokenizer.eotId }), null);
    assert.ok(session.finished);
    assert.equal(await session.advance(greedy), null);
    await session.rewind(5);
    assert.ok(!session.finished && session.tokens.length === 5);
    assert.deepEqual(session.tokens.map((t) => t.id), original.slice(0, 5));
  });
}

test("sampleToken: temperature 0 picks the top score, candidates are sorted and sum to at most 1", () => {
  const logits = Float32Array.from([0.1, 2.5, -1, 2.4, 0]);
  const { id, candidates } = sampleToken(logits, { temperature: 0, topK: 3 });
  assert.equal(id, 1);
  assert.equal(id, argmax(logits));
  assert.deepEqual(candidates.map((c) => c.id), [1, 3, 0]);
  const total = candidates.reduce((sum, c) => sum + c.probability, 0);
  assert.ok(Math.abs(total - 1) < 1e-9);
});

test("sampleToken: only the top k tokens are ever chosen, roughly in proportion to their probability", () => {
  const logits = Float32Array.from([Math.log(6), Math.log(3), Math.log(1), 5 - 100, -100]);
  const random = seededRandom(1);
  const counts = [0, 0, 0, 0, 0];
  for (let i = 0; i < 6000; i++) counts[sampleToken(logits, { temperature: 1, topK: 2, random }).id]++;
  assert.equal(counts[2] + counts[3] + counts[4], 0);
  assert.ok(Math.abs(counts[0] / 6000 - 2 / 3) < 0.03, `got ${counts}`);
});
