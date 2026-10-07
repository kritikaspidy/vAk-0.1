"""Tests for the tokenizer and the model.

    python training/test_all.py

They run in a few seconds on the CPU and need no dataset.
"""
import tempfile
import unittest
from pathlib import Path

import torch

from gpt import GPT, GPTConfig, PRESETS
from tokenizer import EOT, Tokenizer, normalise

TEXT = EOT.join([
    'Once upon a time, there was a little girl named Lily. She loved to play outside.',
    'Tom had a big red ball. "Let\'s play!" said Tom. The ball was 3 years old.',
    "One day, Lily and Tom went to the park.\n\nThey saw a little dog and they were happy.",
] * 20)


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.train(TEXT, vocab_size=120)

    def test_round_trip_is_lossless(self):
        for text in ["Once upon a time", "Tom had a big red ball.", 'She said, "Let\'s play!"\n\nThe end.', "  two  spaces", ""]:
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_frequent_words_become_single_tokens(self):
        self.assertIn(" the", self.tok.token_to_id)
        self.assertEqual(len(self.tok.encode(" little")), 1)

    def test_end_of_text_is_one_special_token(self):
        ids = self.tok.encode(f"The end.{EOT}Once")
        self.assertEqual(ids.count(self.tok.eot_id), 1)
        self.assertEqual(self.tok.eot_id, 0)
        self.assertEqual(self.tok.decode(ids), f"The end.{EOT}Once")

    def test_unknown_characters_are_dropped_not_crashed_on(self):
        self.assertEqual(self.tok.decode(self.tok.encode("a big é中 ball")), "a big  ball")

    def test_curly_quotes_are_normalised(self):
        self.assertEqual(normalise("“Hi” — it’s…"), '"Hi" - it\'s...')
        self.assertEqual(self.tok.encode("“Let’s play!”"), self.tok.encode('"Let\'s play!"'))

    def test_training_is_deterministic(self):
        again = Tokenizer.train(TEXT, vocab_size=120)
        self.assertEqual(again.vocab, self.tok.vocab)
        self.assertEqual(again.merges, self.tok.merges)

    def test_vocab_size_is_respected(self):
        self.assertLessEqual(len(self.tok), 120)
        self.assertEqual(len(set(self.tok.vocab)), len(self.tok.vocab))  # no duplicate tokens

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tokenizer.json"
            self.tok.save(path)
            loaded = Tokenizer.load(path)
        sample = "Lily and Tom went to the park."
        self.assertEqual(loaded.encode(sample), self.tok.encode(sample))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32)
        self.model = GPT(self.config)

    def test_output_shape_and_initial_loss(self):
        x = torch.randint(50, (4, 16))
        y = torch.randint(50, (4, 16))  # random targets: nothing to exploit
        logits, loss = self.model(x, y)
        self.assertEqual(logits.shape, (4, 16, 50))
        # An untrained model should be about as good as guessing: loss close to ln(50) = 3.91
        self.assertAlmostEqual(loss.item(), 3.91, delta=0.3)

    def test_model_cannot_see_the_future(self):
        """Changing a later token must not change the predictions at earlier positions."""
        self.model.eval()
        a = torch.randint(50, (1, 16))
        b = a.clone()
        b[0, 10] = (b[0, 10] + 1) % 50
        logits_a, _ = self.model(a)
        logits_b, _ = self.model(b)
        self.assertTrue(torch.allclose(logits_a[0, :10], logits_b[0, :10], atol=1e-5))
        self.assertFalse(torch.allclose(logits_a[0, 10:], logits_b[0, 10:], atol=1e-5))

    def test_it_can_learn(self):
        """A few optimiser steps on one batch must lower the loss a lot (it should memorise it)."""
        x = torch.randint(50, (8, 16))
        y = torch.roll(x, -1, dims=1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        first = self.model(x, y)[1].item()
        for _ in range(60):
            loss = self.model(x, y)[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), first * 0.5)

    def test_generate_extends_the_sequence_and_stops_at_stop_token(self):
        self.model.eval()
        start = torch.zeros((1, 3), dtype=torch.long)
        out = self.model.generate(start, max_new_tokens=30)
        self.assertEqual(out.shape, (1, 33))  # also proves it copes with more tokens than block_size
        self.assertTrue(torch.equal(out[:, :3], start))
        stopped = self.model.generate(start, max_new_tokens=500, temperature=1.5, top_k=None, stop_token=7)
        self.assertTrue(stopped.shape[1] < 503 and stopped[0, -1].item() == 7)

    def test_output_layer_shares_weights_with_embedding(self):
        self.assertIs(self.model.head.weight, self.model.tok_emb.weight)

    def test_preset_sizes(self):
        sizes = {name: GPT(GPTConfig(vocab_size=2048, **preset)).num_parameters() for name, preset in PRESETS.items()}
        self.assertTrue(0.8e6 < sizes["tiny"] < 1.5e6, sizes)
        self.assertTrue(2.5e6 < sizes["small"] < 4e6, sizes)
        self.assertTrue(6e6 < sizes["base"] < 8e6, sizes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
