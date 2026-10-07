"""A byte-pair encoding (BPE) tokenizer, written from scratch.

A language model works on numbers, not text. The tokenizer turns text into a
list of token ids and back again.

How BPE works:
  1. Start with one token per character.
  2. Find the pair of neighbouring tokens that occurs most often in the
     training text and glue it into a new token ("t" + "h" -> "th").
  3. Repeat until the vocabulary has the size you asked for.
Frequent words end up as a single token; rare words are spelled out of pieces.

To keep it fast, the text is first cut into chunks (a word with its leading
space, a digit, or a punctuation mark) and merges only happen inside a chunk.
Training then works on the count of each distinct chunk instead of the raw text.
"""
import json
import re
from collections import Counter

EOT = "<|endoftext|>"  # special token that marks the end of a story (always id 0)

# One chunk = a word (with optional leading space), one digit, one punctuation
# mark, or one whitespace character. Every character lands in exactly one chunk,
# so joining the chunks gives back the original text.
CHUNK = re.compile(r" ?[A-Za-z']+| ?\d| ?[^\sA-Za-z\d']|\s")

# Text is normalised to plain ASCII so the browser version of this tokenizer
# (written later in JavaScript) behaves identically.
_ASCII = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'",
                        "—": "-", "–": "-", "…": "...", " ": " ", "\r": "", "\t": " "})


def normalise(text: str) -> str:
    """Replace curly quotes, long dashes and similar with their plain ASCII forms."""
    return text.translate(_ASCII)


def _merge_pair(symbols: list[str], a: str, b: str) -> list[str]:
    """Return `symbols` with every neighbouring (a, b) replaced by the single token a+b."""
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, vocab: list[str], merges: list[tuple[str, str]]):
        self.vocab = vocab  # id -> token text
        self.merges = merges  # learned merges, in the order they were learned
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.rank = {pair: i for i, pair in enumerate(merges)}  # earlier merge = lower rank
        self.eot_id = self.token_to_id[EOT]
        self._cache: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------ training
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "Tokenizer":
        """Learn a vocabulary of `vocab_size` tokens from `text`."""
        chunk_counts: Counter = Counter()
        for story in normalise(text).split(EOT):
            chunk_counts.update(CHUNK.findall(story))
        alphabet = sorted({char for chunk in chunk_counts for char in chunk})
        vocab = [EOT] + alphabet
        merges: list[tuple[str, str]] = []

        # Each distinct chunk as a list of symbols, with how often it occurs.
        words = [(list(chunk), count) for chunk, count in chunk_counts.items()]

        while len(vocab) < vocab_size:
            pair_counts: Counter = Counter()
            for symbols, count in words:
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                break  # every chunk is already a single token
            # Most frequent pair; ties are broken alphabetically so training is repeatable.
            best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            if pair_counts[best] < 2:
                break
            a, b = best
            merges.append(best)
            vocab.append(a + b)
            words = [(_merge_pair(s, a, b) if a in s and b in s else s, c) for s, c in words]
            if verbose and len(vocab) % 200 == 0:
                print(f"  vocab {len(vocab):>5}/{vocab_size}   newest token: {a + b!r}")

        return cls(vocab, merges)

    # ------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        symbols = [char for char in chunk if char in self.token_to_id]  # unknown characters are dropped
        while len(symbols) > 1:
            # Apply the merge that was learned earliest among the pairs present.
            best_rank, best_pair = min(
                (self.rank.get(pair, float("inf")), pair) for pair in zip(symbols, symbols[1:])
            )
            if best_rank == float("inf"):
                break  # no learned merge applies any more
            symbols = _merge_pair(symbols, *best_pair)
        return [self.token_to_id[s] for s in symbols]

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids. The EOT marker in the text becomes the EOT token."""
        ids: list[int] = []
        for i, part in enumerate(normalise(text).split(EOT)):
            if i > 0:
                ids.append(self.eot_id)
            for chunk in CHUNK.findall(part):
                cached = self._cache.get(chunk)
                if cached is None:
                    cached = self._cache[chunk] = self._encode_chunk(chunk)
                ids.extend(cached)
        return ids

    def decode(self, ids) -> str:
        """List of token ids -> text."""
        return "".join(self.vocab[int(i)] for i in ids)

    # ------------------------------------------------------------ saving
    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f, ensure_ascii=True)

    @classmethod
    def load(cls, path) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["vocab"], [tuple(pair) for pair in data["merges"]])
