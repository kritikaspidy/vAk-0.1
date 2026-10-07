// The BPE tokenizer, ported from training/tokenizer.py. It must turn text into
// exactly the same token ids as the Python version, or the model would be fed
// numbers it was never trained on. The self-test checks that.

export const EOT = "<|endoftext|>";

// One chunk = a word (with optional leading space), one digit, one punctuation
// mark, or one whitespace character. Same pattern as the Python tokenizer.
const CHUNK = / ?[A-Za-z']+| ?\d| ?[^\sA-Za-z\d']|\s/g;

const ASCII = {
  "“": '"', "”": '"', "‘": "'", "’": "'",
  "—": "-", "–": "-", "…": "...", " ": " ", "\r": "", "\t": " ",
};

/** Replace curly quotes, long dashes and similar with their plain ASCII forms. */
export function normalise(text) {
  return text.replace(/[“”‘’—–… \r\t]/g, (char) => ASCII[char]);
}

const pairKey = (a, b) => `${a}\u0001${b}`;

function mergePair(symbols, a, b) {
  const out = [];
  for (let i = 0; i < symbols.length; i++) {
    if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
      out.push(a + b);
      i++;
    } else {
      out.push(symbols[i]);
    }
  }
  return out;
}

export class Tokenizer {
  /** vocab: id -> token text. merges: [a, b] pairs in the order they were learned. */
  constructor(vocab, merges) {
    this.vocab = vocab;
    this.tokenToId = new Map(vocab.map((token, id) => [token, id]));
    this.rank = new Map(merges.map(([a, b], index) => [pairKey(a, b), index]));
    this.eotId = this.tokenToId.get(EOT);
    this.cache = new Map();
  }

  encodeChunk(chunk) {
    let symbols = Array.from(chunk).filter((char) => this.tokenToId.has(char)); // unknown characters are dropped
    while (symbols.length > 1) {
      // Apply the merge that was learned earliest among the pairs present.
      let bestRank = Infinity;
      let bestIndex = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.rank.get(pairKey(symbols[i], symbols[i + 1]));
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex === -1) break; // no learned merge applies any more
      symbols = mergePair(symbols, symbols[bestIndex], symbols[bestIndex + 1]);
    }
    return symbols.map((symbol) => this.tokenToId.get(symbol));
  }

  /** Text -> array of token ids. */
  encode(text) {
    const ids = [];
    normalise(text).split(EOT).forEach((part, index) => {
      if (index > 0) ids.push(this.eotId);
      for (const chunk of part.match(CHUNK) ?? []) {
        let cached = this.cache.get(chunk);
        if (cached === undefined) {
          cached = this.encodeChunk(chunk);
          this.cache.set(chunk, cached);
        }
        ids.push(...cached);
      }
    });
    return ids;
  }

  /** Array of token ids -> text. */
  decode(ids) {
    return Array.from(ids, (id) => this.vocab[id]).join("");
  }
}
