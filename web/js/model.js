// The transformer from training/gpt.py, re-implemented in plain JavaScript.
// No libraries: the forward pass is loops over Float32Arrays.
//
// It runs one token at a time and remembers each layer's attention keys and
// values for the tokens it has already seen (a "KV cache"). Without the cache
// every new token would recompute the whole sequence; with it, each new token
// costs one pass through the network.

const LAYER_NORM_EPSILON = 1e-5; // PyTorch's default

/** out = W x, where W is stored row by row with shape (rows, x.length). */
function matVec(W, x, rows, out) {
  const cols = x.length;
  for (let r = 0, base = 0; r < rows; r++, base += cols) {
    let sum = 0;
    for (let c = 0; c < cols; c++) sum += W[base + c] * x[c];
    out[r] = sum;
  }
  return out;
}

/** Shift and scale a vector to mean 0 and variance 1, then apply the learned gain and bias. */
function layerNorm(x, gain, bias, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
  const scale = 1 / Math.sqrt(variance / n + LAYER_NORM_EPSILON);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * scale * gain[i] + bias[i];
  return out;
}

/** The same tanh approximation of GELU that the PyTorch model uses. */
const GELU_SCALE = Math.sqrt(2 / Math.PI);
const gelu = (x) => 0.5 * x * (1 + Math.tanh(GELU_SCALE * (x + 0.044715 * x * x * x)));

/** Decode IEEE 754 half-precision numbers (the format of weights.bin) into a Float32Array. */
function halfToFloat32(bits) {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x3ff;
    if (exponent === 0) out[i] = sign * fraction * 2 ** -24; // subnormal (and zero)
    else if (exponent === 31) out[i] = fraction ? NaN : sign * Infinity;
    else out[i] = sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
  }
  return out;
}

export class VakModel {
  /**
   * manifest: the parsed model.json. buffer: the ArrayBuffer of weights.bin.
   */
  constructor(manifest, buffer) {
    this.name = manifest.name;
    this.config = manifest.config;
    this.parameters = manifest.parameters;
    const all = manifest.dtype === "float16" ? halfToFloat32(new Uint16Array(buffer)) : new Float32Array(buffer);

    const tensor = {};
    for (const { name, shape, offset } of manifest.tensors) {
      tensor[name] = all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
    }
    this.tokEmb = tensor["tok_emb.weight"]; // also the output layer (weight tying)
    this.posEmb = tensor["pos_emb.weight"];
    this.lnF = { gain: tensor["ln_f.weight"], bias: tensor["ln_f.bias"] };
    this.layers = [];
    for (let i = 0; i < this.config.n_layer; i++) {
      const p = `blocks.${i}.`;
      this.layers.push({
        ln1: { gain: tensor[`${p}ln1.weight`], bias: tensor[`${p}ln1.bias`] },
        qkv: tensor[`${p}attn.qkv.weight`],
        attnProj: tensor[`${p}attn.proj.weight`],
        ln2: { gain: tensor[`${p}ln2.weight`], bias: tensor[`${p}ln2.bias`] },
        fc: tensor[`${p}mlp.fc.weight`],
        mlpProj: tensor[`${p}mlp.proj.weight`],
        keys: new Float32Array(this.config.block_size * this.config.n_embd), // KV cache
        values: new Float32Array(this.config.block_size * this.config.n_embd),
      });
    }

    // Scratch buffers, reused on every step so generation allocates nothing.
    const C = this.config.n_embd;
    this.x = new Float32Array(C);
    this.normed = new Float32Array(C);
    this.qkvOut = new Float32Array(3 * C);
    this.attended = new Float32Array(C);
    this.projected = new Float32Array(C);
    this.hidden = new Float32Array(4 * C);
    this.scores = new Float32Array(this.config.block_size);
    this.length = 0; // how many tokens are in the cache
  }

  /** Fetch model.json and weights.bin from a folder URL. */
  static async load(baseUrl) {
    const [manifest, buffer] = await Promise.all([
      fetch(`${baseUrl}/model.json`).then((response) => {
        if (!response.ok) throw new Error(`Could not load ${baseUrl}/model.json (${response.status})`);
        return response.json();
      }),
      fetch(`${baseUrl}/weights.bin`).then((response) => {
        if (!response.ok) throw new Error(`Could not load ${baseUrl}/weights.bin (${response.status})`);
        return response.arrayBuffer();
      }),
    ]);
    return { model: new VakModel(manifest, buffer), manifest };
  }

  /** Forget everything read so far. */
  reset() {
    this.length = 0;
  }

  /**
   * Feed the next token. Returns the scores (logits) for the token after it:
   * one number per vocabulary entry, higher = more likely. The returned array
   * is freshly allocated, so callers may keep it.
   */
  step(tokenId) {
    const { n_embd: C, n_head: heads, block_size: blockSize, vocab_size: vocabSize } = this.config;
    const t = this.length;
    if (t >= blockSize) throw new Error(`The context is full (${blockSize} tokens). Call reset() first.`);
    const headDim = C / heads;
    const attentionScale = 1 / Math.sqrt(headDim);
    const x = this.x;

    // Token embedding (what the token means) + position embedding (where it is).
    for (let i = 0; i < C; i++) x[i] = this.tokEmb[tokenId * C + i] + this.posEmb[t * C + i];

    for (const layer of this.layers) {
      // ---- self-attention: gather information from this and earlier positions
      layerNorm(x, layer.ln1.gain, layer.ln1.bias, this.normed);
      const qkv = matVec(layer.qkv, this.normed, 3 * C, this.qkvOut); // [query | key | value]
      layer.keys.set(qkv.subarray(C, 2 * C), t * C);
      layer.values.set(qkv.subarray(2 * C), t * C);

      for (let h = 0; h < heads; h++) {
        const start = h * headDim;
        // How well does this position's query match each earlier position's key?
        let max = -Infinity;
        for (let j = 0; j <= t; j++) {
          let dot = 0;
          const keyBase = j * C + start;
          for (let d = 0; d < headDim; d++) dot += qkv[start + d] * layer.keys[keyBase + d];
          this.scores[j] = dot * attentionScale;
          if (this.scores[j] > max) max = this.scores[j];
        }
        // Softmax turns the scores into weights that sum to 1.
        let total = 0;
        for (let j = 0; j <= t; j++) {
          this.scores[j] = Math.exp(this.scores[j] - max);
          total += this.scores[j];
        }
        // Weighted average of the earlier positions' values.
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let j = 0; j <= t; j++) sum += this.scores[j] * layer.values[j * C + start + d];
          this.attended[start + d] = sum / total;
        }
      }
      matVec(layer.attnProj, this.attended, C, this.projected);
      for (let i = 0; i < C; i++) x[i] += this.projected[i]; // residual connection

      // ---- MLP: process what was gathered
      layerNorm(x, layer.ln2.gain, layer.ln2.bias, this.normed);
      matVec(layer.fc, this.normed, 4 * C, this.hidden);
      for (let i = 0; i < 4 * C; i++) this.hidden[i] = gelu(this.hidden[i]);
      matVec(layer.mlpProj, this.hidden, C, this.projected);
      for (let i = 0; i < C; i++) x[i] += this.projected[i]; // residual connection
    }

    this.length = t + 1;
    layerNorm(x, this.lnF.gain, this.lnF.bias, this.normed);
    return matVec(this.tokEmb, this.normed, vocabSize, new Float32Array(vocabSize));
  }
}
