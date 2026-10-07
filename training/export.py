"""Step 5: export a trained model so it can run in the browser.

    python training/export.py                 # the most recently trained run
    python training/export.py --run small
    python training/export.py --run tiny --dtype float32

Writes three files to web/models/<run>/:
    weights.bin   every weight matrix, one after another, as raw numbers
    model.json    the model's shape, where each matrix sits in weights.bin, and the tokenizer
    parity.json   reference answers computed here with PyTorch

The browser code (web/js/) re-implements the tokenizer and the transformer in
plain JavaScript. parity.json lets it prove it computes the same thing as
PyTorch: same token ids, same output scores, same text when always picking
the most likely token.

Weights are stored as 16-bit floats by default, which halves the download.
The reference answers are computed with the same rounded weights, so the
JavaScript result must match them closely whichever precision is used.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from gpt import GPT, GPTConfig
from sample import load_model
from tokenizer import EOT

ROOT = Path(__file__).resolve().parent.parent
WEB_MODELS = ROOT / "web" / "models"

PARITY_PROMPT = "Once upon a time, there was a little"
GREEDY_TOKENS = 40
TOKENIZER_CASES = [
    "Once upon a time",
    "Lily found a tiny dragon in the garden.",
    'She said, "Let\'s play!" and they laughed.',
    "Tom had 3 red balls.\n\nThe end.",
    "“Curly quotes” and a dash — like this…",
    "  two  spaces and unknown café 中 characters",
    f"The end.{EOT}Once more",
    "",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run folder under runs/ (default: the most recently trained)")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="precision of the saved weights")
    args = parser.parse_args()

    checkpoints = sorted((ROOT / "runs").glob("*/ckpt.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        sys.exit("No trained model yet. Run this first:  python training/train.py")
    run_dir = ROOT / "runs" / args.run if args.run else checkpoints[-1].parent
    if not (run_dir / "ckpt.pt").exists():
        sys.exit(f"No trained model in runs/{args.run}. Available runs: {', '.join(p.parent.name for p in checkpoints)}")

    # The CPU is used so the reference numbers are plain, exact float32 arithmetic.
    model, tokenizer, ckpt = load_model(run_dir, "cpu")
    numpy_dtype = "<f2" if args.dtype == "float16" else "<f4"  # "<" = little-endian, what browsers expect

    # 1. Weights. The output layer shares the embedding matrix, so it is not stored twice.
    tensors, chunks, offset = [], [], 0
    rounded_state = {}
    for name, tensor in model.state_dict().items():
        if name == "head.weight":
            continue
        values = tensor.numpy().astype(numpy_dtype)
        chunks.append(values.tobytes())
        tensors.append({"name": name, "shape": list(tensor.shape), "offset": offset})
        offset += values.size
        rounded_state[name] = torch.from_numpy(values.astype(np.float32))
    rounded_state["head.weight"] = rounded_state["tok_emb.weight"]

    out_dir = WEB_MODELS / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(b"".join(chunks))

    # 2. A copy of the model that holds exactly the numbers the browser will load.
    exported = GPT(GPTConfig(**ckpt["config"]))
    exported.load_state_dict(rounded_state)
    exported.eval()

    manifest = {
        "name": run_dir.name,
        "config": ckpt["config"],
        "parameters": model.num_parameters(),
        "step": ckpt["step"],
        "val_loss": round(ckpt["val_loss"], 4),
        "dtype": args.dtype,
        "tensors": tensors,
        "tokenizer": {"vocab": tokenizer.vocab, "merges": tokenizer.merges},
    }
    (out_dir / "model.json").write_text(json.dumps(manifest, ensure_ascii=True))

    # 3. Reference answers from PyTorch.
    prompt_ids = tokenizer.encode(PARITY_PROMPT)
    with torch.no_grad():
        idx = torch.tensor([prompt_ids])
        logits = exported(idx)[0][0, -1]
        original_logits = model(idx)[0][0, -1]
        greedy = list(prompt_ids)
        for _ in range(GREEDY_TOKENS):  # always pick the single most likely token
            context = torch.tensor([greedy[-exported.config.block_size:]])
            greedy.append(int(exported(context)[0][0, -1].argmax()))
    parity = {
        "tokenizer_cases": [{"text": text, "ids": tokenizer.encode(text)} for text in TOKENIZER_CASES],
        "prompt": PARITY_PROMPT,
        "prompt_ids": prompt_ids,
        "logits": [round(float(value), 5) for value in logits],
        "greedy_ids": greedy,
        "greedy_text": tokenizer.decode(greedy),
    }
    (out_dir / "parity.json").write_text(json.dumps(parity, ensure_ascii=True))

    # 4. The list of models the web page offers.
    listing = []
    for path in sorted(WEB_MODELS.glob("*/model.json")):
        info = json.loads(path.read_text())
        listing.append({"name": info["name"], "parameters": info["parameters"], "val_loss": info["val_loss"],
                        "megabytes": round((path.parent / "weights.bin").stat().st_size / 1e6, 1)})
    listing.sort(key=lambda item: item["parameters"])
    (WEB_MODELS / "index.json").write_text(json.dumps(listing, indent=2) + "\n")

    size_mb = (out_dir / "weights.bin").stat().st_size / 1e6
    drift = float((logits - original_logits).abs().max())
    print(f"Exported runs/{run_dir.name} to web/models/{run_dir.name}/  ({size_mb:.1f} MB, {args.dtype})")
    if args.dtype == "float16":
        print(f"Rounding to 16-bit changed the output scores by at most {drift:.4f} (they range over about "
              f"{float(original_logits.max() - original_logits.min()):.0f}).")
    print(f"Reference text: {parity['greedy_text']!r}")
    print("\nNext: python serve.py    then open http://localhost:8080")


if __name__ == "__main__":
    main()
