"""Step 3: write stories with a trained model.

    python training/sample.py
    python training/sample.py --prompt "Lily found a tiny dragon" --temperature 0.6
    python training/sample.py --run tiny --num 3      # a specific run instead of the newest

--temperature controls randomness: near 0 always picks the likeliest next
token (safe, repetitive); 1.0 samples exactly as the model predicts (creative,
more mistakes). --top-k limits each choice to the k most likely tokens.
"""
import argparse
import sys
from pathlib import Path

import torch

from gpt import GPT, GPTConfig, pick_device
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def load_model(run_dir: Path, device: str) -> tuple[GPT, Tokenizer, dict]:
    """Load a trained model and its tokenizer from a run folder."""
    ckpt = torch.load(run_dir / "ckpt.pt", map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, Tokenizer.load(run_dir / "tokenizer.json"), ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run folder under runs/ (default: the most recently trained)")
    parser.add_argument("--prompt", default="Once upon a time", help="how the story starts")
    parser.add_argument("--tokens", type=int, default=250, help="maximum tokens to write")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num", type=int, default=1, help="how many stories")
    parser.add_argument("--seed", type=int, help="set for repeatable output")
    parser.add_argument("--device", help="mps, cuda or cpu (default: best available)")
    args = parser.parse_args()

    checkpoints = sorted((ROOT / "runs").glob("*/ckpt.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        sys.exit("No trained model yet. Run this first:  python training/train.py")
    run_dir = ROOT / "runs" / args.run if args.run else checkpoints[-1].parent
    if not (run_dir / "ckpt.pt").exists():
        runs = ", ".join(path.parent.name for path in checkpoints)
        sys.exit(f"No trained model in runs/{args.run}. Available runs: {runs}")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device or pick_device()
    model, tokenizer, ckpt = load_model(run_dir, device)
    print(f"runs/{run_dir.name}: {model.num_parameters() / 1e6:.2f}M parameters, "
          f"step {ckpt['step']}, validation loss {ckpt['val_loss']:.3f}\n")

    ids = tokenizer.encode(args.prompt) or [tokenizer.eot_id]  # empty prompt: start a fresh story
    for _ in range(args.num):
        start = torch.tensor([ids], device=device)
        out = model.generate(start, args.tokens, temperature=args.temperature, top_k=args.top_k,
                             stop_token=tokenizer.eot_id)[0].tolist()
        if out[-1] == tokenizer.eot_id:
            out = out[:-1]
        print(tokenizer.decode(out).replace("<|endoftext|>", "").strip())
        print("-" * 60)


if __name__ == "__main__":
    main()
