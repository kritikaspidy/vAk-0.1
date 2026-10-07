"""Step 2: train the model.

    python training/train.py                     # the "small" model
    python training/train.py --preset tiny       # quickest, to check everything works
    python training/train.py --preset base --run-name base-v1

Training repeats one loop:
    1. take a random batch of text from train.bin
    2. ask the model to predict every next token
    3. measure how wrong it was (the loss)
    4. nudge every weight a little in the direction that lowers the loss

Every --eval-every steps it measures the loss on held-out text (val.bin),
writes a row to runs/<name>/log.csv, prints a sample story, and saves the
model to runs/<name>/ckpt.pt whenever the validation loss is the best so far.
"""
import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gpt import GPT, GPTConfig, PRESETS, pick_device
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_ITERS = {"tiny": 3000, "small": 6000, "base": 10000}


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    """Pick random windows of text. x is a window; y is the same window shifted by one token."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
    x = np.stack([data[s : s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1 : s + 1 + block_size] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def estimate_loss(model: GPT, data: np.ndarray, batch_size: int, device: str, batches: int) -> float:
    """Average loss over several random batches (one batch alone is too noisy)."""
    model.eval()
    losses = [model(*get_batch(data, model.config.block_size, batch_size, device))[1].item() for _ in range(batches)]
    model.train()
    return sum(losses) / len(losses)


def learning_rate(step: int, max_iters: int, base_lr: float, warmup: int) -> float:
    """Ramp up for `warmup` steps, then follow a cosine curve down to 10% of base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_iters - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=PRESETS, default="small", help="model size (default small)")
    parser.add_argument("--run-name", help="folder name under runs/ (default: the preset name)")
    parser.add_argument("--max-iters", type=int, help="training steps (default depends on the preset)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", help="mps, cuda or cpu (default: best available)")
    parser.add_argument("--resume", action="store_true", help="continue from this run's saved checkpoint")
    args = parser.parse_args()

    if not (DATA / "train.bin").exists():
        sys.exit("data/train.bin not found. Run this first:  python training/prepare_data.py")

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    max_iters = args.max_iters or DEFAULT_ITERS[args.preset]
    run_dir = ROOT / "runs" / (args.run_name or args.preset)
    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists() and not args.resume:
        sys.exit(f"{run_dir.relative_to(ROOT)} already has a trained model.\n"
                 "Add --resume to continue it, or --run-name <new-name> to keep it and start another.")
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.load(DATA / "tokenizer.json")
    shutil.copy(DATA / "tokenizer.json", run_dir / "tokenizer.json")  # each run is self-contained
    train_data = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")

    config = GPTConfig(vocab_size=len(tokenizer), dropout=args.dropout, **PRESETS[args.preset])
    if min(len(train_data), len(val_data)) <= config.block_size + 1:
        sys.exit("The dataset is smaller than one context window. Prepare more data.")
    model = GPT(config).to(device)

    # Weight decay (a pull towards zero) is applied to weight matrices only,
    # not to LayerNorm gains and biases.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    start_step, best_val = 0, float("inf")
    if args.resume:
        if not ckpt_path.exists():
            sys.exit(f"Nothing to resume: {ckpt_path.relative_to(ROOT)} does not exist.")
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt["config"] != model.config_dict():
            sys.exit("The saved model has a different shape from this preset. Use the same --preset as before.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt["val_loss"]
        print(f"Resuming from step {start_step} (validation loss {best_val:.3f})")

    settings = {**vars(args), "max_iters": max_iters, "device": device, "parameters": model.num_parameters(),
                "model": model.config_dict(), "train_tokens": len(train_data)}
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")

    print(f"Model: {args.preset}, {model.num_parameters() / 1e6:.2f}M parameters | device: {device} | "
          f"{max_iters} steps of {args.batch_size} x {config.block_size} tokens")
    print(f"A model that guesses at random would have loss {math.log(config.vocab_size):.2f}. Lower is better.\n")

    log_path = run_dir / "log.csv"
    if not (args.resume and log_path.exists()):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])

    prompt = torch.tensor([tokenizer.encode("Once upon a time")], device=device)
    started = time.time()

    def evaluate(step: int) -> None:
        nonlocal best_val
        train_loss = estimate_loss(model, train_data, args.batch_size, device, args.eval_batches)
        val_loss = estimate_loss(model, val_data, args.batch_size, device, args.eval_batches)
        elapsed = time.time() - started
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}",
                                    f"{optimizer.param_groups[0]['lr']:.6f}", f"{elapsed:.0f}"])
        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": model.config_dict(),
                        "step": step, "val_loss": val_loss}, ckpt_path)
            saved = "  (saved)"
        print(f"step {step:>6} | train loss {train_loss:.3f} | val loss {val_loss:.3f}{saved}")
        model.eval()
        sample = model.generate(prompt, max_new_tokens=40, stop_token=tokenizer.eot_id)[0].tolist()
        model.train()
        print(f"            sample: {tokenizer.decode(sample)!r}\n")

    model.train()
    step = start_step
    try:
        for step in range(start_step, max_iters):
            if step % args.eval_every == 0 and step > start_step:
                evaluate(step)

            lr = learning_rate(step, max_iters, args.lr, args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_data, config.block_size, args.batch_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # work out how each weight affected the loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid occasional huge updates
            optimizer.step()  # nudge the weights

            done = step - start_step + 1
            if done == 20 or done % 100 == 0:
                rate = done / (time.time() - started)
                remaining = (max_iters - step - 1) / rate / 60
                print(f"step {step + 1:>6} | loss {loss.item():.3f} | {rate:.1f} steps/s | about {remaining:.0f} min left",
                      flush=True)
        step = max_iters
    except KeyboardInterrupt:
        print("\nStopped early (Ctrl+C).")

    evaluate(step)
    print(f"Best validation loss: {best_val:.3f}. Model saved in {run_dir.relative_to(ROOT)}/")
    print(f"Next: python training/sample.py --run {run_dir.name}")


if __name__ == "__main__":
    main()
