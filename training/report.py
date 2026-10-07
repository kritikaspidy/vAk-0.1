"""Step 4: compare the models you have trained.

    python training/report.py                    # every run in runs/
    python training/report.py --runs tiny small  # only these

For each run this script:
  1. scores the saved model on the same held-out stories, with the same
     context length for every model, so the numbers are directly comparable
  2. writes two sample stories from fixed prompts and a fixed random seed
  3. reads the training log

and then writes
    results/loss_curves.png   validation loss during training, one line per model
    results/RESULTS.md        comparison table, the chart, and the sample stories

Perplexity is e^loss. Roughly: how many tokens the model is still undecided
between at each step. A model guessing at random scores the vocabulary size.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to a file; no window needed
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

from gpt import pick_device
from sample import load_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

PROMPTS = ["Once upon a time", "Lily found a"]
SAMPLE_TOKENS = 120

# Colours: a colourblind-checked categorical palette, used in this fixed order.
# A run keeps its colour whichever other runs are on the chart.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FIXED_SLOTS = {"tiny": 0, "small": 1, "base": 2}
SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def read_log(path: Path) -> list[dict]:
    """Rows of log.csv as numbers. Elapsed time restarts when a run is resumed, so it is re-accumulated."""
    rows, offset, previous = [], 0.0, 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            elapsed = float(row["elapsed_s"])
            if elapsed < previous:
                offset += previous
            previous = elapsed
            rows.append({"step": int(row["step"]), "train_loss": float(row["train_loss"]),
                         "val_loss": float(row["val_loss"]), "minutes": (offset + elapsed) / 60})
    return rows


@torch.no_grad()
def exact_val_loss(model, val_data: np.ndarray, context: int, max_tokens: int, device: str) -> float:
    """Average loss over fixed, non-overlapping windows from the start of the validation set."""
    window = context + 1  # `context` inputs, each predicting the token after it
    count = min(len(val_data), max_tokens) // window
    windows = np.asarray(val_data[: count * window]).astype(np.int64).reshape(count, window)
    total = 0.0
    for start in range(0, count, 64):
        batch = torch.from_numpy(windows[start : start + 64]).to(device)
        _, loss = model(batch[:, :-1], batch[:, 1:])
        total += loss.item() * len(batch)
    return total / count


def write_samples(model, tokenizer, device: str) -> list[tuple[str, str]]:
    samples = []
    for prompt in PROMPTS:
        torch.manual_seed(0)  # same seed for every model, so differences come from the model
        start = torch.tensor([tokenizer.encode(prompt)], device=device)
        out = model.generate(start, SAMPLE_TOKENS, temperature=0.8, top_k=40, stop_token=tokenizer.eot_id)[0].tolist()
        text = tokenizer.decode(out).replace("<|endoftext|>", "").strip()
        samples.append((prompt, text))
    return samples


def assign_colours(names: list[str]) -> dict[str, str]:
    free = [i for i in range(len(PALETTE)) if i not in {FIXED_SLOTS[n] for n in names if n in FIXED_SLOTS}]
    colours = {}
    for name in names:
        colours[name] = PALETTE[FIXED_SLOTS[name]] if name in FIXED_SLOTS else PALETTE[free.pop(0)]
    return colours


def plot_curves(runs: list[dict], path: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY, "axes.labelcolor": INK_SECONDARY,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    panels = [("step", "Training step", lambda v, _: f"{v:,.0f}"), ("minutes", "Minutes of training", lambda v, _: f"{v:g}")]

    for ax, (key, label, formatter) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for run in runs:
            xs = [row[key] for row in run["log"]]
            ys = [row["val_loss"] for row in run["log"]]
            ax.plot(xs, ys, color=run["colour"], linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                    label=f"{run['name']} ({run['parameters'] / 1e6:.2f}M parameters)")
            # End dot with a ring in the surface colour so it stays readable where lines cross.
            ax.plot(xs[-1], ys[-1], "o", color=run["colour"], markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_xlabel(label)
        ax.xaxis.set_major_formatter(FuncFormatter(formatter))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
        ax.margins(x=0.04)
        ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.08)  # room for the end labels

    axes[0].set_ylabel("Validation loss")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.76, bottom=0.13, wspace=0.06)

    # Label each line's final value, but only where the labels would not overlap;
    # otherwise the legend and the table in RESULTS.md carry the numbers.
    fig.canvas.draw()
    for ax, (key, _, _) in zip(axes, panels):
        ends = [(run["log"][-1][key], run["log"][-1]["val_loss"]) for run in runs]
        pixels = sorted(ax.transData.transform(end)[1] for end in ends)
        if all(b - a >= 16 * fig.dpi / 100 for a, b in zip(pixels, pixels[1:])):
            for x, y in ends:
                label = ax.annotate(f"{y:.2f}", (x, y), xytext=(8, 0), textcoords="offset points",
                                    va="center", color=INK, fontsize=9.5)
                # A halo in the surface colour keeps the number readable if another line passes behind it.
                label.set_path_effects([patheffects.withStroke(linewidth=3, foreground=SURFACE)])

    handles, labels = axes[0].get_legend_handles_labels()
    subtitle = "Lower is better. Measured on stories the model never saw during training."
    if len(runs) > 1:
        title = "Validation loss by model size"
        subtitle = subtitle.replace("the model never", "the models never")
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.062, 0.86), ncol=min(len(runs), 4),
                   frameon=False, labelcolor=INK_SECONDARY, handlelength=1.6, columnspacing=1.8, fontsize=9.5)
    else:
        title = f"Validation loss during training: {labels[0]}"  # one line needs no legend; the title names it
    fig.text(0.07, 0.94, title, fontsize=14, fontweight="bold", color=INK, va="center")
    fig.text(0.07, 0.885, subtitle, fontsize=10, color=INK_SECONDARY, va="center")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def write_markdown(runs: list[dict], meta: dict, context: int, eval_tokens: int, path: Path) -> None:
    lines = [
        "# Results",
        "",
        "Generated by `python training/report.py`. Re-run it after training another model; do not edit by hand.",
        "",
        "## Setup",
        "",
        f"- Training data: {meta['train_tokens'] / 1e6:.1f}M tokens of {meta['dataset']}, vocabulary of {meta['vocab_size']:,} tokens.",
        f"- Every model is scored on the same {eval_tokens:,} held-out tokens, using {context} tokens of context.",
        f"- Perplexity is e^loss. A model guessing at random would score {meta['vocab_size']:,}.",
        "",
        "## Comparison",
        "",
        "| Model | Parameters | Layers x heads x width | Steps | Training time | Validation loss | Perplexity | Loss at full context |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        shape = run["model"]
        lines.append(
            f"| `{run['name']}` | {run['parameters'] / 1e6:.2f}M | {shape['n_layer']} x {shape['n_head']} x {shape['n_embd']} "
            f"| {run['step']:,} | {max(1, round(run['log'][-1]['minutes']))} min | {run['val_loss']:.3f} | {math.exp(run['val_loss']):.1f} "
            f"| {run['val_loss_full']:.3f} ({shape['block_size']} tokens) |"
        )
    lines += [
        "",
        f"\"Validation loss\" and \"Perplexity\" give every model the same {context} tokens of context, so they compare like with like. "
        "\"Loss at full context\" lets each model use its whole context window; a longer window gives a trained model "
        "more of the story to go on, which usually lowers the loss.",
        "",
        "![Validation loss during training](loss_curves.png)",
        "",
        "The chart shows the loss logged while training: each model at its own full context, estimated from a random "
        "sample of batches. Its end points therefore sit near the \"Loss at full context\" column, not the \"Validation loss\" column.",
        "",
    ]

    if len(runs) > 1:
        smallest, best = runs[0], min(runs, key=lambda r: r["val_loss"])
        if best is not smallest:
            lines += [
                f"`{best['name']}` has {best['parameters'] / smallest['parameters']:.1f}x the parameters of `{smallest['name']}` "
                f"and lowers perplexity from {math.exp(smallest['val_loss']):.1f} to {math.exp(best['val_loss']):.1f}.",
                "",
            ]

    lines += ["## Sample stories", "", "Same prompts and same random seed for every model (temperature 0.8, top-k 40).", ""]
    for prompt_index, prompt in enumerate(PROMPTS):
        lines += [f"### Prompt: \"{prompt}\"", ""]
        for run in runs:
            story = run["samples"][prompt_index][1].replace("\n\n", "\n").replace("\n", "\n> ")
            lines += [f"**`{run['name']}`**", "", f"> {story}", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", help="run names to compare (default: every trained run)")
    parser.add_argument("--eval-tokens", type=int, default=400_000, help="held-out tokens to score each model on")
    parser.add_argument("--device", help="mps, cuda or cpu (default: best available)")
    args = parser.parse_args()

    names = args.runs or sorted(p.parent.name for p in (ROOT / "runs").glob("*/ckpt.pt"))
    missing = [name for name in names if not (ROOT / "runs" / name / "ckpt.pt").exists()]
    if not names or missing:
        sys.exit(f"No trained model for: {', '.join(missing)}" if missing
                 else "No trained models yet. Run this first:  python training/train.py")
    if len(names) > len(PALETTE):
        sys.exit(f"The chart compares at most {len(PALETTE)} runs. Choose some with --runs.")

    device = args.device or pick_device()
    val_data = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    meta = json.loads((DATA / "meta.json").read_text())
    colours = assign_colours(names)

    loaded = {name: load_model(ROOT / "runs" / name, device) for name in names}
    # Score every model with the shortest context any of them supports, so the comparison is fair.
    context = min(model.config.block_size for model, _, _ in loaded.values())
    eval_tokens = min(len(val_data), args.eval_tokens) // (context + 1) * (context + 1)

    runs = []
    for name in names:
        model, tokenizer, ckpt = loaded[name]
        if len(tokenizer) != meta["vocab_size"]:
            sys.exit(f"runs/{name} was trained with a different tokenizer from the one in data/. Retrain it or leave it out.")
        print(f"Scoring {name}...", flush=True)
        runs.append({
            "name": name, "colour": colours[name], "step": ckpt["step"],
            "parameters": model.num_parameters(), "model": model.config_dict(),
            "val_loss": exact_val_loss(model, val_data, context, args.eval_tokens, device),
            "val_loss_full": exact_val_loss(model, val_data, model.config.block_size, args.eval_tokens, device),
            "samples": write_samples(model, tokenizer, device),
            "log": read_log(ROOT / "runs" / name / "log.csv"),
        })
    runs.sort(key=lambda run: run["parameters"])

    RESULTS.mkdir(exist_ok=True)
    plot_curves(runs, RESULTS / "loss_curves.png")
    write_markdown(runs, meta, context, eval_tokens, RESULTS / "RESULTS.md")

    print(f"\n{'model':<12}{'parameters':>12}{'val loss':>10}{'perplexity':>12}")
    for run in runs:
        print(f"{run['name']:<12}{run['parameters'] / 1e6:>11.2f}M{run['val_loss']:>10.3f}{math.exp(run['val_loss']):>12.1f}")
    print("\nWrote results/RESULTS.md and results/loss_curves.png")


if __name__ == "__main__":
    main()
