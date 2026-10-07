"""Step 1: download the stories, train the tokenizer, and turn the text into token ids.

    python training/prepare_data.py

Dataset: TinyStories (Eldan & Li, 2023), short stories written with the
vocabulary of a young child. The limited vocabulary is what lets a model with
only a few million parameters learn to write text that makes sense.
https://huggingface.co/datasets/roneneldan/TinyStories

Output (in data/):
    tokenizer.json   the learned vocabulary
    train.bin        training text as uint16 token ids
    val.bin          held-out text, used only to measure the model
    meta.json        sizes and statistics
"""
import argparse
import array
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from tokenizer import EOT, Tokenizer, normalise

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"  # 2.2 GB; only the first --train-mb are downloaded
VAL_FILE = "TinyStoriesV2-GPT4-valid.txt"  # 22 MB

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MB = 1024 * 1024

ALLOWED = {chr(code) for code in range(32, 127)} | {"\n"}  # printable ASCII and newlines


def download(name: str, dest: Path, max_mb: float) -> None:
    """Download the first `max_mb` megabytes of a dataset file (skipped if already there)."""
    if dest.exists():
        print(f"  {dest.name} already downloaded ({dest.stat().st_size / MB:.0f} MB)")
        return
    limit = int(max_mb * MB)
    request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "vak-prepare/0.1"})
    partial = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as out:
            done = 0
            while done < limit:
                block = response.read(min(MB, limit - done))
                if not block:
                    break
                out.write(block)
                done += len(block)
                print(f"\r  downloading {name}: {done / MB:.0f} MB", end="", flush=True)
        print()
    except OSError as error:
        partial.unlink(missing_ok=True)
        sys.exit(
            f"\nCould not download {name}: {error}\n"
            f"Check your internet connection. You can also download the file in a browser from\n"
            f"  {BASE_URL}{name}\nand pass it with --train-file / --val-file."
        )
    partial.rename(dest)


def load_stories(path: Path, max_mb: float) -> list[str]:
    """Read a raw file and return its cleaned stories."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = f.read(int(max_mb * MB))
    pieces = raw.split(EOT)
    if len(pieces) > 1:
        pieces = pieces[:-1]  # the last piece was probably cut mid-story by the size limit
    stories = []
    for piece in pieces:
        story = re.sub(r"[ \t]+\n", "\n", normalise(piece)).strip()
        story = re.sub(r"\n{3,}", "\n\n", story)
        # Skip fragments and the few stories with stray non-English characters.
        if len(story) >= 100 and all(char in ALLOWED for char in story):
            stories.append(story)
    return stories


def encode_to_file(tokenizer: Tokenizer, stories: list[str], dest: Path) -> int:
    """Encode stories as  story <EOT> story <EOT> ...  and save them as uint16. Returns the token count."""
    ids = array.array("H")  # unsigned 16-bit: 2 bytes per token instead of a Python int each
    for i, story in enumerate(stories):
        ids.extend(tokenizer.encode(story))
        ids.append(tokenizer.eot_id)
        if i % 5000 == 0:
            print(f"\r  encoding {dest.name}: {i:,}/{len(stories):,} stories", end="", flush=True)
    print(f"\r  encoding {dest.name}: {len(stories):,}/{len(stories):,} stories")
    with open(dest, "wb") as f:
        ids.tofile(f)
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-mb", type=float, default=100, help="MB of training text to use (default 100)")
    parser.add_argument("--vocab-size", type=int, default=2048, help="tokenizer vocabulary size (default 2048)")
    parser.add_argument("--tokenizer-mb", type=float, default=5, help="MB of text the tokenizer learns from")
    parser.add_argument("--train-file", type=Path, help="use this local text file instead of downloading")
    parser.add_argument("--val-file", type=Path, help="use this local text file instead of downloading")
    args = parser.parse_args()
    if not 256 <= args.vocab_size <= 65536:
        sys.exit("--vocab-size must be between 256 and 65536 (token ids are stored as 16-bit numbers)")

    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("1/3 Getting the stories")
    train_path = args.train_file or raw / f"train_{args.train_mb:g}mb.txt"
    val_path = args.val_file or raw / "valid.txt"
    if not args.train_file:
        download(TRAIN_FILE, train_path, args.train_mb)
    if not args.val_file:
        download(VAL_FILE, val_path, 64)
    train_stories = load_stories(train_path, args.train_mb)
    val_stories = load_stories(val_path, 10)  # 10 MB is plenty for measuring
    if not train_stories or not val_stories:
        sys.exit(f"No usable stories found. Stories must be separated by {EOT}")
    train_chars = sum(len(s) for s in train_stories)
    print(f"  {len(train_stories):,} training stories ({train_chars / MB:.0f} MB), {len(val_stories):,} validation stories")

    print(f"2/3 Training the tokenizer (vocabulary of {args.vocab_size})")
    sample, size = [], 0
    for story in train_stories:
        sample.append(story)
        size += len(story)
        if size >= args.tokenizer_mb * MB:
            break
    tokenizer = Tokenizer.train(EOT.join(sample), args.vocab_size, verbose=True)
    tokenizer.save(DATA / "tokenizer.json")
    print(f"  learned {len(tokenizer):,} tokens, saved to data/tokenizer.json")

    print("3/3 Encoding the text into token ids")
    train_tokens = encode_to_file(tokenizer, train_stories, DATA / "train.bin")
    val_tokens = encode_to_file(tokenizer, val_stories, DATA / "val.bin")

    meta = {
        "dataset": "TinyStories (V2, GPT-4 subset)" if not args.train_file else str(args.train_file),
        "vocab_size": len(tokenizer),
        "train_stories": len(train_stories),
        "val_stories": len(val_stories),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "chars_per_token": round(train_chars / train_tokens, 2),
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nDone in {time.time() - started:.0f}s: {train_tokens:,} training tokens, {val_tokens:,} validation tokens, "
          f"{meta['chars_per_token']} characters per token.")
    example = train_stories[0][:60]
    print(f"Example: {example!r}\n      -> {[tokenizer.vocab[i] for i in tokenizer.encode(example)]}")
    print("\nNext: python training/train.py")


if __name__ == "__main__":
    main()
