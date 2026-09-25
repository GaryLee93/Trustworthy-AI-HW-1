"""
prepare_zhtw_wikipedia_pretrain.py

Streams zetavg/zh-tw-wikipedia (a public, genuinely Traditional Chinese
-- Taiwan usage -- Wikipedia dump, 2.53M pages, 8.19GB total across all
columns) and writes a size-capped MiniMind pretrain jsonl:
{"text": "..."} per line.

Why streaming: the dataset is 8.19GB across all columns (pageid, html,
markdown, coordinate, length, touched, lastrevid, original_title). We only
need "markdown", and only up to a target output size, so streaming=True
lets us pull shards on demand instead of caching the whole 8.19GB dataset
locally first.

Why clean the markdown: the "markdown" column still contains Markdown
syntax (**bold**, ## headers, etc.), which would show up as literal
symbol noise in a next-token-prediction pretrain corpus if left in.

License: Wikipedia content is CC BY-SA -- fine for this assignment's
"any other publicly available data" clause, but record the source,
snapshot date (May 2023, per the dataset card), and license in your
data_manifest.md.

Usage:
    python prepare_zhtw_wikipedia_pretrain.py \
        --target_mb 1024 \
        --seed 42 \
        --min_length 100 \
        --out_path ../dataset/zhtw_wikipedia_pretrain.jsonl
"""

import argparse
import json
import re
import sys

from datasets import load_dataset

DATASET_NAME = "zetavg/zh-tw-wikipedia"

_HEADER_RE = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)")
_UNDERSCORE_RE = re.compile(r"_(.*?)_")
_BLANK_LINES_RE = re.compile(r"\n{2,}")


def markdown_to_plain(md):
    """Strip common Markdown syntax, leaving plain prose."""
    text = _HEADER_RE.sub("", md)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _UNDERSCORE_RE.sub(r"\1", text)
    text = _BLANK_LINES_RE.sub("\n", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Stream zh-tw Wikipedia into a size-capped MiniMind pretrain jsonl")
    parser.add_argument("--target_mb", type=float, default=1024, help="stop once approximately this many MB of text have been written")
    parser.add_argument("--seed", type=int, default=42, help="seed for the streaming shuffle buffer")
    parser.add_argument("--buffer_size", type=int, default=10000, help="shuffle buffer size for streaming (datasets library shuffles a rolling window, not a true global shuffle)")
    parser.add_argument("--min_length", type=int, default=100, help="skip cleaned articles shorter than this many characters (drops stubs)")
    parser.add_argument("--out_path", type=str, default="../dataset/zhtw_wikipedia_pretrain.jsonl", help="output jsonl path")
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024 * 1024

    print(f"Streaming {DATASET_NAME} (train split), shuffle buffer={args.buffer_size}, seed={args.seed} ...", file=sys.stderr)
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    written_bytes = 0
    n_written = 0
    n_skipped_short = 0

    with open(args.out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = markdown_to_plain(row["markdown"])
            if len(text) < args.min_length:
                n_skipped_short += 1
                continue

            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))

            f.write(line)
            written_bytes += line_bytes
            n_written += 1

            if n_written % 5000 == 0:
                print(f"  {n_written} articles written, {written_bytes / 1024 / 1024:.1f} MB so far ...", file=sys.stderr)

            if written_bytes >= target_bytes:
                break

    print("Done.", file=sys.stderr)
    print(f"  articles written: {n_written}", file=sys.stderr)
    print(f"  articles skipped (too short after cleaning): {n_skipped_short}", file=sys.stderr)
    print(f"  output size: {written_bytes / 1024 / 1024:.1f} MB -> {args.out_path}", file=sys.stderr)
    print(
        "\nRecord in data_manifest.md: source=zetavg/zh-tw-wikipedia (CC BY-SA, "
        "snapshot ~May 2023), seed, buffer_size, target_mb, min_length.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()