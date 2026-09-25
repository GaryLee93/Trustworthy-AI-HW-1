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

Why chunk articles: MiniMind's train_pretrain.py truncates every jsonl
line to --max_seq_len tokens (340 by default, ~500-580 Chinese chars).
Wikipedia articles run far longer than that (up to ~170K chars in this
dataset's markdown column), so writing one line per article would mean
the trainer silently drops everything past the opening paragraph of
almost every article -- target_mb would count bytes that never actually
reach the model. Each cleaned article is instead split into <=max_chars
chunks at paragraph, then sentence, boundaries, and each chunk becomes
its own jsonl line.

License: Wikipedia content is CC BY-SA -- fine for this assignment's
"any other publicly available data" clause, but record the source,
snapshot date (May 2023, per the dataset card), and license in your
data_manifest.md.

Usage:
    python prepare_zhtw_wikipedia_pretrain.py \
        --target_mb 1024 \
        --seed 42 \
        --min_length 100 \
        --max_chars 500 \
        --out_path ../dataset/zhtw_wikipedia_pretrain.jsonl
"""

import argparse
import json
import os
import re
import sys

from datasets import load_dataset

DATASET_NAME = "zetavg/zh-tw-wikipedia"

# Columns present in this dataset besides "markdown" (see the dataset card:
# pageid, html, markdown, coordinate, length, touched, lastrevid,
# original_title). "html" alone can be up to 432KB per row. We drop these
# BEFORE shuffling: datasets' streaming shuffle buffer holds full rows, so
# without this, a buffer_size=10000 shuffle can balloon to multiple GB and
# crash with a native OOM abort (no Python traceback, just "core dumped").
_COLUMNS_TO_DROP = ["pageid", "html", "coordinate", "length", "touched", "lastrevid", "original_title"]

_HEADER_RE = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)")
_UNDERSCORE_RE = re.compile(r"_(.*?)_")
_BLANK_LINES_RE = re.compile(r"\n{2,}")

# Sentence-ending punctuation used as fallback cut points for paragraphs
# that alone exceed max_chars (long articles' intro paragraphs especially).
_SENTENCE_END_RE = re.compile(r"[。！？；\n]")


def markdown_to_plain(md):
    """Strip common Markdown syntax, leaving plain prose."""
    text = _HEADER_RE.sub("", md)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _UNDERSCORE_RE.sub(r"\1", text)
    text = _BLANK_LINES_RE.sub("\n", text)
    return text.strip()


def _split_long_paragraph(paragraph, max_chars):
    """Cut a single paragraph that alone exceeds max_chars, breaking at the
    nearest sentence-ending punctuation at or before max_chars instead of
    mid-sentence. Falls back to a hard cut if no punctuation is found."""
    pieces = []
    remaining = paragraph
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        matches = list(_SENTENCE_END_RE.finditer(window))
        cut = matches[-1].end() if matches else max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:]
    if remaining.strip():
        pieces.append(remaining.strip())
    return pieces


def chunk_article(text, max_chars, min_length):
    """Split one cleaned article into MiniMind-sized chunks instead of
    writing the whole article as a single pretrain line.

    Without this, PretrainDataset truncates each row to max_seq_len tokens
    (~max_chars characters), silently discarding everything after the
    opening paragraph of every article -- the model would only ever see
    article introductions, never body content, and target_mb would count
    bytes that never actually reach the model.

    Chunks are built by greedily packing whole paragraphs up to max_chars;
    a paragraph that alone exceeds max_chars is further split at sentence
    boundaries (see _split_long_paragraph) rather than cut mid-sentence.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        candidates = [para] if len(para) <= max_chars else _split_long_paragraph(para, max_chars)
        for piece in candidates:
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chars:
                current += "\n" + piece
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) >= min_length]


def main():
    parser = argparse.ArgumentParser(description="Stream zh-tw Wikipedia into a size-capped MiniMind pretrain jsonl")
    parser.add_argument("--target_mb", type=float, default=1024, help="stop once approximately this many MB of text have been written")
    parser.add_argument("--seed", type=int, default=42, help="seed for the streaming shuffle buffer")
    parser.add_argument("--buffer_size", type=int, default=2000, help="shuffle buffer size for streaming, in ROWS (datasets library shuffles a rolling window, not a true global shuffle). Kept modest since even markdown-only rows can be up to ~170KB each.")
    parser.add_argument("--min_length", type=int, default=100, help="skip cleaned chunks shorter than this many characters (drops stubs / tail fragments)")
    parser.add_argument("--max_chars", type=int, default=500, help="target max characters per output line, chosen to fit MiniMind pretrain's max_seq_len=340 tokens (~1.5-1.7 chars/token). Long articles are split into multiple lines at paragraph/sentence boundaries instead of being truncated by the trainer.")
    parser.add_argument("--out_path", type=str, default="../dataset/zhtw_wikipedia_pretrain.jsonl", help="output jsonl path")
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024 * 1024

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Streaming {DATASET_NAME} (train split), shuffle buffer={args.buffer_size}, seed={args.seed} ...", file=sys.stderr)
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    ds = ds.remove_columns([c for c in _COLUMNS_TO_DROP if c in ds.column_names])
    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    written_bytes = 0
    n_written = 0
    n_skipped_short = 0

    n_articles_seen = 0
    reached_target = False

    with open(args.out_path, "w", encoding="utf-8") as f:
        for row in ds:
            n_articles_seen += 1
            text = markdown_to_plain(row["markdown"])
            chunks = chunk_article(text, args.max_chars, args.min_length)
            if not chunks:
                n_skipped_short += 1
                continue

            for chunk in chunks:
                line = json.dumps({"text": chunk}, ensure_ascii=False) + "\n"
                line_bytes = len(line.encode("utf-8"))

                f.write(line)
                written_bytes += line_bytes
                n_written += 1

                if n_written % 5000 == 0:
                    print(f"  {n_written} chunks written, {written_bytes / 1024 / 1024:.1f} MB so far ...", file=sys.stderr)

                if written_bytes >= target_bytes:
                    reached_target = True
                    break
            if reached_target:
                break

    print("Done.", file=sys.stderr)
    print(f"  articles read: {n_articles_seen}", file=sys.stderr)
    print(f"  chunks written: {n_written} (max_chars={args.max_chars})", file=sys.stderr)
    print(f"  articles skipped (no chunk >= min_length after cleaning): {n_skipped_short}", file=sys.stderr)
    print(f"  output size: {written_bytes / 1024 / 1024:.1f} MB -> {args.out_path}", file=sys.stderr)
    print(
        "\nRecord in data_manifest.md: source=zetavg/zh-tw-wikipedia (CC BY-SA, "
        "snapshot ~May 2023), seed, buffer_size, target_mb, min_length.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()