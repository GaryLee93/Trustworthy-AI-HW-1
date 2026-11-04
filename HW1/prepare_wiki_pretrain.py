"""
prepare_zhtw_wikipedia_pretrain.py

Streams zetavg/zh-tw-wikipedia (a public, genuinely Traditional Chinese
-- Taiwan usage -- Wikipedia dump, 2.53M pages, 8.19GB total across all
columns) and writes a size-capped MiniMind pretrain jsonl:
{"text": "..."} per line.

MEMORY FIX (see below): this version reads the dataset's own auto-converted
parquet shards directly with pyarrow, requesting ONLY the "markdown" column
at decode time (true column pruning). The previous version used
datasets.load_dataset(..., streaming=True) + .remove_columns(...): that
pipeline decodes every column of a row (including "html", up to ~432KB/row)
into memory BEFORE the column drop is applied, so a shuffle buffer of
buffer_size rows could genuinely hold gigabytes of never-used html text --
this is what was hitting a native malloc abort ("core dumped", no Python
traceback) rather than a catchable MemoryError. Reading with
pyarrow.parquet + columns=["markdown"] means html/coordinate/etc are never
decoded at all, on top of never being buffered.

Why chunk articles: MiniMind's train_pretrain.py truncates every jsonl
line to --max_seq_len tokens. Wikipedia articles run far longer than that
(up to ~170K chars in this dataset's markdown column), so writing one line
per article would mean the trainer silently drops everything past the
opening paragraph of almost every article -- target_mb would count bytes
that never actually reach the model. Each cleaned article is instead split
into <=max_chars chunks at paragraph, then sentence, boundaries, and each
chunk becomes its own jsonl line.

Why "shuffle by shard order" instead of a rolling shuffle buffer: with
column pruning, each shard is downloaded and processed whole (only its
markdown column, so far smaller than before) rather than row-streamed from
a live HTTP connection, so a global rolling shuffle buffer doesn't apply
the same way. Randomness instead comes from (a) shuffling the order the 44
shards are visited in, seeded by --seed, and (b) shuffling row order
within each shard's decoded batch before writing. This is coarser than a
true global shuffle but bounded, deterministic, and -- most importantly --
doesn't reintroduce the memory problem.

License: Wikipedia content is CC BY-SA -- fine for this assignment's
"any other publicly available data" clause, but record the source,
snapshot date (per the dataset card), and license in your data_manifest.md.

Usage:
    python prepare_zhtw_wikipedia_pretrain.py \
        --target_mb 1024 \
        --seed 42 \
        --min_length 100 \
        --max_chars 400 \
        --out_path ../dataset/zhtw_wikipedia_pretrain.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import tempfile
import urllib.request

import pyarrow.parquet as pq

DATASET_NAME = "zetavg/zh-tw-wikipedia"
N_SHARDS = 44  # confirmed via https://huggingface.co/api/datasets/zetavg/zh-tw-wikipedia/parquet
SHARD_URL_TMPL = (
    "https://huggingface.co/api/datasets/zetavg/zh-tw-wikipedia/parquet/default/train/{index}.parquet"
)

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


def _download_shard(index, scratch_dir):
    url = SHARD_URL_TMPL.format(index=index)
    fd, tmp_path = tempfile.mkstemp(suffix=f".shard{index}.parquet", dir=scratch_dir)
    os.close(fd)
    print(f"  downloading shard {index} ...", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp_path)
    return tmp_path


def _iter_markdown_rows(shard_path, batch_size, rng):
    """Yield markdown strings from one parquet shard, reading ONLY the
    markdown column (true column pruning -- html/coordinate/etc are never
    decoded), in row-batches to keep peak memory bounded, with row order
    shuffled within each batch for some local randomization."""
    pf = pq.ParquetFile(shard_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["markdown"]):
        rows = batch.column("markdown").to_pylist()
        rng.shuffle(rows)
        for md in rows:
            if md:
                yield md


def main():
    parser = argparse.ArgumentParser(description="Stream zh-tw Wikipedia (column-pruned) into a size-capped MiniMind pretrain jsonl")
    parser.add_argument("--target_mb", type=float, default=1024, help="stop once approximately this many MB of text have been written")
    parser.add_argument("--seed", type=int, default=42, help="seed for shard-order and within-batch row shuffling")
    parser.add_argument("--batch_size", type=int, default=512, help="rows per pyarrow read batch (bounds transient memory per shard read)")
    parser.add_argument("--min_length", type=int, default=100, help="skip cleaned chunks shorter than this many characters (drops stubs / tail fragments)")
    parser.add_argument("--max_chars", type=int, default=400, help="target max characters per output line. Calibrated against the REAL measured tokenizer ratio (~1.19 tokens/char for Traditional Chinese, see check_tokenizer_zhtw.py), not an assumed chars/token ratio -- keep this in sync with whatever --max_seq_len you train pretrain with.")
    parser.add_argument("--out_path", type=str, default="../dataset/zhtw_wikipedia_pretrain.jsonl", help="output jsonl path")
    parser.add_argument("--scratch_dir", type=str, default=None, help="where to download shard parquet files temporarily (default: system temp dir). Each shard is ~180MB and is deleted right after it's processed.")
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024 * 1024

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.scratch_dir:
        os.makedirs(args.scratch_dir, exist_ok=True)

    rng = random.Random(args.seed)
    shard_order = list(range(N_SHARDS))
    rng.shuffle(shard_order)

    print(f"Reading {DATASET_NAME} via column-pruned parquet ({N_SHARDS} shards, order seeded by --seed={args.seed}) ...", file=sys.stderr)

    written_bytes = 0
    n_written = 0
    n_skipped_short = 0
    n_articles_seen = 0
    reached_target = False

    with open(args.out_path, "w", encoding="utf-8") as f:
        for shard_idx in shard_order:
            if reached_target:
                break

            shard_path = _download_shard(shard_idx, args.scratch_dir)
            try:
                for md in _iter_markdown_rows(shard_path, args.batch_size, rng):
                    n_articles_seen += 1
                    text = markdown_to_plain(md)
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
            finally:
                # Delete the shard as soon as we're done with it (or as soon as
                # we've hit target and are bailing out), regardless of outcome --
                # this is what keeps disk usage bounded to ~1 shard at a time.
                os.remove(shard_path)

    print("Done.", file=sys.stderr)
    print(f"  articles read: {n_articles_seen}", file=sys.stderr)
    print(f"  chunks written: {n_written} (max_chars={args.max_chars})", file=sys.stderr)
    print(f"  articles skipped (no chunk >= min_length after cleaning): {n_skipped_short}", file=sys.stderr)
    print(f"  output size: {written_bytes / 1024 / 1024:.1f} MB -> {args.out_path}", file=sys.stderr)
    print(
        "\nRecord in data_manifest.md: source=zetavg/zh-tw-wikipedia (CC BY-SA), "
        "seed, batch_size, target_mb, min_length, max_chars.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()