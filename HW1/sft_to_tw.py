"""
convert_sft_to_zhtw.py

Converts MiniMind's general-purpose SFT corpus (sft_t2t_mini.jsonl, Simplified
Chinese, hundreds of thousands of examples) to Traditional Chinese with
OpenCC, for use as a Stage-1 "general instruction-following" SFT pass before
fine-tuning on the small, format-specific tmmluplus_sft.jsonl (Stage 2).

IMPORTANT CAVEAT (record this in data_manifest.md): OpenCC's s2t conversion
is CHARACTER-LEVEL only (software -> "軟件", not the Taiwan-usage "軟體";
network -> "網絡", not "網路"). It does NOT relocalize vocabulary/phrasing to
actual Taiwan usage the way zh-tw-wikipedia's source content does. Report
this accurately: this is "character-converted general SFT data for signal
volume", not "authentic Taiwan Traditional Chinese data".

Why streaming, line-by-line read+write: sft_t2t_mini.jsonl is ~1.7GB. Loading
it into a list before converting/writing (or building a big list of converted
lines before writing) risks the same kind of native OOM abort seen with the
wiki pretrain data. This script never holds more than one example in memory
at a time.

Why recursive string conversion instead of assuming a fixed schema: the
exact JSON shape of sft_t2t_mini.jsonl (conversations list vs some other
key) isn't hardcoded here since it wasn't inspected directly -- every string
value found anywhere in each parsed JSON record is converted in place, keys
and structure untouched. Non-Chinese strings (role names like "user"/
"assistant", punctuation-only tokens) pass through OpenCC unchanged, so this
is safe regardless of the exact schema.

Sampling: full-corpus conversion is a lot of extra SFT training time on top
of an already-large pretrain corpus. --sample_frac or --sample_n let you
convert only a subset. --sample_n does one cheap line-count pass first (O(1)
memory) so the kept fraction is calibrated to hit that count on average;
either way sampling is a streaming per-line Bernoulli draw, not a reservoir
buffer, to keep memory flat.

Usage:
    python3 sft_to_TC.py \
        --in_path minimind/dataset/sft_t2t_mini.jsonl \
        --out_path minimind/dataset/sft_t2t_mini_zhtw.jsonl \
        --seed 824

    # or an absolute target count instead of a fraction:
    python convert_sft_to_zhtw.py \
        --in_path ../dataset/sft_t2t_mini.jsonl \
        --out_path ../dataset/sft_t2t_mini_zhtw.jsonl \
        --sample_n 150000 \
        --seed 824

Remember for the two-stage SFT plan: train Stage 1 on this file with
`--save_weight sft_general --from_weight pretrain`, then Stage 2 on
tmmluplus_sft.jsonl with `--save_weight full_sft --from_weight sft_general`
(full_sft is the name convert_model.py expects, so Stage 2 must keep it).
"""

import argparse
import json
import random
import sys


def convert_value(value, converter):
    """Recursively convert every string found in a JSON value, leaving
    structure (dict keys, list order, non-string types) untouched."""
    if isinstance(value, str):
        return converter.convert(value)
    if isinstance(value, list):
        return [convert_value(v, converter) for v in value]
    if isinstance(value, dict):
        return {k: convert_value(v, converter) for k, v in value.items()}
    return value


def count_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Convert MiniMind's Simplified-Chinese SFT corpus to Traditional Chinese via OpenCC (streaming, memory-safe)")
    parser.add_argument("--in_path", type=str, required=True, help="source jsonl (e.g. ../dataset/sft_t2t_mini.jsonl)")
    parser.add_argument("--out_path", type=str, required=True, help="output jsonl path")
    parser.add_argument("--sample_frac", type=float, default=None, help="keep roughly this fraction of lines (0-1). Mutually exclusive with --sample_n.")
    parser.add_argument("--sample_n", type=int, default=None, help="keep roughly this many lines total (does one cheap line-count pass first to calibrate). Mutually exclusive with --sample_frac.")
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling")
    parser.add_argument("--report_every", type=int, default=20000, help="progress print interval, in lines read")
    args = parser.parse_args()

    if args.sample_frac is not None and args.sample_n is not None:
        sys.exit("--sample_frac and --sample_n are mutually exclusive, set at most one.")

    try:
        import opencc
    except ImportError:
        sys.exit(
            "opencc-python-reimplemented not installed. Install with:\n"
            "  pip install opencc-python-reimplemented --break-system-packages"
        )

    converter = opencc.OpenCC("s2t")  # Simplified -> Traditional (character-level, see module docstring)
    rng = random.Random(args.seed)

    keep_frac = 1.0
    if args.sample_frac is not None:
        keep_frac = args.sample_frac
    elif args.sample_n is not None:
        print("Counting source lines to calibrate sampling ...", file=sys.stderr)
        total_lines = count_lines(args.in_path)
        keep_frac = min(1.0, args.sample_n / total_lines) if total_lines else 1.0
        print(f"  {total_lines} lines total, sampling fraction={keep_frac:.4f} to target ~{args.sample_n}", file=sys.stderr)

    n_read = 0
    n_kept = 0
    n_bad_json = 0
    written_bytes = 0

    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_read += 1

            if keep_frac < 1.0 and rng.random() >= keep_frac:
                if n_read % args.report_every == 0:
                    print(f"  read {n_read}, kept {n_kept} ...", file=sys.stderr)
                continue

            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue

            converted = convert_value(record, converter)
            out_line = json.dumps(converted, ensure_ascii=False) + "\n"
            fout.write(out_line)
            written_bytes += len(out_line.encode("utf-8"))
            n_kept += 1

            if n_read % args.report_every == 0:
                print(f"  read {n_read}, kept {n_kept}, {written_bytes / 1024 / 1024:.1f} MB written so far ...", file=sys.stderr)

    print("Done.", file=sys.stderr)
    print(f"  lines read: {n_read}", file=sys.stderr)
    print(f"  lines kept/converted: {n_kept}", file=sys.stderr)
    print(f"  lines skipped (bad JSON): {n_bad_json}", file=sys.stderr)
    print(f"  output size: {written_bytes / 1024 / 1024:.1f} MB -> {args.out_path}", file=sys.stderr)
    print(
        "\nRecord in data_manifest.md: source=MiniMind sft_t2t_mini.jsonl, converted "
        "Simplified->Traditional via opencc-python-reimplemented (s2t, CHARACTER-LEVEL "
        "only -- not relocalized Taiwan vocabulary/phrasing), seed, sample_frac/sample_n.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()