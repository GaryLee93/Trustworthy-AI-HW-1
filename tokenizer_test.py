"""
check_tokenizer_zhtw.py

Answers the Task 2 question directly: "Inspect how the MiniMind tokenizer
handles Traditional Chinese characters. Is this a bottleneck?"

Method: take a batch of real text, convert the SAME content to both
Traditional and Simplified Chinese with OpenCC (a rule-based converter, not
a model -- deterministic, no contamination risk), tokenize both versions
with the MiniMind tokenizer, and compare tokens-per-character. If Traditional
text needs meaningfully MORE tokens per character than the Simplified
version of the identical content, that's direct evidence the tokenizer
vocab under-serves Traditional characters -- each TMMLU+ question then eats
more of your --max_seq_len budget than it would in Simplified, translating
directly into more silent truncation and less effective context for a task
that's already token-budget-constrained (see the earlier truncation checks
in this repo's dataset/*.jsonl).

Secondary check: vocabulary coverage. For each unique Traditional character
across the sample, checks whether the tokenizer represents it as a single
token (efficient) or splits it into multiple sub-character tokens / falls
back to UNK (inefficient, and for UNK, actively lossy).

Usage:
    python3 tokenizer_test.py \
        --tokenizer_path minimind/model \
        --text_file minimind/dataset/zhtw_wikipedia_pretrain.jsonl \
        --sample 500
"""

import argparse
import json
import random
import sys
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Compare MiniMind tokenizer efficiency on Traditional vs Simplified Chinese")
    parser.add_argument("--tokenizer_path", type=str, default="../model")
    parser.add_argument("--text_file", type=str, required=True, help="a jsonl with a 'text' field (e.g. your zhtw wikipedia pretrain file, or tmmluplus_sft.jsonl's user turns)")
    parser.add_argument("--text_field", type=str, default="text", help="which field holds the text; use 'conversations' handling automatically if the file looks like SFT data")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers not installed here -- run this on your training machine.")

    try:
        import opencc
    except ImportError:
        sys.exit(
            "opencc-python-reimplemented not installed. Install with:\n"
            "  pip install opencc-python-reimplemented --break-system-packages\n"
            "This is a rule-based Traditional<->Simplified converter (not a model), so using it "
            "to build a comparison text pair carries no contamination risk under the assignment's rules."
        )

    import os
    if not os.path.isdir(args.tokenizer_path):
        sys.exit(f"tokenizer_path '{args.tokenizer_path}' not found.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    t2s = opencc.OpenCC('t2s')  # Traditional -> Simplified

    # Load a sample of Traditional Chinese text
    with open(args.text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(args.seed)
    sample_lines = rng.sample(lines, min(args.sample, len(lines)))

    traditional_texts = []
    for line in sample_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "conversations" in record:
            # SFT-style file: use the user turn's content
            text = record["conversations"][0]["content"]
        else:
            text = record.get(args.text_field, "")
        if text.strip():
            traditional_texts.append(text)

    if not traditional_texts:
        sys.exit(f"No usable text found in {args.text_file} (looked for field '{args.text_field}' or 'conversations').")

    # === Tokens-per-character comparison ===
    trad_chars_total = 0
    trad_tokens_total = 0
    simp_chars_total = 0
    simp_tokens_total = 0
    per_example_ratios_trad = []
    per_example_ratios_simp = []

    for trad_text in traditional_texts:
        simp_text = t2s.convert(trad_text)

        trad_n_chars = len(trad_text)
        simp_n_chars = len(simp_text)
        trad_n_tokens = len(tokenizer(trad_text, add_special_tokens=False).input_ids)
        simp_n_tokens = len(tokenizer(simp_text, add_special_tokens=False).input_ids)

        trad_chars_total += trad_n_chars
        trad_tokens_total += trad_n_tokens
        simp_chars_total += simp_n_chars
        simp_tokens_total += simp_n_tokens

        if trad_n_chars > 0:
            per_example_ratios_trad.append(trad_n_tokens / trad_n_chars)
        if simp_n_chars > 0:
            per_example_ratios_simp.append(simp_n_tokens / simp_n_chars)

    trad_ratio = trad_tokens_total / trad_chars_total
    simp_ratio = simp_tokens_total / simp_chars_total

    print(f"=== Tokens-per-character: Traditional vs Simplified ===")
    print(f"  sample size: {len(traditional_texts)} texts, {trad_chars_total} Traditional chars total")
    print(f"  Traditional: {trad_tokens_total} tokens / {trad_chars_total} chars = {trad_ratio:.4f} tokens/char")
    print(f"  Simplified:  {simp_tokens_total} tokens / {simp_chars_total} chars = {simp_ratio:.4f} tokens/char")
    pct_worse = 100 * (trad_ratio - simp_ratio) / simp_ratio
    print(f"  Traditional uses {pct_worse:+.1f}% {'more' if pct_worse > 0 else 'fewer'} tokens per character than Simplified")
    if pct_worse > 5:
        print(f"  -> Meaningful gap. Every Traditional-Chinese question eats roughly {pct_worse:.0f}% more of your")
        print(f"     --max_seq_len budget than the same content would in Simplified. This directly worsens")
        print(f"     truncation risk and shrinks effective context for TMMLU+, which is already token-tight.")
    else:
        print(f"  -> Small gap. The tokenizer doesn't look like a major bottleneck by this measure alone.")

    # === Per-character vocabulary coverage ===
    unique_trad_chars = Counter()
    for t in traditional_texts:
        for ch in t:
            if '一' <= ch <= '鿿':  # CJK unified ideographs
                unique_trad_chars[ch] += 1

    multi_token_chars = []
    unk_chars = []
    unk_id = tokenizer.unk_token_id
    for ch, count in unique_trad_chars.most_common(2000):  # cap for speed
        ids = tokenizer(ch, add_special_tokens=False).input_ids
        if unk_id is not None and unk_id in ids:
            unk_chars.append((ch, count))
        elif len(ids) > 1:
            multi_token_chars.append((ch, count, len(ids)))

    n_checked = min(len(unique_trad_chars), 2000)
    print(f"\n=== Per-character vocabulary coverage (top {n_checked} most frequent Traditional chars in sample) ===")
    print(f"  chars needing >1 token: {len(multi_token_chars)}/{n_checked} ({100*len(multi_token_chars)/n_checked:.1f}%)")
    print(f"  chars falling back to UNK: {len(unk_chars)}/{n_checked} ({100*len(unk_chars)/n_checked:.1f}%)")
    if unk_chars:
        print(f"  most frequent UNK characters (char, occurrence count in sample):")
        for ch, count in sorted(unk_chars, key=lambda x: -x[1])[:15]:
            print(f"    {ch!r}: {count}")
    if multi_token_chars:
        print(f"  most frequent multi-token characters (char, occurrence count, token count):")
        for ch, count, n_tok in sorted(multi_token_chars, key=lambda x: -x[1])[:15]:
            print(f"    {ch!r}: seen {count}x, costs {n_tok} tokens")

    print()
    print("Summary for your report: if tokens/char is meaningfully higher for Traditional than")
    print("Simplified AND/OR a nontrivial fraction of common Traditional characters need multiple")
    print("tokens or hit UNK, that's direct evidence the tokenizer is a real bottleneck for this")
    print("task -- worth reporting even if you don't have time to retrain a custom tokenizer.")


if __name__ == "__main__":
    main()