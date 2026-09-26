"""
diagnose_sft_truncation.py

Checks whether MiniMind's SFT training is losing the assistant's answer
letter to truncation, and whether the ground-truth answer letters are
skewed -- the two things that would produce "stuck at ~0.25 accuracy,
model almost always answers A" even when training "succeeded" with no
errors.

For each conversation in a MiniMind SFT jsonl ({"conversations": [...]})
this reconstructs the full chat-formatted text with the REAL tokenizer's
chat template (the same one SFTDataset will tokenize), and reports:

  1. What fraction of conversations exceed --max_seq_len tokens in total
     (these get truncated somewhere).
  2. Of those, what fraction lose PART OR ALL of the assistant turn
     specifically -- i.e. the "答案：X" tokens fall past max_seq_len and
     never receive a loss signal at all. This assumes tail truncation
     (keep tokens[:max_seq_len]), which is what most from-scratch
     tokenize-and-truncate implementations do; if your SFTDataset
     truncates from the front instead, invert this reasoning.
  3. The distribution of correct-answer letters (A/B/C/D) in the file --
     if this is already skewed toward "A", a model that learns nothing
     more than the marginal answer distribution would score close to
     that skew, which can look identical to "random guessing" if the
     skew happens to be near-uniform.

Usage:
    python diagnose_sft_truncation.py \
        --file ../dataset/tmmluplus_sft.jsonl \
        --tokenizer_path ../model \
        --max_seq_len 768 \
        --sample 3000
"""

import argparse
import json
import os
import random
import sys
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Check SFT truncation and answer-letter skew")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="../model")
    parser.add_argument("--max_seq_len", type=int, default=768, help="must match --max_seq_len used in train_full_sft.py")
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers not installed here -- run this on your training machine.")

    if not os.path.isdir(args.tokenizer_path):
        sys.exit(f"tokenizer_path '{args.tokenizer_path}' not found -- point --tokenizer_path at "
                  "MiniMind's ../model directory (the same one init_model() loads).")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.chat_template is None:
        print("WARNING: this tokenizer has no chat_template configured. MiniMind's "
              "SFTDataset must be building the chat format some other way (manual "
              "special tokens?) -- the length check below may not match what actually "
              "happens during training. Share dataset/lm_dataset.py's SFTDataset if "
              "you want this checked precisely.", file=sys.stderr)

    with open(args.file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(args.seed)
    sample_lines = rng.sample(lines, min(args.sample, len(lines)))

    total_len_over = 0
    assistant_partially_or_fully_lost = 0
    n_checked = 0
    answer_counter = Counter()

    for line in sample_lines:
        try:
            convo = json.loads(line)["conversations"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if len(convo) != 2:
            continue
        user_msg, assistant_msg = convo[0], convo[1]

        # answer-letter distribution: last non-space char of the assistant turn
        ans = assistant_msg["content"].strip()[-1:]
        if ans in ("A", "B", "C", "D"):
            answer_counter[ans] += 1

        if tokenizer.chat_template is None:
            continue  # can't do the length check without a chat template

        n_checked += 1
        full_text = tokenizer.apply_chat_template(convo, tokenize=False)
        prefix_text = tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)

        full_len = len(tokenizer(full_text)["input_ids"])
        prefix_len = len(tokenizer(prefix_text)["input_ids"])

        if full_len > args.max_seq_len:
            total_len_over += 1
            # tail-truncation assumption: tokens[:max_seq_len] keeps the prefix but
            # cuts into (or entirely past) where the assistant turn starts
            if prefix_len >= args.max_seq_len:
                assistant_partially_or_fully_lost += 1  # answer completely gone
            else:
                assistant_partially_or_fully_lost += 1  # answer at least partially cut

    print(f"=== {args.file} ===")
    print(f"  sampled: {len(sample_lines)} lines")

    if n_checked:
        print(f"\n--- truncation check (assumes tail truncation, max_seq_len={args.max_seq_len}) ---")
        print(f"  conversations exceeding max_seq_len: {total_len_over}/{n_checked} ({100*total_len_over/n_checked:.1f}%)")
        print(f"  of those, assistant answer at least partially cut off: "
              f"{assistant_partially_or_fully_lost}/{n_checked} ({100*assistant_partially_or_fully_lost/n_checked:.1f}%)")
        if n_checked and assistant_partially_or_fully_lost / n_checked > 0.02:
            print("  -> this is likely your bug: a meaningful fraction of training examples "
                  "never actually supervise the answer token. Try raising --max_seq_len, or "
                  "check whether SFTDataset truncates from the front instead (which would "
                  "protect the answer but cut the question instead -- also bad, differently).")
    else:
        print("\n--- truncation check skipped (no chat_template on this tokenizer) ---")

    print(f"\n--- answer-letter distribution (ground truth, n={sum(answer_counter.values())}) ---")
    total = sum(answer_counter.values()) or 1
    for letter in ("A", "B", "C", "D"):
        c = answer_counter.get(letter, 0)
        print(f"  {letter}: {c:5d}  ({100*c/total:.1f}%)")
    max_frac = max(answer_counter.values(), default=0) / total
    if max_frac > 0.30:
        print(f"  -> noticeably skewed (max class = {100*max_frac:.1f}%). A model that learned "
              f"nothing but this marginal distribution would score close to {100*max_frac:.1f}%, "
              "not necessarily exactly 25%.")
    else:
        print("  -> roughly balanced. If the model still always predicts one letter, that's "
              "consistent with the model learning the output FORMAT but not the content -> "
              "points back to the truncation/chat_template/loss-mask checks above, not to "
              "data skew.")


if __name__ == "__main__":
    main()