"""
check_sft_label_mask.py

The truncation check (1.1%) and answer-letter skew check (~25% each) both
came back clean, which rules out the two most common surface-level bugs.
Reading SFTDataset.generate_labels() (dataset/lm_dataset.py) surfaces a
different, more severe failure mode it's structurally prone to:

    self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n',
                             add_special_tokens=False).input_ids

is computed by tokenizing that literal string IN ISOLATION, then
generate_labels() searches for that exact token-id subsequence inside the
full chat-formatted input_ids to find where each assistant turn starts.
If the tokenizer's BPE merges that text differently when it's embedded in
the full prompt (a common issue right at whitespace/newline boundaries)
than it does in isolation, the substring search silently never matches --
labels stays [-100] * len(input_ids) for that entire example, meaning
ZERO gradient signal, for the ENTIRE conversation, not just the truncated
tail. If this happens for most/all examples, "SFT looks like it ran fine
but the model never actually learned anything and just regressed to
whatever pretrain biased it toward" is exactly what you'd observe --
matching "stuck at ~0.24, always answers A" far better than the
truncation or skew hypotheses already ruled out.

This script instantiates the REAL SFTDataset class from your repo (not a
reimplementation) and reports, for a sample of examples:
  - what fraction have ZERO supervised (non -100) label positions at all
    -- these contribute nothing to training, ever
  - for examples that DO have supervised positions, decodes them so you
    can visually confirm they're actually "答案：X<eos>..." and not noise

IMPORTANT: run this from your trainer/ directory (or wherever
train_full_sft.py lives), OR pass --repo_root pointing at the minimind
repo root that contains dataset/lm_dataset.py, since this imports
SFTDataset directly rather than reimplementing its logic.

Usage:
    python3 ../../check_sft.py \
        --file tmmluplus_sft.jsonl \
        --tokenizer_path ../../minimind-3 \
        --repo_root .. \
        --max_seq_len 768 \
        --sample 500
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Check SFTDataset's real label mask for silently all-masked examples")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="../model")
    parser.add_argument("--repo_root", type=str, default="..", help="path to the minimind repo root containing dataset/lm_dataset.py")
    parser.add_argument("--max_seq_len", type=int, default=768, help="must match --max_seq_len used in train_full_sft.py")
    parser.add_argument("--sample", type=int, default=500, help="how many examples (from the start of the dataset) to check")
    parser.add_argument("--show_decoded", type=int, default=3, help="how many supervised spans to print decoded, for a sanity look")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo_root))
    try:
        from dataset.lm_dataset import SFTDataset
    except ImportError as e:
        sys.exit(f"Could not import SFTDataset from {args.repo_root}/dataset/lm_dataset.py: {e}\n"
                  "Pass --repo_root pointing at your minimind repo root (the directory that "
                  "contains the dataset/ folder).")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers not installed here -- run this on your training machine.")

    if not os.path.isdir(args.tokenizer_path):
        sys.exit(f"tokenizer_path '{args.tokenizer_path}' not found.")

    print(f"Loading tokenizer from {args.tokenizer_path} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"Building SFTDataset from {args.file} (max_length={args.max_seq_len}) ...", file=sys.stderr)
    ds = SFTDataset(args.file, tokenizer, max_length=args.max_seq_len)
    print(f"  dataset size: {len(ds)}", file=sys.stderr)

    n = min(args.sample, len(ds))
    n_zero_signal = 0
    n_shown = 0
    supervised_lengths = []

    for idx in range(n):
        input_ids, labels = ds[idx]
        supervised_positions = [j for j, l in enumerate(labels.tolist()) if l != -100]

        if not supervised_positions:
            n_zero_signal += 1
            continue

        supervised_lengths.append(len(supervised_positions))

        if n_shown < args.show_decoded:
            n_shown += 1
            decoded = tokenizer.decode([labels[j].item() for j in supervised_positions])
            print(f"--- example {idx}: {len(supervised_positions)} supervised tokens ---")
            print(f"  decoded supervised span: {decoded!r}")

    print()
    print(f"=== results over {n} examples ===")
    print(f"  examples with ZERO supervised tokens (labels all -100, contributes NO gradient): "
          f"{n_zero_signal}/{n} ({100*n_zero_signal/n:.1f}%)")
    if n_zero_signal / n > 0.05:
        print("  -> THIS IS LIKELY YOUR BUG. A meaningful fraction of training examples never")
        print("     supervise anything at all -- the bos_id/eos_id marker search in")
        print("     generate_labels() is failing to find the assistant-turn boundary for them.")
        print("     Likely cause: tokenizing f'{bos_token}assistant\\n' IN ISOLATION produces a")
        print("     different token-id sequence than the same text produces embedded in the full")
        print("     chat-formatted prompt (BPE merge context-sensitivity, often at the newline).")
        print("     Try: print(self.bos_id) and print the actual input_ids around where the")
        print("     assistant turn should start, for one all--100 example, to see exactly where")
        print("     the token sequences diverge.")
    else:
        print("  -> low. Most examples DO get a supervised span; the label-mask boundary search")
        print("     is working for the large majority. If the model still collapses to one")
        print("     letter, look at the decoded spans above (are they really '答案：X...'?), the")
        print("     eval-time chat_template consistency check, or training loss curve /")
        print("     hyperparameters / model capacity next.")

    if supervised_lengths:
        avg_len = sum(supervised_lengths) / len(supervised_lengths)
        print(f"\n  among examples WITH a supervised span: avg length = {avg_len:.1f} tokens "
              f"(min={min(supervised_lengths)}, max={max(supervised_lengths)})")


if __name__ == "__main__":
    main()