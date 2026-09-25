"""
prepare_tmmluplus_sft.py

Converts TMMLU+ (train/dev + validation minus the stratified holdout) into
MiniMind's SFT jsonl format:

    {"conversations": [{"role": "user", "content": "..."},
                        {"role": "assistant", "content": "..."}]}

The user/assistant content is built from the *actual* official prompt
template imported directly from eval_tmmluplus.py (INSTRUCTION,
ANSWER_PREFIX, format_question, SUBJECTS, LABELS) -- not reimplemented --
so there is zero risk of the training format drifting from the real
evaluation format as that script gets updated.

Mirrors eval_tmmluplus.py's build_prompt() chat-mode branch exactly:
  user content    = INSTRUCTION + question + options   (everything up to,
                     but NOT including, the trailing "答案：")
  assistant content = "答案：" + correct_letter

This is the chat_template=True alignment. Whether the actual grading run
uses --chat_template or not is not something this script can know -- test
both on your local holdout with eval_tmmluplus.py and report the
difference, exactly as HW1's own "Useful options" table suggests.

IMPORTANT: run this from the same directory as (or with the same
sys.path as) eval_tmmluplus.py, since it's imported directly.

IMPORTANT: uses the same seed + stratified-holdout algorithm as
prepare_tmmluplus_pretrain.py, so with the same --seed and --holdout_frac
the holdout set here is identical to that script's holdout -- the same
questions are excluded from training in both.

Usage:
    python prepare_tmmluplus_sft.py \
        --seed 42 --holdout_frac 0.2 --revision v1.1 \
        --out_path ../dataset/tmmluplus_sft.jsonl
"""

import argparse
import json
import random
import sys

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

# Import the REAL template so this can never drift from the grading script.
try:
    from eval_tmmluplus import SUBJECTS, INSTRUCTION, ANSWER_PREFIX, format_question, LABELS
except ImportError:
    sys.exit(
        "Could not import eval_tmmluplus.py. Run this script from the same "
        "directory as eval_tmmluplus.py (or add it to PYTHONPATH)."
    )

DATASET_NAME = "ikala/tmmluplus"
DATA_DIR_PREFIX = "data/"


def clean_value(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def discover_subjects(revision):
    api = HfApi()
    files = api.list_repo_files(DATASET_NAME, repo_type="dataset", revision=revision)
    train_files = [f for f in files if f.startswith(DATA_DIR_PREFIX) and f.endswith("_train.csv")]
    return sorted(f[len(DATA_DIR_PREFIX):-len("_train.csv")] for f in train_files)


def load_all_subjects(revision):
    """Same train/val-only loading as prepare_tmmluplus_pretrain.py -- test
    is never requested. See that script's docstring for the rationale."""
    subjects = discover_subjects(revision)
    per_subject = {}
    for i, subj in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] loading subject: {subj}", file=sys.stderr)
        train_path = hf_hub_download(DATASET_NAME, filename=f"{DATA_DIR_PREFIX}{subj}_train.csv", repo_type="dataset", revision=revision)
        val_path = hf_hub_download(DATASET_NAME, filename=f"{DATA_DIR_PREFIX}{subj}_val.csv", repo_type="dataset", revision=revision)
        per_subject[subj] = {
            "train": pd.read_csv(train_path).to_dict("records"),
            "validation": pd.read_csv(val_path).to_dict("records"),
        }
    return per_subject


def stratified_holdout_split(per_subject, holdout_frac, seed):
    """Identical algorithm to prepare_tmmluplus_pretrain.py -- same seed
    and holdout_frac reproduce the exact same holdout set."""
    rng = random.Random(seed)
    train_pool, val_train_pool, val_holdout_pool = [], [], []
    for subj, splits in per_subject.items():
        train_pool.extend((subj, ex) for ex in splits["train"])
        val = list(splits["validation"])
        rng.shuffle(val)
        n_holdout = max(1, round(len(val) * holdout_frac)) if val else 0
        val_holdout_pool.extend((subj, ex) for ex in val[:n_holdout])
        val_train_pool.extend((subj, ex) for ex in val[n_holdout:])
    return train_pool, val_train_pool, val_holdout_pool


def to_conversation(subj, ex):
    """Build one SFT example matching eval_tmmluplus.py's chat-mode prompt
    exactly: user turn = INSTRUCTION + question + options (no trailing
    ANSWER_PREFIX); assistant turn = ANSWER_PREFIX + correct letter."""
    row = {
        "subject": subj,
        "question": clean_value(ex["question"]),
        "A": clean_value(ex["A"]),
        "B": clean_value(ex["B"]),
        "C": clean_value(ex["C"]),
        "D": clean_value(ex["D"]),
        "answer": clean_value(ex["answer"]),
    }
    if row["answer"] not in LABELS:
        return None  # skip malformed rows rather than crash mid-run

    subject_zh = SUBJECTS.get(subj, (subj, ""))[0]
    body = INSTRUCTION.format(subject_zh=subject_zh) + format_question(row, with_answer=False)
    user_content = body[: -len(ANSWER_PREFIX)].rstrip()
    assistant_content = ANSWER_PREFIX + row["answer"]

    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def write_holdout_jsonl(pool, path):
    """Same shape as prepare_tmmluplus_pretrain.py's holdout file: full raw
    fields plus subject, never converted to conversations, never trained on."""
    with open(path, "w", encoding="utf-8") as f:
        for subj, ex in pool:
            record = {
                "subject": subj,
                "question": clean_value(ex["question"]),
                "A": clean_value(ex["A"]),
                "B": clean_value(ex["B"]),
                "C": clean_value(ex["C"]),
                "D": clean_value(ex["D"]),
                "answer": clean_value(ex["answer"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare TMMLU+ SFT data matching the official eval prompt exactly")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--revision", type=str, default="v1.1")
    parser.add_argument("--out_path", type=str, default="../dataset/tmmluplus_sft.jsonl")
    parser.add_argument("--holdout_out_path", type=str, default=None, help="optional: also write the holdout set here (same shape as prepare_tmmluplus_pretrain.py's holdout file) -- useful as a cross-check that both scripts compute the identical holdout under the same --seed/--holdout_frac")
    args = parser.parse_args()

    print(f"Loading {DATASET_NAME} (revision={args.revision}), train+validation only ...", file=sys.stderr)
    per_subject = load_all_subjects(args.revision)

    print(f"Splitting validation into {args.holdout_frac:.0%} holdout / rest, seed={args.seed} ...", file=sys.stderr)
    train_pool, val_train_pool, val_holdout_pool = stratified_holdout_split(per_subject, args.holdout_frac, args.seed)

    n_written, n_skipped = 0, 0
    with open(args.out_path, "w", encoding="utf-8") as f:
        for subj, ex in train_pool + val_train_pool:
            record = to_conversation(subj, ex)
            if record is None:
                n_skipped += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    if args.holdout_out_path:
        write_holdout_jsonl(val_holdout_pool, args.holdout_out_path)

    print("Done.", file=sys.stderr)
    print(f"  SFT examples written: {n_written}", file=sys.stderr)
    print(f"  skipped (malformed answer): {n_skipped}", file=sys.stderr)
    print(f"  holdout size (never in {args.out_path}): {len(val_holdout_pool)}", file=sys.stderr)
    print(f"  -> {args.out_path}", file=sys.stderr)
    if args.holdout_out_path:
        print(f"  -> holdout also written to {args.holdout_out_path}", file=sys.stderr)
        print(
            "     Diff this against tmmluplus_local_eval_holdout.jsonl from "
            "prepare_tmmluplus_pretrain.py (same --seed/--holdout_frac) to "
            "confirm both scripts picked the identical holdout set.",
            file=sys.stderr,
        )
    print(
        "\nThis matches the --chat_template=True path exactly. Also test "
        "--chat_template off on your holdout with eval_tmmluplus.py and "
        "compare -- record both numbers in your report.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()