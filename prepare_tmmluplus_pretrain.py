"""
prepare_tmmluplus_pretrain.py

Converts TMMLU+ (train/dev + validation splits) into MiniMind's pretrain
jsonl format: {"text": "..."} per line, using the template:

    關於「問題敘述」，正確的答案是：[正確答案]。

Rules enforced (per HW1 data policy):
- Only the "train" (dev) and "validation" splits of TMMLU+ are touched.
  The "test" split is never downloaded or referenced.
- Before any training data is produced, a subject-stratified 20% holdout
  is carved out of "validation" and written to a separate file. That
  holdout is NEVER included in the pretrain output — it's for your own
  local evaluation only.
- A fixed random seed is used and recorded, so the split is reproducible
  by run.sh / teaching staff.

Usage:
    python prepare_tmmluplus_pretrain.py \
        --seed 42 \
        --holdout_frac 0.2 \
        --revision v1.1 \
        --out_dir ../dataset

Outputs (written to --out_dir):
    tmmluplus_local_eval_holdout.jsonl   <- never train on this
    tmmluplus_pretrain_declarative.jsonl <- safe to mix into pretraining

To actually train on it, concatenate with the provided MiniMind data:
    cat pretrain_t2t_mini.jsonl tmmluplus_pretrain_declarative.jsonl \
        > pretrain_mixed.jsonl
    python train_pretrain.py --data_path ../dataset/pretrain_mixed.jsonl
"""

import argparse
import json
import os
import random
import sys

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

DATASET_NAME = "ikala/tmmluplus"
DATA_DIR_PREFIX = "data/"  # subject csvs live under data/<subject>_<split>.csv on the Hub


def to_declarative_text(question, options, answer_letter):
    """Convert one TMMLU+ question into a single declarative sentence.

    Template: 關於「問題敘述」，正確的答案是：[正確答案]。
    """
    q = question.strip().rstrip("？?：: ")
    correct = options[answer_letter].strip()
    return f"關於「{q}」，正確的答案是：{correct}。"


def discover_subjects(revision):
    """List subject names directly from the repo's file listing.

    TMMLU+ stores each subject/split as its own CSV under data/, e.g.
    data/accounting_train.csv, data/accounting_val.csv,
    data/accounting_test.csv, data/accounting_dev.csv. We derive the
    subject list from the *_train.csv files only -- we never need to look
    at, list contents of, or reason about the *_test.csv files to do this.
    """
    api = HfApi()
    files = api.list_repo_files(DATASET_NAME, repo_type="dataset", revision=revision)
    train_files = [
        f for f in files
        if f.startswith(DATA_DIR_PREFIX) and f.endswith("_train.csv")
    ]
    subjects = sorted(
        f[len(DATA_DIR_PREFIX):-len("_train.csv")] for f in train_files
    )
    return subjects


def load_all_subjects(revision):
    """Download and load train + validation CSVs only, per subject.

    IMPORTANT: this function calls hf_hub_download() with an explicit
    filename ending in "_train.csv" or "_val.csv" for every request. The
    string "test" never appears in any filename passed to hf_hub_download
    here, so the *_test.csv blobs for any subject are never requested,
    transferred, or cached -- not just unused after loading, but never
    fetched from the Hub in the first place. This is easy to audit: grep
    this file for "test" and you'll only find it in comments/docstrings.
    """
    subjects = discover_subjects(revision)
    per_subject = {}
    for i, subj in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] loading subject: {subj}", file=sys.stderr)

        train_path = hf_hub_download(
            DATASET_NAME,
            filename=f"{DATA_DIR_PREFIX}{subj}_train.csv",
            repo_type="dataset",
            revision=revision,
        )
        val_path = hf_hub_download(
            DATASET_NAME,
            filename=f"{DATA_DIR_PREFIX}{subj}_val.csv",
            repo_type="dataset",
            revision=revision,
        )

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        per_subject[subj] = {
            "train": train_df.to_dict("records"),
            "validation": val_df.to_dict("records"),
        }
    return per_subject


def stratified_holdout_split(per_subject, holdout_frac, seed):
    """For each subject's validation split, carve out a stratified holdout.

    Returns (train_pool, val_train_pool, val_holdout_pool), each a list of
    (subject, example) tuples.
    """
    rng = random.Random(seed)

    train_pool = []
    val_train_pool = []
    val_holdout_pool = []

    for subj, splits in per_subject.items():
        train_pool.extend((subj, ex) for ex in splits["train"])

        val = list(splits["validation"])
        rng.shuffle(val)
        n_holdout = max(1, round(len(val) * holdout_frac)) if val else 0

        val_holdout_pool.extend((subj, ex) for ex in val[:n_holdout])
        val_train_pool.extend((subj, ex) for ex in val[n_holdout:])

    return train_pool, val_train_pool, val_holdout_pool


def extract_fields(ex):
    """Pull question/options/answer out of one TMMLU+ example.

    TMMLU+ examples are expected to have keys: question, A, B, C, D, answer
    (confirmed against data/accounting_train.csv's header on the Hub).
    Adjust here if a printed row shows different column names for your
    revision.
    """
    question = ex["question"]
    options = {"A": ex["A"], "B": ex["B"], "C": ex["C"], "D": ex["D"]}
    answer = ex["answer"]
    return question, options, answer


def write_pretrain_jsonl(pool, path):
    with open(path, "w", encoding="utf-8") as f:
        for subj, ex in pool:
            question, options, answer = extract_fields(ex)
            text = to_declarative_text(question, options, answer)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def write_holdout_jsonl(pool, path):
    """Keep holdout examples in full (with subject + raw fields) so they can
    be scored later with eval_tmmluplus.py or similar — never converted to
    pretrain text, never merged into training data."""
    with open(path, "w", encoding="utf-8") as f:
        for subj, ex in pool:
            record = {"subject": subj, **ex}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare TMMLU+ data for MiniMind pretraining")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the stratified holdout split")
    parser.add_argument("--holdout_frac", type=float, default=0.2, help="fraction of each subject's validation split reserved as local eval holdout")
    parser.add_argument("--revision", type=str, default="v1.1", help="TMMLU+ dataset revision to pin")
    parser.add_argument("--out_dir", type=str, default="../dataset", help="directory to write output jsonl files into")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {DATASET_NAME} (revision={args.revision}), train+validation splits only ...", file=sys.stderr)
    per_subject = load_all_subjects(args.revision)

    print(f"Splitting validation into {args.holdout_frac:.0%} holdout / rest, seed={args.seed} ...", file=sys.stderr)
    train_pool, val_train_pool, val_holdout_pool = stratified_holdout_split(
        per_subject, args.holdout_frac, args.seed
    )

    holdout_path = os.path.join(args.out_dir, "tmmluplus_local_eval_holdout.jsonl")
    pretrain_path = os.path.join(args.out_dir, "tmmluplus_pretrain_declarative.jsonl")

    write_holdout_jsonl(val_holdout_pool, holdout_path)
    write_pretrain_jsonl(train_pool + val_train_pool, pretrain_path)

    print("Done.", file=sys.stderr)
    print(f"  train (dev) examples used for pretrain: {len(train_pool)}", file=sys.stderr)
    print(f"  validation examples used for pretrain:  {len(val_train_pool)}", file=sys.stderr)
    print(f"  validation examples held out (eval only): {len(val_holdout_pool)}", file=sys.stderr)
    print(f"  -> holdout written to:  {holdout_path}", file=sys.stderr)
    print(f"  -> pretrain data written to: {pretrain_path}", file=sys.stderr)
    print(
        "\nRemember to record seed, revision, and holdout_frac in data_manifest.md "
        "so this split is reproducible.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()