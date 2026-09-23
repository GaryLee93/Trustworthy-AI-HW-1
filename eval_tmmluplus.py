#!/usr/bin/env python3
"""
eval_tmmluplus.py — TMMLU+ evaluation script for Homework X.

Scores a causal LM on TMMLU+ multiple-choice questions by log-likelihood:
for each question the model sees the prompt, and the predicted answer is the
option label (A/B/C/D) with the highest log-probability as the next token(s).

Students only need this:

    python eval_tmmluplus.py eval --model_path ./model

That evaluates on the practice set: a fixed 10% sample of the TMMLU+ test
split, identical for everyone, derived automatically. Nothing to configure.

    python eval_tmmluplus.py build --output practice.jsonl

writes the same practice questions to a file, if you want to inspect them.

Staff: passing a --seed to build produces a private grading set from the
questions NOT in the practice set; the two can never overlap.

    python eval_tmmluplus.py build --seed <secret> --output private_eval.jsonl
    python eval_tmmluplus.py eval --model_path ./model --data private_eval.jsonl

Local data files (.jsonl or .csv) need the columns:
  subject, question, A, B, C, D, answer
"""


import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Subjects and categories
# Subject -> (Chinese name, fine category). Source: lm-evaluation-harness
# tmmluplus/subject.tsv, which follows the official iKala/ievals settings.
# ---------------------------------------------------------------------------
SUBJECTS = {
    "dentistry": ("牙醫學", "health"),
    "traditional_chinese_medicine_clinical_medicine": ("中醫臨床醫學", "health"),
    "clinical_psychology": ("臨床心理學", "psychology"),
    "technical": ("技術工相關", "other"),
    "culinary_skills": ("餐旅", "other"),
    "mechanical": ("機械與機電概論", "other"),
    "logic_reasoning": ("邏輯思維", "other"),
    "real_estate": ("房地產", "other"),
    "general_principles_of_law": ("法學大意", "law"),
    "finance_banking": ("金融與法規", "business"),
    "anti_money_laundering": ("洗錢防制", "law"),
    "ttqav2": ("台灣在地用語", "culture"),
    "marketing_management": ("行銷管理", "other"),
    "business_management": ("企業管理", "other"),
    "organic_chemistry": ("有機化學", "chemistry"),
    "advance_chemistry": ("化學", "chemistry"),
    "physics": ("物理", "physics"),
    "secondary_physics": ("高中物理", "physics"),
    "human_behavior": ("人類行為與社會", "psychology"),
    "national_protection": ("軍事", "politics"),
    "jce_humanities": ("指考人文科目", "philosophy"),
    "politic_science": ("政治", "politics"),
    "agriculture": ("農業", "other"),
    "official_document_management": ("機關文書", "other"),
    "financial_analysis": ("財務分析", "business"),
    "pharmacy": ("藥劑學", "biology"),
    "educational_psychology": ("教育心理", "psychology"),
    "statistics_and_machine_learning": ("統計與機器學習", "engineering"),
    "management_accounting": ("管理會計", "business"),
    "introduction_to_law": ("法律概論", "law"),
    "computer_science": ("資訊工程", "computer science"),
    "veterinary_pathology": ("獸醫病理學", "health"),
    "accounting": ("會計學", "business"),
    "fire_science": ("火災學", "other"),
    "optometry": ("視光學", "other"),
    "insurance_studies": ("保險學", "other"),
    "pharmacology": ("藥理學", "health"),
    "taxation": ("稅務", "law"),
    "education_(profession_level)": ("教育專業", "education"),
    "economics": ("經濟學", "economics"),
    "veterinary_pharmacology": ("獸醫藥理學", "health"),
    "nautical_science": ("航海", "other"),
    "occupational_therapy_for_psychological_disorders": ("心理障礙職能治療學", "psychology"),
    "trust_practice": ("信託實務", "law"),
    "geography_of_taiwan": ("台灣地理", "geography"),
    "physical_education": ("體育", "education"),
    "auditing": ("審計學", "business"),
    "administrative_law": ("行政法", "law"),
    "basic_medical_science": ("基礎醫學", "biology"),
    "macroeconomics": ("總經", "economics"),
    "trade": ("貿易", "business"),
    "chinese_language_and_literature": ("國文", "culture"),
    "tve_design": ("統測＿設計", "other"),
    "junior_science_exam": ("國中會考基測自然科", "biology"),
    "junior_math_exam": ("國中會考基測數學科", "math"),
    "junior_chinese_exam": ("國中會考基測國文", "culture"),
    "junior_social_studies": ("國中會考基測社會科", "other"),
    "tve_mathematics": ("統測數學", "math"),
    "tve_chinese_language": ("統測國文", "culture"),
    "tve_natural_sciences": ("統測自然科", "biology"),
    "junior_chemistry": ("國中理化", "chemistry"),
    "music": ("音樂科", "other"),
    "education": ("教育常識", "education"),
    "three_principles_of_people": ("三民主義", "culture"),
    "taiwanese_hokkien": ("閩南語", "culture"),
    "engineering_math": ("工程數學", "math"),
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "Humanities": ["history", "philosophy", "law"],
    "Social Sciences": ["politics", "culture", "economics", "geography", "psychology", "education"],
    "Other": ["other", "business", "health"],
}
FINE_TO_CATEGORY = {fine: cat for cat, fines in CATEGORIES.items() for fine in fines}

LABELS = ["A", "B", "C", "D"]
DATASET_ID = "ikala/tmmluplus"
DATASET_REVISION = "v1.1"

# The practice set: a fixed 10% of each subject's test split. Everyone gets the
# same questions, so these constants must never change once the homework is out.
PRACTICE_SEED = 20260101
SAMPLE_FRACTION = 0.10
MIN_PER_SUBJECT = 10

# ---------------------------------------------------------------------------
# Prompt template (the official one for this homework — do not change)
# ---------------------------------------------------------------------------
INSTRUCTION = "以下是關於{subject_zh}的單選題，請直接給出正確答案的選項。\n\n"
ANSWER_PREFIX = "答案："


def format_question(row, with_answer=False):
    text = row["question"].strip() + "\n"
    for label in LABELS:
        text += f"{label}. {str(row[label]).strip()}\n"
    text += ANSWER_PREFIX
    if with_answer:
        text += row["answer"] + "\n\n"
    return text


def build_prompt(row, fewshot_rows, tokenizer, use_chat_template):
    subject_zh = SUBJECTS.get(row["subject"], (row["subject"], ""))[0]
    body = INSTRUCTION.format(subject_zh=subject_zh)
    for ex in fewshot_rows:
        body += format_question(ex, with_answer=True)
    body += format_question(row)

    if not use_chat_template:
        return body

    # Chat mode: question (without the trailing answer prefix) goes in the user
    # turn, and the answer prefix starts the assistant turn.
    user_content = body[: -len(ANSWER_PREFIX)].rstrip()
    messages = [{"role": "user", "content": user_content}]
    try:
        # MiniMind templates accept open_thinking; disable explicit thinking.
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, open_thinking=False
        )
    except TypeError:
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat + ANSWER_PREFIX


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_hf_split(split, subjects):
    from datasets import load_dataset

    rows = []
    for subject in subjects:
        ds = load_dataset(DATASET_ID, subject, split=split, revision=DATASET_REVISION)
        for r in ds:
            rows.append({"subject": subject, **{k: r[k] for k in ["question", *LABELS, "answer"]}})
    return rows


def load_local_file(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif path.endswith(".csv"):
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        sys.exit(f"Unsupported file type: {path} (use .jsonl or .csv)")
    for i, r in enumerate(rows):
        missing = [k for k in ["subject", "question", *LABELS, "answer"] if k not in r]
        if missing:
            sys.exit(f"Row {i} in {path} is missing columns: {missing}")
        if r["answer"] not in LABELS:
            sys.exit(f"Row {i} has invalid answer {r['answer']!r}")
    return rows


def shuffle_options(row, rng):
    """Return a copy of row with options permuted and the answer relabeled."""
    options = [row[l] for l in LABELS]
    correct_text_index = LABELS.index(row["answer"])
    order = list(range(4))
    rng.shuffle(order)
    new_row = dict(row)
    for new_pos, old_pos in enumerate(order):
        new_row[LABELS[new_pos]] = options[old_pos]
    new_row["answer"] = LABELS[order.index(correct_text_index)]
    return new_row


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
class Scorer:
    def __init__(self, model, tokenizer, max_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.label_ids = [tokenizer(l, add_special_tokens=False)["input_ids"] for l in LABELS]
        self.single_token = all(len(ids) == 1 for ids in self.label_ids)
        if not self.single_token:
            print("[warn] Some labels span multiple tokens; using slower full-sequence scoring.")
        self.n_truncated = 0

    def _truncate(self, ids, reserve):
        limit = self.max_length - reserve
        if len(ids) > limit:
            self.n_truncated += 1
            ids = ids[-limit:]  # keep the end: the question being asked
        return ids

    @torch.no_grad()
    def score(self, prompt):
        """Return log-probabilities for A/B/C/D."""
        ctx = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        if self.single_token:
            ctx = self._truncate(ctx, reserve=0)
            logits = self.model(torch.tensor([ctx], device=self.device)).logits[0, -1]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            return [logprobs[ids[0]].item() for ids in self.label_ids]

        scores = []
        max_cont = max(len(ids) for ids in self.label_ids)
        ctx = self._truncate(ctx, reserve=max_cont)
        for cont in self.label_ids:
            ids = torch.tensor([ctx + cont], device=self.device)
            logprobs = torch.log_softmax(self.model(ids).logits[0].float(), dim=-1)
            start = len(ctx) - 1
            scores.append(sum(logprobs[start + i, tok].item() for i, tok in enumerate(cont)))
        return scores


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_eval(args):
    torch.manual_seed(0)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"auto": "auto", "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    # Data
    subjects = args.subjects.split(",") if args.subjects else list(SUBJECTS)
    if args.data == "practice":
        rows = load_practice_set()
    elif args.data in ("validation", "train"):
        rows = load_hf_split(args.data, subjects)
    else:
        rows = load_local_file(args.data)
    rows = [r for r in rows if r["subject"] in subjects]
    if args.limit:
        rows = rows[: args.limit]

    fewshot = defaultdict(list)
    if args.num_fewshot > 0:
        for r in load_hf_split("train", sorted({r["subject"] for r in rows})):
            if len(fewshot[r["subject"]]) < args.num_fewshot:
                fewshot[r["subject"]].append(r)

    # Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    max_length = args.max_length or getattr(model.config, "max_position_embeddings", 2048)
    if args.max_params and n_params > args.max_params:
        sys.exit(f"Model has {n_params:,} parameters, exceeding the limit of {args.max_params:,}.")

    print(f"Model: {args.model_path} | params: {n_params / 1e6:.1f}M | device: {device}")
    print(f"Questions: {len(rows)} | subjects: {len({r['subject'] for r in rows})} | "
          f"few-shot: {args.num_fewshot} | chat template: {args.chat_template}")

    # Run
    scorer = Scorer(model, tokenizer, max_length, device)
    correct_by_subject = defaultdict(int)
    total_by_subject = defaultdict(int)
    predictions = []
    start = time.time()
    for i, row in enumerate(rows):
        prompt = build_prompt(row, fewshot[row["subject"]], tokenizer, args.chat_template)
        scores = scorer.score(prompt)
        pred = LABELS[max(range(4), key=lambda k: scores[k])]
        ok = pred == row["answer"]
        correct_by_subject[row["subject"]] += ok
        total_by_subject[row["subject"]] += 1
        predictions.append({"subject": row["subject"], "answer": row["answer"], "pred": pred,
                            "scores": [round(s, 4) for s in scores]})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}  running acc={sum(correct_by_subject.values()) / (i + 1):.4f}")
    elapsed = time.time() - start

    # Aggregate: subject accuracy -> category = mean of its subjects -> macro over 4 categories
    subject_acc = {s: correct_by_subject[s] / total_by_subject[s] for s in total_by_subject}
    cat_subject_accs = defaultdict(list)
    for s, acc in subject_acc.items():
        fine = SUBJECTS.get(s, (None, "other"))[1]
        cat_subject_accs[FINE_TO_CATEGORY[fine]].append(acc)
    category_acc = {c: sum(v) / len(v) for c, v in cat_subject_accs.items()}
    macro = sum(category_acc.values()) / len(category_acc)
    n_total = sum(total_by_subject.values())
    micro = sum(correct_by_subject.values()) / n_total

    results = {
        "model_path": args.model_path,
        "n_params": n_params,
        "data": args.data,
        "n_questions": n_total,
        "num_fewshot": args.num_fewshot,
        "chat_template": args.chat_template,
        "macro_accuracy": round(macro, 4),          # official homework metric
        "micro_accuracy": round(micro, 4),
        "stderr_micro": round(math.sqrt(micro * (1 - micro) / n_total), 4),
        "category_accuracy": {c: round(a, 4) for c, a in sorted(category_acc.items())},
        "subject_accuracy": {s: round(a, 4) for s, a in sorted(subject_acc.items())},
        "n_truncated_prompts": scorer.n_truncated,
        "seconds": round(elapsed, 1),
    }

    print("\n=== Results ===")
    for c, a in sorted(category_acc.items()):
        print(f"  {c:<16} {a:.4f}")
    print(f"  {'MACRO (official)':<16} {macro:.4f}")
    print(f"  {'micro':<16} {micro:.4f} ± {results['stderr_micro']:.4f}")
    print(f"  time: {elapsed:.1f}s | truncated prompts: {scorer.n_truncated}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if args.save_predictions:
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved results to {args.output}")


def subject_rng(seed, subject):
    """Deterministic per-subject RNG.

    Seeding per subject (rather than sharing one stream) means the sample for
    one subject never depends on how many subjects were processed before it, so
    the same --seed always yields the same questions.
    """
    digest = hashlib.sha256(f"{seed}:{subject}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def question_key(row):
    """Stable identity of a question, unaffected by option shuffling."""
    return hashlib.sha256(row["question"].strip().encode("utf-8")).hexdigest()


def sample_test_split(seed, exclude_keys=(), shuffle_opts=True):
    """Subject-stratified sample of the test split, fully determined by seed."""
    by_subject = defaultdict(list)
    for r in load_hf_split("test", list(SUBJECTS)):
        if question_key(r) not in exclude_keys:
            by_subject[r["subject"]].append(r)

    sampled = []
    for subject in sorted(by_subject):
        # Sort the pool so the sample cannot depend on dataset row order.
        pool = sorted(by_subject[subject], key=lambda r: (r["question"], r["answer"]))
        k = min(max(MIN_PER_SUBJECT, math.ceil(len(pool) * SAMPLE_FRACTION)), len(pool))
        rng = subject_rng(seed, subject)
        picked = [pool[i] for i in sorted(rng.sample(range(len(pool)), k))]
        if shuffle_opts:
            picked = [shuffle_options(r, rng) for r in picked]
        sampled.extend(picked)

    random.Random(seed).shuffle(sampled)
    return sampled


def load_practice_set():
    return sample_test_split(PRACTICE_SEED, shuffle_opts=False)


def cmd_build(args):
    if args.seed is None:
        rows = load_practice_set()
        label = "practice set"
    else:
        practice_keys = {question_key(r) for r in load_practice_set()}
        rows = sample_test_split(args.seed, exclude_keys=practice_keys)
        label = "private grading set (disjoint from the practice set)"

    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(open(args.output, "rb").read()).hexdigest()[:16]
    print(f"Wrote {len(rows)} questions to {args.output}  ({label})")
    print(f"sha256[:16]={digest}  — the same inputs always reproduce this checksum.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("eval", help="Evaluate a model")
    e.add_argument("--model_path", required=True, help="Hugging Face format model directory")
    e.add_argument("--data", default="practice",
                   help="'practice' (default, the fixed 10%% practice set), 'validation', "
                        "'train', or the path to a local .jsonl/.csv file")
    e.add_argument("--subjects", default=None, help="Comma-separated subset of subjects")
    e.add_argument("--num_fewshot", type=int, default=0, help="Few-shot examples from the train split")
    e.add_argument("--chat_template", action="store_true", help="Wrap the prompt with the tokenizer chat template")
    e.add_argument("--max_length", type=int, default=None, help="Max prompt tokens (default: model max)")
    e.add_argument("--max_params", type=int, default=None, help="Reject models above this parameter count")
    e.add_argument("--dtype", default="fp32", choices=["auto", "fp32", "fp16", "bf16"])
    e.add_argument("--device", default=None)
    e.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions (debugging)")
    e.add_argument("--output", default="results.json")
    e.add_argument("--save_predictions", default=None, help="Optional .jsonl of per-question predictions")
    e.set_defaults(func=cmd_eval)

    b = sub.add_parser("build", help="Write the practice questions to a file")
    b.add_argument("--output", default="practice.jsonl")
    b.add_argument("--seed", type=int, default=None,
                   help="(Staff) build a private grading set with this seed instead, "
                        "drawn from the questions not in the practice set")
    b.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
