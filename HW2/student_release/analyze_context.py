# -*- coding: utf-8 -*-
"""離線分析每個版本放進 prompt 的記憶內容（報告 1.6 節的 Table 2、Table 3）。

不重新跑模型：讀 eval_agent.py 輸出的 memory_pred.jsonl，對照驗證集的 evidence 與 annotations。

用法：
  python analyze_context.py
  python analyze_context.py --runs "Update 1=KeywordRetrievalAgent_val" "Update 4=E1_update4"

注意：
  - Update 1 只改了 tool 題，非 tool 題的 context 與 KeywordRetrievalAgent 完全相同，所以用它的結果代表。
  - 只看非 tool 題（240 題）；evidence 相關的指標只看有證據句的 single / update / multi / distractor（197 題）。
"""
import argparse
import json
import os

EVIDENCE_TYPES = ("single", "update", "multi", "distractor")
LETTERS = "ABCD"
DEFAULT_RUNS = [
    "Update 1=KeywordRetrievalAgent_val",
    "Update 2=E1_update2",
    "Update 3=E1_update3",
    "Update 4=E1_update4",
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_val(data_dir):
    """回傳 {qid: (問題, 證據句, {句子: 標註類型})}。"""
    out = {}
    for p in load_jsonl(os.path.join(data_dir, "memory.jsonl")):
        labels = {s["turns"][a["turn"]]: a["type"] for s in p["sessions"] for a in s["annotations"]}
        for q in p["questions"]:
            evidence = [p["sessions"][sid - 1]["turns"][i] for sid, i in q["evidence"]]
            out[q["qid"]] = (q, evidence, labels)
    return out


def analyze(preds, no_memory, val):
    n = lines_total = small_talk = chars = with_other = picks_other = 0
    ev_n = ev_hit = correct_ev = correct_no_ev = 0
    abstain_n = abstain_empty = 0
    boosts = []
    for r in preds:
        if r["type"] == "tool":
            continue
        q, evidence, labels = val[r["qid"]]
        lines = [line for line in r["context"].split("\n") if line]
        n += 1
        lines_total += len(lines)
        chars += len(r["context"])
        small_talk += sum(line not in labels for line in lines)
        with_other += any(labels.get(line) == "other" for line in lines)
        pred_text = q["options"][r["pred"]]
        picks_other += any(labels.get(line) == "other" and pred_text in line for line in lines)
        if r["type"] == "abstain":
            abstain_n += 1
            abstain_empty += not lines
        if r["type"] in EVIDENCE_TYPES:
            hit = all(e in lines for e in evidence)
            ev_n += 1
            ev_hit += hit
            correct_ev += hit and r["correct"]
            correct_no_ev += (not hit) and r["correct"]
            # 正解分數相對 NoMemory 的提升，減去其他三個選項的平均提升
            gold = LETTERS.index(q["answer"])
            d = [a - b for a, b in zip(r["scores"], no_memory[r["qid"]]["scores"])]
            boosts.append(d[gold] - sum(d[i] for i in range(4) if i != gold) / 3)
    return {
        "context": {
            "Evidence recall*": ev_hit / ev_n,
            "Share of small talk in context lines": small_talk / max(lines_total, 1),
            "Context length (characters)": chars / n,
            "Abstain questions with empty context": abstain_empty / abstain_n,
            "Questions with other people in context": with_other / n,
        },
        "grounding": {
            "Correct with evidence in context*": correct_ev,
            "Correct without evidence in context (guesses)*": correct_no_ev,
            "Predictions of another person's value": picks_other,
            "Gold boost*": sum(boosts) / len(boosts),
        },
        "counts": dict(non_tool=n, evidence_questions=ev_n, abstain=abstain_n),
    }


def fmt(v):
    if isinstance(v, int):
        return str(v)
    return f"{v:.0f}" if v >= 10 else f"{v:.2f}"


def print_table(title, names, results, key):
    metrics = list(results[0][key])
    width = max(len(m) for m in metrics) + 2
    print(f"\n{title}")
    print(f"{'Metric':<{width}}" + "".join(f"{name:>12}" for name in names))
    for m in metrics:
        print(f"{m:<{width}}" + "".join(f"{fmt(r[key][m]):>12}" for r in results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--data_dir", default="data/val")
    ap.add_argument("--no_memory", default="NoMemoryAgent_val", help="算 gold boost 用的 NoMemory 結果資料夾")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="名稱=結果資料夾")
    args = ap.parse_args()

    val = load_val(args.data_dir)
    no_memory = {r["qid"]: r for r in load_jsonl(os.path.join(args.results_dir, args.no_memory, "memory_pred.jsonl"))}
    names, results = [], []
    for spec in args.runs:
        name, folder = spec.split("=", 1)
        preds = load_jsonl(os.path.join(args.results_dir, folder, "memory_pred.jsonl"))
        names.append(name)
        results.append(analyze(preds, no_memory, val))

    c = results[0]["counts"]
    print(f"non-tool questions: {c['non_tool']}, with evidence (*): {c['evidence_questions']}, abstain: {c['abstain']}")
    print_table("Table 2: context quality", names, results, "context")
    print_table("Table 3: grounding of answers", names, results, "grounding")


if __name__ == "__main__":
    main()
