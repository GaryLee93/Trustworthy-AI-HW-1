# -*- coding: utf-8 -*-
"""資料健全性檢查 + 簡單 heuristic baseline（用來確認題目難度合理）。

  python check_data.py data/train data/val data/test
"""
import collections
import json
import sys

ABSTAIN = "你沒有提過"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def last_mention(p, q):
    """作弊式 heuristic：看得到選項，挑歷史中最後出現的選項值；都沒出現就選『沒提過』。
    只用來檢查題目會不會被字串比對輕易破解（學生的 prepare() 看不到選項）。"""
    hist = [t for s in p["sessions"] for t in s["turns"]]
    best, best_pos = None, -1
    for L, o in q["options"].items():
        if o == ABSTAIN:
            continue
        parts = [x for x in o.replace("住", "，").replace("當", "，").replace("；", "，").split("，") if x]
        for i, t in enumerate(hist):
            if any(x in t for x in parts) and i > best_pos:
                best, best_pos = L, i
    if best is None:
        best = ([L for L, o in q["options"].items() if o == ABSTAIN] or ["A"])[0]
    return best


def main(dirs):
    for d in dirs:
        mem, tool = load(f"{d}/memory.jsonl"), load(f"{d}/tool.jsonl")
        qs = [(p, q) for p in mem for q in p["questions"]]
        by_type = collections.Counter(q["type"] for _, q in qs)
        letters = collections.Counter(q["answer"] for _, q in qs)
        chars = [sum(len(t) for s in p["sessions"] for t in s["turns"]) for p in mem]
        turns = [sum(len(s["turns"]) for s in p["sessions"]) for p in mem]
        bad = [q["qid"] for _, q in qs if len(set(q["options"].values())) != 4
               or (ABSTAIN not in q["options"].values()) != (q["type"] == "multi")]

        acc_abs = collections.defaultdict(list)
        acc_last = collections.defaultdict(list)
        for p, q in qs:
            ans_abs = ([L for L, o in q["options"].items() if o == ABSTAIN] or ["-"])[0]
            acc_abs[q["type"]].append(ans_abs == q["answer"])
            acc_last[q["type"]].append(last_mention(p, q) == q["answer"])

        def macro(acc):
            return sum(sum(v) / len(v) for v in acc.values()) / len(acc)

        print(f"\n===== {d} =====")
        print(f"personas={len(mem)}  questions={len(qs)}  tool_items={len(tool)}")
        print("題型分布:", dict(by_type))
        print("答案字母分布:", dict(sorted(letters.items())))
        print(f"每人對話: 平均 {sum(turns)/len(turns):.0f} 句, {sum(chars)/len(chars):.0f} 字")
        print("格式錯誤題:", bad[:5] or "無")
        print("tool 類別:", dict(collections.Counter(t["kind"] for t in tool)))
        print(f"{'type':<11}{'always-abstain':>15}{'last-mention':>14}")
        for t in ["single", "update", "multi", "abstain", "distractor", "tool"]:
            a, b = acc_abs[t], acc_last[t]
            print(f"{t:<11}{sum(a)/len(a):>15.2f}{sum(b)/len(b):>14.2f}")
        print(f"{'macro':<11}{macro(acc_abs):>15.2f}{macro(acc_last):>14.2f}")

    # train / test 重疊檢查
    if len(dirs) >= 3:
        tr = load(f"{dirs[0]}/memory.jsonl")
        te = load(f"{dirs[-1]}/memory.jsonl")
        tr_sent = {t for p in tr for s in p["sessions"] for t in s["turns"]}
        te_fact = [t for p in te for s in p["sessions"]
                   for a in s["annotations"] if a["type"] != "other" for t in [s["turns"][a["turn"]]]]
        tr_q = {q["question"] for p in tr for q in p["questions"]}
        te_q = [q["question"] for p in te for q in p["questions"]]
        print(f"\ntest 事實句完全出現在 train 的比例: {sum(t in tr_sent for t in te_fact)/len(te_fact):.2f}")
        print(f"test 提問句完全出現在 train 的比例: {sum(q in tr_q for q in te_q)/len(te_q):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
