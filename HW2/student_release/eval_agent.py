# -*- coding: utf-8 -*-
"""HW2 評測程式。

範例：
  # 無記憶 baseline
  python eval_agent.py --model_dir ../out/hw1_model --agent agent_base:NoMemoryAgent --data_dir data/val
  # 學生自己的 agent（my_agent.py 裡的 MyAgent）
  python eval_agent.py --model_dir ../out/hw1_model --agent my_agent:MyAgent --data_dir data/val
  # Oracle：直接把標準答案的證據句當成記憶（只用來檢查題目難度）
  python eval_agent.py --model_dir ../out/hw1_model --oracle --data_dir data/val

輸出：<out>/summary.json、<out>/memory_pred.jsonl、<out>/tool_pred.jsonl
"""
import argparse
import importlib
import json
import math
import os
import re
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QTYPES = ["single", "update", "multi", "abstain", "distractor", "tool"]
LETTERS = "ABCD"


# ====================================================================== prompt & scoring
def build_prompt(tok, context, q, use_chat):
    body = ""
    if context:
        body += f"以下是你記得的關於使用者的資訊：\n{context}\n\n"
    body += f"問題：{q['question']}\n"
    body += "".join(f"{L}. {q['options'][L]}\n" for L in LETTERS)
    body += "請直接回答選項字母。"
    if use_chat:
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True) + "答案："
    return body + "\n答案："


@torch.no_grad()
def score_letters(model, tok, prompt, letter_ids, device):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    logits = model(ids).logits[0, -1].float()
    logp = torch.log_softmax(logits, -1)
    return [logp[i].item() for i in letter_ids]


@torch.no_grad()
def score_options(model, tok, prompt, q, device):
    """option 模式：比較每個選項內容的平均 log-likelihood（像 lm-eval 的 acc_norm）。"""
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    scores = []
    for L in LETTERS:
        o_ids = tok(q["options"][L], add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + o_ids], device=device)
        logp = torch.log_softmax(model(ids).logits[0].float(), -1)
        tot = sum(logp[len(p_ids) + j - 1, t].item() for j, t in enumerate(o_ids))
        scores.append(tot / max(len(q["options"][L]), 1))
    return scores


def _option_logprob(model, tok, prompt, option, device):
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    o_ids = tok(option, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + o_ids], device=device)
    logp = torch.log_softmax(model(ids).logits[0].float(), -1)
    return sum(logp[len(p_ids) + j - 1, t].item() for j, t in enumerate(o_ids))


_UNCOND_CACHE = {}


@torch.no_grad()
def score_options_calib(model, tok, prompt, q, device, use_chat):
    """option_calib 模式：log P(選項 | 完整 prompt) − log P(選項 | 沒有題目的空 prompt)。
    抵銷模型對選項本身的偏好（例如偏愛「你沒有提過」），只看記憶與題目讓機率提高多少。"""
    if use_chat:
        null = tok.apply_chat_template([{"role": "user", "content": "請回答。"}],
                                       tokenize=False, add_generation_prompt=True) + "答案："
    else:
        null = "答案："
    scores = []
    for L in LETTERS:
        o = q["options"][L]
        if o not in _UNCOND_CACHE:
            _UNCOND_CACHE[o] = _option_logprob(model, tok, null, o, device)
        scores.append(_option_logprob(model, tok, prompt, o, device) - _UNCOND_CACHE[o])
    return scores


# ====================================================================== tool-call parsing
CALL_RE = re.compile(r"<call>\s*([A-Za-z_]+)\s*(\{.*?\})?\s*</call>", re.S)          # 簡化格式
TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)              # MiniMind 原生格式
SAFE_EXPR = re.compile(r"^[\d\s\.\+\-\*/\(\)]+$")


def parse_call(text):
    """回傳 (has_call, parsed_ok, tool, args)。"""
    if "<tool_call>" in text:
        m = TOOLCALL_RE.search(text)
        if not m:
            return True, False, None, None
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict) or not isinstance(obj.get("name"), str):
                raise ValueError
            return True, True, obj["name"], args
        except Exception:
            return True, False, None, None
    if "<call>" not in text:
        return False, False, None, None
    m = CALL_RE.search(text)
    if not m:
        return True, False, None, None
    try:
        args = json.loads(m.group(2)) if m.group(2) else {}
        if not isinstance(args, dict):
            raise ValueError
    except Exception:
        return True, False, m.group(1), None
    return True, True, m.group(1), args


def safe_eval(expr):
    expr = str(expr).replace("×", "*").replace("÷", "/").replace("x", "*")
    if not SAFE_EXPR.match(expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


def args_match(tool, args, accept):
    def s(k):
        return str(args.get(k, ""))
    if tool == "memory_save":
        return any(a in s("key") for a in accept["key"]) and any(v in s("value") for v in accept["value"])
    if tool == "memory_search":
        return any(a in s("query") for a in accept["query"])
    if tool == "calculator":
        v = safe_eval(s("expression"))
        return v is not None and math.isclose(v, accept["value"], rel_tol=1e-6, abs_tol=1e-6)
    if tool == "get_date":
        return True
    return False


# ====================================================================== main loops
def load_jsonl(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return rows[:limit] if limit else rows


def eval_memory(args, model, tok, agent, device, use_chat, letter_ids):
    personas = load_jsonl(os.path.join(args.data_dir, "memory.jsonl"), args.limit)
    preds, acc = [], defaultdict(list)
    t0 = time.time()
    for pi, p in enumerate(personas):
        if not args.oracle:
            agent.reset()
            for s in p["sessions"]:
                agent.new_session(s["date"])
                for msg in s["turns"]:
                    agent.chat(msg)
            snap = agent.save_snapshot()
        turns = {s["session_id"]: s["turns"] for s in p["sessions"]}
        for q in p["questions"]:
            if args.oracle:
                context = "\n".join(turns[sid][i] for sid, i in q.get("evidence", []))
            else:
                agent.load_snapshot(snap)
                context = agent.prepare(q["question"]) or ""
            context = context[-args.max_context_chars:]
            prompt = build_prompt(tok, context, q, use_chat)
            if args.score == "letter":
                sc = score_letters(model, tok, prompt, letter_ids, device)
            elif args.score == "option":
                sc = score_options(model, tok, prompt, q, device)
            else:
                sc = score_options_calib(model, tok, prompt, q, device, use_chat)
            pred = LETTERS[max(range(4), key=lambda i: sc[i])]
            ok = pred == q["answer"]
            acc[q["type"]].append(ok)
            preds.append(dict(qid=q["qid"], type=q["type"], pred=pred, answer=q["answer"],
                              correct=ok, context=context, scores=sc))
        if (pi + 1) % 10 == 0 or pi + 1 == len(personas):
            print(f"  memory {pi + 1}/{len(personas)} personas, {time.time() - t0:.0f}s", flush=True)
    per_type = {t: sum(acc[t]) / len(acc[t]) for t in QTYPES if acc[t]}
    res = dict(per_type=per_type,
               macro_acc=sum(per_type.values()) / len(per_type),
               micro_acc=sum(sum(v) for v in acc.values()) / sum(len(v) for v in acc.values()),
               n_questions=len(preds), seconds=time.time() - t0)
    return res, preds


def eval_tool(args, agent):
    items = load_jsonl(os.path.join(args.data_dir, "tool.jsonl"), args.limit_tool)
    preds = []
    n = defaultdict(int)
    t0 = time.time()
    for it in items:
        agent.reset()
        out = agent.next_action(it["context"])
        has_call, parsed, tool, a = parse_call(out)
        exp = it["expected"]
        want_call = exp["action"] == "call"
        tool_ok = want_call and parsed and tool == exp["tool"]
        arg_ok = tool_ok and args_match(tool, a, exp.get("accept", {}))
        e2e = arg_ok if want_call else not has_call
        n["total"] += 1
        n["decision_ok"] += has_call == want_call
        n["has_call"] += has_call
        n["parsed"] += parsed
        n["want_call"] += want_call
        n["tool_ok"] += tool_ok
        n["arg_ok"] += arg_ok
        n["e2e"] += e2e
        preds.append(dict(id=it["id"], kind=it["kind"], output=out, has_call=has_call, parsed=parsed,
                          tool=tool, args=a, tool_ok=tool_ok, arg_ok=arg_ok, e2e=e2e))
    div = lambda x, y: x / y if y else 0.0
    res = dict(
        format_validity=div(n["parsed"], n["has_call"]),   # 有輸出 <call> 的裡面，能被解析的比例
        decision_acc=div(n["decision_ok"], n["total"]),    # 該不該呼叫工具判斷正確
        tool_acc=div(n["tool_ok"], n["want_call"]),        # 該呼叫的題目中，工具選對
        arg_acc=div(n["arg_ok"], n["want_call"]),          # 該呼叫的題目中，工具與參數都對
        end_to_end=div(n["e2e"], n["total"]),              # 整體：該呼叫的全對、不該呼叫的沒呼叫
        n_items=n["total"], seconds=time.time() - t0)
    return res, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--agent", default="agent_base:NoMemoryAgent", help="模組:類別")
    ap.add_argument("--data_dir", default="data/val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_context_chars", type=int, default=300)
    ap.add_argument("--score", choices=["letter", "option", "option_calib"], default="option",
                    help="letter: 比 A–D 字母機率；option: 比選項內容平均機率；option_calib: 選項機率減去無題目時的機率")
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="用標準答案證據句當記憶（檢查難度用）")
    ap.add_argument("--skip_tool", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 個人物（除錯用）")
    ap.add_argument("--limit_tool", type=int, default=None)
    ap.add_argument("--time_limit", type=int, default=1800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True,
                                                 dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    use_chat = bool(getattr(tok, "chat_template", None)) and not args.no_chat_template
    letter_ids = [tok(L, add_special_tokens=False).input_ids[0] for L in LETTERS]
    assert len(set(letter_ids)) == 4, f"A–D 的 token id 重複：{letter_ids}"

    mod, cls = args.agent.split(":")
    agent = getattr(importlib.import_module(mod), cls)(model, tok, device)
    name = "oracle" if args.oracle else cls
    out_dir = args.out or os.path.join("results", f"{name}_{os.path.basename(args.data_dir.rstrip('/'))}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"model params={n_params / 1e6:.1f}M  chat_template={use_chat}  agent={name}  data={args.data_dir}")

    summary = dict(agent=name, model_dir=args.model_dir, params_M=round(n_params / 1e6, 2),
                   data_dir=args.data_dir, max_context_chars=args.max_context_chars,
                   score=args.score, chat_template=use_chat)
    t0 = time.time()
    mem_res, mem_pred = eval_memory(args, model, tok, agent, device, use_chat, letter_ids)
    summary["memory"] = mem_res
    with open(os.path.join(out_dir, "memory_pred.jsonl"), "w", encoding="utf-8") as f:
        for r in mem_pred:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not args.skip_tool and not args.oracle:
        tool_res, tool_pred = eval_tool(args, agent)
        summary["tool"] = tool_res
        with open(os.path.join(out_dir, "tool_pred.jsonl"), "w", encoding="utf-8") as f:
            for r in tool_pred:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if "tool" in summary:
        # 排行榜分數 = 0.7 × Memory Accuracy + 0.3 × Tool End-to-End
        summary["leaderboard_score"] = 0.7 * mem_res["macro_acc"] + 0.3 * summary["tool"]["end_to_end"]
    summary["total_seconds"] = time.time() - t0
    summary["within_time_limit"] = summary["total_seconds"] <= args.time_limit
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ RESULT ================")
    print(f"Memory Accuracy (macro) = {mem_res['macro_acc']:.4f}   (micro {mem_res['micro_acc']:.4f}, random ≈ 0.25)")
    for t, v in mem_res["per_type"].items():
        print(f"  {t:<11} {v:.4f}")
    if "tool" in summary:
        tr = summary["tool"]
        print(f"Tool: format={tr['format_validity']:.3f} decision={tr['decision_acc']:.3f} "
              f"tool={tr['tool_acc']:.3f} arg={tr['arg_acc']:.3f} e2e={tr['end_to_end']:.3f}")
        print(f"Leaderboard Score = 0.7 × {mem_res['macro_acc']:.4f} + 0.3 × {tr['end_to_end']:.4f} "
              f"= {summary['leaderboard_score']:.4f}")
    print(f"Time: {summary['total_seconds']:.0f}s  (limit {args.time_limit}s, ok={summary['within_time_limit']})")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
