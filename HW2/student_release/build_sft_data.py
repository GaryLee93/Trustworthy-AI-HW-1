# -*- coding: utf-8 -*-
"""從 train split 產生 SFT 資料，prompt 一律由評測時實際使用的程式碼產生，確保訓練與評測一致。

輸出（每行一筆）：
  sft_tool.jsonl     {"task": "tool", "id", "kind", "messages", "tools", "target"}
      messages / tools 是 MyAgent.next_action() 實際送進 self.generate() 的內容（直接攔截取得）。
      target：要呼叫工具時是 chat template 渲染 tool_calls 的格式 <tool_call>\n{...}\n</tool_call>，
      不該呼叫時是一句簡短回覆。
  sft_reading.jsonl  {"task": "reading", "id", "type", "context", "question", "target"}
      context 是 MyAgent.prepare() 在 train 人物上的輸出、截到最後 max_context_chars 字（同 eval_memory）；
      target 是正確選項的文字（option 計分比的就是選項文字的機率）。
  sft_mix.jsonl      sft_tool.jsonl 接上 sft_reading.jsonl（Experiment 3 用）；train_sft.py 每個 epoch 會打亂順序。

用法（val 只能用來評測，不可拿來產生訓練資料）：
  python build_sft_data.py --data_dir data/train --out_dir sft_data
"""
import argparse
import json
import os
import random
from collections import Counter

from my_agent import MyAgent

# 不該呼叫工具時（閒聊、講別人的事）的回覆；內容不影響評分，只要不含 <tool_call>。
REPLIES = ["好的，我知道了。", "嗯嗯，了解！", "謝謝你跟我分享。"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def capture_action_prompt(agent, context):
    """執行 agent.next_action()，但攔截 self.generate()，取得它實際要送給模型的 messages 與 tools。"""
    captured = {}

    def fake_generate(messages, max_new_tokens=64, tools=None):
        captured["messages"] = messages
        captured["tools"] = tools
        return ""

    agent.generate = fake_generate
    agent.reset()  # 與 eval_tool 相同：每筆前先 reset
    agent.next_action(context)
    return captured["messages"], captured["tools"]


def tool_target(expected, rng):
    if expected["action"] == "call":
        # 與 chat template 渲染 tool_calls 的字串完全相同：
        # '<tool_call>\n{"name": "<name>", "arguments": <args | tojson>}\n</tool_call>'
        call = {"name": expected["tool"], "arguments": expected["args"]}
        return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
    return rng.choice(REPLIES)


def build_tool(data_dir, agent, rng):
    rows = []
    for it in load_jsonl(os.path.join(data_dir, "tool.jsonl")):
        messages, tools = capture_action_prompt(agent, it["context"])
        rows.append(dict(task="tool", id=it["id"], kind=it["kind"], messages=messages, tools=tools,
                         target=tool_target(it["expected"], rng)))
    return rows


def build_reading(data_dir, agent, max_context_chars):
    """與 eval_agent.eval_memory 相同的呼叫順序。"""
    rows = []
    for p in load_jsonl(os.path.join(data_dir, "memory.jsonl")):
        agent.reset()
        for s in p["sessions"]:
            agent.new_session(s["date"])
            for msg in s["turns"]:
                agent.chat(msg)
        snap = agent.save_snapshot()
        for q in p["questions"]:
            agent.load_snapshot(snap)
            context = (agent.prepare(q["question"]) or "")[-max_context_chars:]
            rows.append(dict(task="reading", id=q["qid"], type=q["type"], context=context,
                             question=dict(question=q["question"], options=q["options"]),
                             target=q["options"][q["answer"]]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/train")
    ap.add_argument("--out_dir", default="sft_data")
    ap.add_argument("--max_context_chars", type=int, default=300, help="與 eval_agent.py 的預設值相同")
    ap.add_argument("--seed", type=int, default=0, help="只影響不呼叫工具時的回覆句")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # 只用到 chat()/prepare()/next_action() 的組 prompt 部分，不需要模型。
    agent = MyAgent(None, None, "cpu")

    tool = build_tool(args.data_dir, agent, random.Random(args.seed))
    reading = build_reading(args.data_dir, MyAgent(None, None, "cpu"), args.max_context_chars)
    write_jsonl(os.path.join(args.out_dir, "sft_tool.jsonl"), tool)
    write_jsonl(os.path.join(args.out_dir, "sft_reading.jsonl"), reading)
    write_jsonl(os.path.join(args.out_dir, "sft_mix.jsonl"), tool + reading)

    print(f"sft_tool.jsonl     {len(tool)} rows  {dict(Counter(r['kind'] for r in tool))}")
    print(f"sft_reading.jsonl  {len(reading)} rows  {dict(Counter(r['type'] for r in reading))}")
    print(f"sft_mix.jsonl      {len(tool) + len(reading)} rows")


if __name__ == "__main__":
    main()
