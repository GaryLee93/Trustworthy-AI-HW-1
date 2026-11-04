# -*- coding: utf-8 -*-
"""HW2 Agent 介面與參考 baseline。

學生要實作一個繼承 BaseAgent 的類別，評測程式會這樣呼叫：

  agent = YourAgent(model, tokenizer, device)
  for 每個人物:
      agent.reset()
      for 每段對話: agent.new_session(date); for 每句: agent.chat(msg)
      snap = agent.save_snapshot()
      for 每題:
          agent.load_snapshot(snap)
          context = agent.prepare(question)     # 回傳「記憶內容」字串（不含題目與選項）
          → 評測程式把 context 截到最後 N 字，套上固定模板，對 A–D 打分
  for 每筆工具測試:
      agent.reset(); out = agent.next_action(context_lines)   # 回傳模型原始輸出
"""
import copy
import json
import os

import torch

TOOL_SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_spec.json")


class BaseAgent:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.reset()

    # ---------- 記憶生命週期 ----------
    def reset(self):
        """清空所有記憶（換下一個人物時呼叫）。"""
        self.memory = []
        self.date = None

    def new_session(self, date: str):
        self.date = date

    def chat(self, msg: str) -> str:
        """處理一句使用者訊息，回傳 agent 的回覆（評測不看回覆內容）。"""
        return ""

    def prepare(self, question: str) -> str:
        """回傳要放進 prompt 的記憶內容。評測程式只保留最後 max_context_chars 字。"""
        return ""

    def save_snapshot(self):
        return copy.deepcopy(self.memory)

    def load_snapshot(self, snap):
        self.memory = copy.deepcopy(snap)

    # ---------- 工具呼叫 ----------
    def next_action(self, context: list) -> str:
        """給定使用者連續說的話（最後一句是要回應的），回傳模型的原始輸出。
        要呼叫工具時輸出 <tool_call>{"name": ..., "arguments": {...}}</tool_call>，否則直接回答。"""
        msgs = [{"role": "system", "content": "你是一個會記住使用者資訊的助理。"},
                {"role": "user", "content": "\n".join(context)}]
        return self.generate(msgs, max_new_tokens=64, tools=self.tools())

    # ---------- 小工具 ----------
    @staticmethod
    def tools():
        with open(TOOL_SPEC_PATH, encoding="utf-8") as f:
            return json.load(f)["tools"]

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=64, tools=None):
        if getattr(self.tok, "chat_template", None):
            text = self.tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                                add_generation_prompt=True)
        else:
            text = "\n".join(m["content"] for m in messages) + "\n"
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                                  eos_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


# ====================================================================== baselines
class NoMemoryAgent(BaseAgent):
    """完全沒有記憶：prompt 只有題目。"""


class RecentContextAgent(BaseAgent):
    """把所有對話串起來；評測程式截斷後等於「只看得到最近的對話」。"""

    def chat(self, msg):
        self.memory.append(msg)
        return ""

    def prepare(self, question):
        return "\n".join(self.memory)


class KeywordRetrievalAgent(BaseAgent):
    """框架寫死的檢索：每句都存，提問時用字元 bigram 重疊挑最相關的句子。"""

    budget = 300

    def chat(self, msg):
        self.memory.append(msg)
        return ""

    @staticmethod
    def _bigrams(s):
        return {s[i:i + 2] for i in range(len(s) - 1)}

    def prepare(self, question):
        qb = self._bigrams(question.replace("我", "").replace("？", ""))
        scored = [(len(qb & self._bigrams(m)), i) for i, m in enumerate(self.memory)]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: (-x[0], -x[1]))   # 分數高、較新的優先
        picked, used = [], 0
        for _, i in scored:
            if used + len(self.memory[i]) + 1 > self.budget:
                break
            picked.append(i)
            used += len(self.memory[i]) + 1
        return "\n".join(self.memory[i] for i in sorted(picked))
