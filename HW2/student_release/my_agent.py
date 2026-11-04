# -*- coding: utf-8 -*-
"""HW2 作業範本：把這個檔案改成你自己的 agent。

評測方式：
  python eval_agent.py --model_dir <你的模型> --agent my_agent:MyAgent --data_dir data/val

規則：
  - 不要修改 eval_agent.py、agent_base.py、tool_spec.json（助教用原版評分）。
  - prepare() 回傳的記憶內容只有「最後 300 字」會被放進 prompt。
  - 主模型 ≤128M；embedding 模型 ≤30M 或只用 BM25；不准呼叫外部 API。

目前這個範本可以直接執行，行為是「每句都存、用關鍵字挑相關的句子」，
大致等於 agent_base.KeywordRetrievalAgent。請從各個 TODO 開始改進。
"""
from agent_base import BaseAgent


class MyAgent(BaseAgent):

    # ------------------------------------------------------------------
    def reset(self):
        """換下一個人物時呼叫，清空所有記憶。
        TODO: 如果你另外建了索引（例如 BM25、向量索引），也要在這裡清空。"""
        self.memory = []       # 範本：每筆是 {"date": ..., "text": ...}
        self.date = None

    def new_session(self, date: str):
        """每段新對話開始時呼叫，date 例如 "2026-03-19"。"""
        self.date = date

    # ------------------------------------------------------------------
    def chat(self, msg: str) -> str:
        """讀進使用者的一句話。回傳值不計分，重點是更新記憶。

        TODO（寫入策略），可以考慮：
          - 只存重要的句子？閒聊要不要丟掉？
          - 談論別人（「我朋友住高雄」）的句子要不要存？
          - 讓模型輸出 <tool_call> memory_save，存成 key-value（例如 居住地: 台南），
            同一個 key 再存一次就覆蓋舊值 → 直接解決 update 題。
            可以用 self.next_action([msg]) 取得模型輸出，再用 eval_agent.parse_call 解析。
        """
        self.memory.append({"date": self.date, "text": msg})
        return ""

    # ------------------------------------------------------------------
    def prepare(self, question: str) -> str:
        """提問時呼叫。回傳要放進 prompt 的記憶內容（題目與選項由評測程式加上）。

        TODO（讀取策略），可以考慮：
          - 更好的檢索（BM25、embedding、讓模型產生 memory_search 的 query）。
          - 同一件事有新舊兩筆時，只放最新的，或把日期一起放進去。
          - 需要計算的題目（例如「兩個小孩加起來幾歲」），先查出數字、
            呼叫計算機，把「計算結果：19歲」直接寫進回傳的內容。
          - 300 字很有限，排序與格式都會影響分數。
        """
        q = question.replace("我", "").replace("？", "")
        q_bigrams = {q[i:i + 2] for i in range(len(q) - 1)}

        def score(m):
            t = m["text"]
            return len(q_bigrams & {t[i:i + 2] for i in range(len(t) - 1)})

        ranked = sorted(range(len(self.memory)),
                        key=lambda i: (-score(self.memory[i]), -i))
        picked, used = [], 0
        for i in ranked:
            if score(self.memory[i]) == 0:
                break
            text = self.memory[i]["text"]
            if used + len(text) + 1 > 300:
                break
            picked.append(i)
            used += len(text) + 1
        return "\n".join(self.memory[i]["text"] for i in sorted(picked))

    # ------------------------------------------------------------------
    def next_action(self, context: list) -> str:
        """工具呼叫測試：context 是使用者連續說的話，最後一句是要回應的。
        回傳模型的原始輸出；要呼叫工具時輸出
          <tool_call>
          {"name": "memory_save", "arguments": {"key": "居住地", "value": "台南"}}
          </tool_call>

        TODO：原版 minimind-3 幾乎每次都呼叫 memory_save、參數也填錯。
          用 data/train/tool.jsonl 做 SFT / LoRA，教模型分辨「存 / 查 / 算 / 不呼叫」。
          也可以試試限制解碼、調整 system prompt、repetition_penalty 等技巧。
        """
        return super().next_action(context)

    # ------------------------------------------------------------------
    # 記憶如果都放在 self.memory，不用改下面兩個方法。
    # 如果你另外有索引或其他狀態，要一起存進 snapshot、一起還原。
    # def save_snapshot(self): ...
    # def load_snapshot(self, snap): ...
