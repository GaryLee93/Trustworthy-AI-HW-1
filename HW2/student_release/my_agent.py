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
import re

from agent_base import BaseAgent

# ---------------------------------------------------------------- 計算題（改進 1）
# 句子提到這些人物時，數字屬於別人，不能拿來計算（例如「我朋友的兒子9歲」）。
OTHER_PEOPLE = (
    "朋友", "同事", "同學", "室友", "鄰居", "主管", "老闆", "學長", "學姊", "學弟", "學妹",
    "表哥", "表姊", "表弟", "表妹", "堂哥", "堂姊", "堂弟", "堂妹",
    "爸", "媽", "哥哥", "姊姊", "弟弟", "妹妹", "老公", "老婆", "男友", "女友",
    "阿姨", "叔叔", "舅舅", "姑姑", "伯伯", "爺爺", "奶奶", "外公", "外婆",
)

# 數值事實：(句子必須包含的關鍵字, 抓數字的 regex)。同一句可以同時符合多個事實。
# 年齡的 regex 可以跨過逗號（「說到我兒子，他今年12歲」），但中間不能出現另一個小孩，
# 避免「我女兒很乖，兒子8歲」把 8 算給女兒。
NUM_FACTS = {
    "daughter": (("女兒", "閨女"), (r"(?:女兒|閨女)(?:(?!兒子)[^\d。；]){0,8}?(\d+)\s*歲",
                                  r"(\d+)\s*歲的?(?:女兒|閨女)")),
    "son": (("兒子",), (r"兒子(?:(?!女兒|閨女)[^\d。；]){0,8}?(\d+)\s*歲", r"(\d+)\s*歲的?兒子")),
    # 房租沒寫單位時（「每個月租金要付15000」），退而求其次抓三位數以上的數字。
    "rent": (("租", "房東"), (r"(\d+)\s*(?:元|塊)", r"(\d{3,})")),
    "siblings": (("兄弟姊妹", "手足"), (r"(\d+)\s*個",)),
}


def _norm(text):
    """統一異體字：「兄弟姐妹」→「兄弟姊妹」、「姐姐」→「姊姊」。"""
    return text.replace("姐", "姊")


def _about_other(text):
    return any(w in text for w in OTHER_PEOPLE)


def _extract_nums(text):
    """從一句話抓出使用者自己的數值事實，回傳 {"daughter": 6, ...}。"""
    text = _norm(text)
    if _about_other(text):
        return {}
    found = {}
    for name, (keys, patterns) in NUM_FACTS.items():
        if not any(k in text for k in keys):
            continue
        for p in patterns:
            m = re.search(p, text)
            if m:
                found[name] = int(m.group(1))
                break
    return found


def _has(q, *words):
    return any(w in q for w in words)


def _calc_plan(q):
    """判斷題目是哪一種計算題，回傳 (需要的事實, 計算函式, 單位)；不是計算題就回傳 None。
    先認出題目在問誰（女兒、兒子、兩個小孩、房租、兄弟姊妹），再看要做什麼運算；
    兩者對不上就回傳 None，寧可退回一般檢索，也不要放進錯的計算結果。"""
    q = _norm(q)
    dau, son = _has(q, "女兒", "閨女"), "兒子" in q
    both = (dau and son) or _has(q, "兒女", "小孩", "孩子")
    rent = _has(q, "租", "房東")
    sib = _has(q, "兄弟姊妹", "手足")
    include_me = _has(q, "包含我", "包括我", "連我", "算上我", "加上我", "含我", "算進我")
    if not (dau or son or rent) and (
            include_me
            or (sib and _has(q, "我和", "我跟", "加起來", "總共", "一共"))
            or (_has(q, "小孩", "孩子", "爸媽", "父母") and "幾個" in q)):
        return ("siblings",), lambda f: f["siblings"] + 1, "個"
    if rent and _has(q, "一年", "每年", "整年", "全年", "年租", "12個月", "十二個月"):
        return ("rent",), lambda f: f["rent"] * 12, "元"
    if dau and not son and _has(q, "幾年", "多少年", "多久", "成年", "18") and "年級" not in q:
        m = re.search(r"(\d+)\s*歲", q)
        target = int(m.group(1)) if m else 18   # 「成年」或沒寫歲數時視為 18
        return ("daughter",), lambda f: target - f["daughter"], "年"
    if son and not dau and _has(q, "明年", "後年", "下一年", "再過一年", "一年後", "過完今年"):
        later = 2 if "後年" in q else 1
        return ("son",), lambda f: f["son"] + later, "歲"
    if both and _has(q, "差", "大幾歲", "小幾歲"):
        return ("daughter", "son"), lambda f: abs(f["daughter"] - f["son"]), "歲"
    # 加總：不能只看「加」，否則「參加」「加入」也會被當成計算題。
    if both and (re.search(r"加(?:起來|總|一加|上)|相加|(?:女兒|閨女|兒子)加", q)
                 or _has(q, "總共", "合計", "一共", "總和", "合起來")):
        return ("daughter", "son"), lambda f: f["daughter"] + f["son"], "歲"
    return None


# ---------------------------------------------------------------- 過濾閒聊與問句（改進 2）
# 疑問詞與虛詞：幾乎每個問句和閒聊都有，留在查詢裡只會把不相關的句子拉進來
# （例如「我叫什麼名字？」靠「什麼」對到「晚餐不知道要吃什麼」）。長的詞要排在短的前面。
QUERY_STOPWORDS = ("你記得", "你知道", "請問", "什麼", "哪裡", "哪個", "哪一", "哪", "多少", "幾",
                   "嗎", "呢", "是", "的", "我")


def _is_question(text):
    """整句都是問句（「你有推薦的書嗎？」）：使用者在問 agent，不是在講自己的事。
    有逗號的不算（「我住台南，你去過嗎？」前半句是事實），寧可多存也不要漏存。"""
    return text.endswith(("？", "?")) and not re.search(r"[，,。；]", text)


def _query_bigrams(question):
    """把疑問詞與標點換成分隔符號，再切成 bigram；跨過分隔符號的 bigram 不算。"""
    q = question
    for w in QUERY_STOPWORDS:
        q = q.replace(w, "|")
    q = re.sub(r"[？?！!，,。、；\s]", "|", q)
    return {q[i:i + 2] for i in range(len(q) - 1) if "|" not in q[i:i + 2]}


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
        text = msg.strip()
        if _is_question(text):
            return ""
        # 同一句話重複出現（閒聊常常重複）只留最新的一筆，避免重複的句子佔掉 300 字。
        self.memory = [m for m in self.memory if m["text"] != text]
        self.memory.append({"date": self.date, "text": text})
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
        calc = self._calc(question)
        if calc:
            return calc

        q_bigrams = _query_bigrams(question)

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

    def _calc(self, question):
        """計算題：從記憶抓出數字並算好，回傳「計算結果：19歲」；不是計算題或數字不齊就回傳 ""。
        依時間順序掃過記憶，後說的會覆蓋先說的，所以房租更新後會用新的值。
        只回傳結果這一行、不附上原句：「我兒子明年幾歲？」的選項包含兒子現在的年紀，
        原句裡的數字反而會把模型拉向錯的選項。"""
        plan = _calc_plan(question)
        if plan is None:
            return ""
        needs, compute, unit = plan
        facts = {}
        for m in self.memory:
            facts.update(_extract_nums(m["text"]))
        if not all(k in facts for k in needs):
            return ""
        return f"計算結果：{compute(facts)}{unit}"

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
