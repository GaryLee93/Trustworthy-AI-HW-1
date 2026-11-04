# TAI HW2 摘要：把 MiniMind 做成有記憶、會用工具的 AI Agent

**日期與配分**：2026/10/30 發布，**2026/11/13 截止**，佔總成績 10%。

> 附註：PDF 第 6 頁的表格文字擷取雜亂，但方法、呼叫時機、作用三欄已與原文對照過，「介面」一節內容一致。

## 0. 前兩頁：誠信與 AI 使用聲明
- 需要填課程、作業、姓名、學號、簽名、日期。
- 允許使用生成式 AI，但不能直接交 AI 的產出。
- 要自己審閱、驗證、修改，並能解釋提交的內容。
- 必要時要揭露使用了哪些 AI 工具。

## 1. 目標
把 ≤128M 參數的小模型（你的 HW1 模型或官方 minimind-3）變成 Agent。Agent 由四部分組成：
- 模型
- 記憶
- 工具
- 行動迴圈（存事實、搜尋記憶、呼叫工具，或直接回答）

評測是繁體中文的多場次記憶 benchmark，會和同學排名。

學習重點：
- 設計記憶的寫入、讀取、更新和拒答（abstain）策略。
- 用 SFT 或 LoRA 教模型固定的 tool-call 格式。
- 用 ablation 區分「框架設計」和「模型訓練」各自的貢獻。
- 觀察 agent 微調造成的災難性遺忘（用 HW1 的 TMMLU+ held-out 測）。

## 2. 環境
1. 下載 `hw2_student_release.zip`。裡面有 data、`eval_agent.py`、`agent_base.py`、`tool_spec.json`、資料產生器、`my_agent.py`。
2. 安裝 `pip install torch transformers`。
3. 準備 HF 格式的模型：HW1 的 `model/`，或 `git clone` minimind-3。
4. 跑 baseline 確認環境：
   `python eval_agent.py --model_dir <model> --agent agent_base:NoMemoryAgent --data_dir data/val`
5. 算力：val 評測約 1–2 分鐘；64M 模型的 SFT 不到 1 小時。

## 3. 資料（合成、繁體中文）
每個 persona 是虛構使用者，有 8 個 session、約 128 句話。事實散落在閒聊中，部分事實之後會被更新，部分句子在講別人。結束後問 10 題四選一。

| Split | 規模 | 用途 |
|---|---|---|
| train | 300 personas / 3000 題 / 3000 筆 tool-call | 隨便用 |
| val | 30 personas / 300 題 / 200 筆 tool-call | 本地評估，**不可拿來訓練** |
| test（私有） | 100 personas / 1000 題 / 300 筆 tool-call | 助教持有，用了 train/val 沒出現過的說法、值與屬性 |

**六種題型（各自獨立報告）：**
- **single**：跨 session 回憶單一事實。
- **distractor**：附近有別人的類似事實（「我朋友住高雄」）。
- **update**：新事實覆蓋舊事實，舊值一定在選項裡。
- **multi**：組合兩個事實。
- **abstain**：使用者沒說過，答案是「你沒有提過」。
- **tool**：從記憶取數字再計算。

**Tool-call 資料**：給幾句話和預期動作，動作是 `memory_save`、`memory_search`、`calculator`、`get_date`，或不呼叫。完整格式見 `README_data.md`。

**擴充資料**：可以用別的 seed 跑 `gen_hw2_data.py`、自寫模板，或用大 LLM 生成，但每個來源都要列進 data manifest。因為 test 的說法是新的，背模板沒用，要追求多樣性。

## 4. Agent 介面（`my_agent.py`，繼承 `BaseAgent`）
**不要改** `eval_agent.py`、`agent_base.py`、`tool_spec.json`，助教會用原版。

| 方法 | 呼叫時機 | 作用 |
|---|---|---|
| `reset()` | 每個 persona 前 | 清空記憶 |
| `new_session(date)` | 每個 session 開始 | 記錄日期（可選） |
| `chat(msg)` | 每句使用者話語 | 決定要不要存、存什麼 |
| `prepare(question)` | 每道題 | 回傳要放進 prompt 的記憶文字（不含選項） |
| `next_action(context)` | 每筆 tool-call 題 | 回傳模型原始輸出，含 `<tool_call>` |
| `save_snapshot()` / `load_snapshot()` | persona 後 / 每題前 | 存取記憶，讓各題互不影響 |

**Tool call 格式**（MiniMind 原生）：

```
<tool_call>{"name": "memory_save", "arguments": {"key": "居住地", "value": "台南"}}</tool_call>
```

**硬性規則：**
- `prepare()` 回傳的內容**只取最後 300 字元**，所以必須自己挑重點。
- 只存使用者自己的資訊，別人的不存。
- 主模型 ≤128M；輔助 embedding 模型 ≤30M，或只用 BM25。
- 不可呼叫外部 API，記憶必須存在本地。
- 完整私有評測要在 **30 分鐘**內跑完。

## 5. 任務
**Task 1 Baselines（必做）**：用你的模型在 val 上跑三個參考 agent：
- `NoMemoryAgent`：只給問題。
- `RecentContextAgent`：全部歷史截到最後 300 字。
- `KeywordRetrievalAgent`：全存，用關鍵字重疊檢索。

**Task 2 建 memory agent（必做）**：目標是打敗 baseline。可考慮的方向：
- **寫入**：只存重要句子，或讓模型輸出 key-value 的 `memory_save`。同 key 覆蓋舊值，直接處理 update 題。
- **檢索**：BM25、小型 embedding，或讓模型產生 `memory_search` query。300 字內的排序與格式很重要。
- **拒答**：檢索不到時，讓模型容易選「你沒有提過」。
- **計算**：tool 題在 `prepare()` 內先查數字、呼叫計算機，把「計算結果：19 歲」放進記憶文字。
- **Tool-call SFT**：minimind-3 會輸出 `<tool_call>`，但幾乎都用錯參數的 `memory_save`。用 `tool.jsonl` 教它何時存、搜、算或不呼叫。
- **Reading SFT**：即使給了正確句子，官方 64M 模型也只答對約 41%。用 `memory.jsonl` 建（記憶 + 問題 → 答案）的訓練資料。
- **小模型技巧**：special tokens、constrained decoding、對候選答案打分、從使用者句子複製參數、解析失敗時的 fallback 規則。

**Task 3 報告（必做，最多 4 頁）**，需包含：
1. 架構圖，以及 write / read / update / abstain 各自的設計決策。
2. SFT 資料怎麼建，附例子。
3. val 結果表：baselines 對各實驗，按題型拆分，加 tool-call 指標。
4. Ablation，至少要有：no memory、keyword retrieval、你的 agent、你的 agent 有無 SFT。
5. TMMLU+ 迴歸：agent 微調前後的 HW1 held-out TMMLU+ 準確率。
6. 失敗分析：哪種題型最差、原因（格式錯、沒存、檢索錯…）。
7. 訓練成本：GPU 型號、總 GPU 小時。

## 6. 繳交
單一壓縮檔 `hw2_<student_id>.zip`，內含 `<student_id>/`：

```
<student_id>/
├── model/             # Hugging Face transformers 格式
├── code/              # my_agent.py、訓練腳本、資料生成、run.sh
├── data_manifest.md   # 每份資料集：來源、大小、產生方式
├── logs/              # 訓練 log
└── report.pdf
```

**要求：**
- 能用 `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` 載入。
- 主模型 ≤128M 參數。
- 在 `code/` 內執行下列指令要能無錯跑完：
  `python eval_agent.py --model_dir ../model --agent my_agent:MyAgent --data_dir <data>`
- 評測 30 分鐘內完成。
- `run.sh` 要能用 manifest 列的資料重現最終模型。

## 7. 評測方式
**記憶題的 prompt**：

```
以下是你記得的關於使用者的資訊：<memory text>

問題：<question>

A. <option>
B. <option>
C. <option>
D. <option>
請直接回答選項字母。答案：
```

**計分**：不直接看 A–D 字母，因為小模型容易全選同一個字母（minimind-3 在 val 全選 A）。改為對每個選項文字計算「每字元平均 log 機率」，取最高者：

`score(o) = (1/|o|) × Σᵢ log P(tᵢ | prompt, t₁…tᵢ₋₁)`

**Memory Accuracy**：六種題型各自算正確率，再做 macro-average，每種題型權重相同。隨機猜約 25%。

**Tool-call 指標**（N 為總數，N_call 為預期要呼叫的數量，N_out 為實際輸出含呼叫的數量）：

| 指標 | 意義 |
|---|---|
| Format validity | 輸出含呼叫的項目中，能被解析的比例（JSON 合法，name 為字串，arguments 為物件） |
| Decision acc. | 該呼叫時有呼叫、不該呼叫時沒呼叫的比例 |
| Tool acc. | 需呼叫的項目中，選對工具的比例 |
| Argument acc. | 工具和參數都對的比例 |
| **End-to-end** | 需呼叫的項目參數都對，加上不需呼叫的項目沒呼叫，除以 N |

**參數比對規則**（內容嚴格，措辭寬鬆）：
- `memory_save`：key 含該屬性的可接受關鍵字（如 通勤/上班/交通），value 含預期值（「11 歲」也算 11）。
- `memory_search`：query 含可接受關鍵字（如 過敏）。
- `calculator`：expression 算出的數值正確即可（`12+6` 和 `6+12` 都對）。
- `get_date`：不需參數。

**官方 minimind-3 基準（val，未訓練）：**

| Agent | single | update | multi | abstain | distractor | tool | Mem Acc |
|---|---|---|---|---|---|---|---|
| NoMemory | 0.22 | 0.26 | 0.19 | 0.64 | 0.22 | 0.27 | 0.30 |
| Oracle（只有助教有） | 0.55 | 0.32 | 0.26 | 0.64 | 0.39 | 0.28 | 0.41 |

Oracle 是把標註的證據句直接放進 prompt，用來看未訓練模型的閱讀能力上限。

| Format validity | Decision | Tool | Argument | End-to-end |
|---|---|---|---|---|
| 0.56 | 0.63 | 0.22 | 0.00 | 0.06 |

**Leaderboard Score = 0.7 × Memory Accuracy + 0.3 × Tool End-to-End**。
minimind-3 + NoMemory 在 val 上約 0.7×0.30 + 0.3×0.06 ≈ **0.23**。`eval_agent.py` 每次跑完都會印出這個分數。

## 8. 配分
| 項目 | 比重 | 重點 |
|---|---|---|
| Baselines | 15% | 三個 baseline 正確跑完；你的 agent 能通過 `eval_agent.py` |
| Leaderboard | 35% | 私有 test 分數，分級給分 |
| Reproducibility | 15% | `run.sh` 可跑；manifest 完整（含 LLM 生成資料）；log 與模型一致 |
| Report | 35% | 清晰度、設計品質、實驗與 ablation、誠實分析 |

**Leaderboard 分級**（差距約 3 個百分點內視為平手）：
- 前 30% 得 35 分。
- 次 30% 得 30 分。
- 再次 30% 得 25 分。
- 其餘有效繳交得 20 分。
- **捷徑**：只要比你自己的 `KeywordRetrievalAgent`（同一個模型）高出 **≥3 個百分點**，不論排名都拿 35 分。

另外，若 agent 微調後 TMMLU+ 掉超過 5 個百分點，報告中要解釋原因。

## 9. 學術誠實
- 可以討論想法，但不可分享程式碼、模型權重或資料 pipeline。
- 可以用 AI 寫程式和生成訓練資料，但要在報告附錄和 data manifest 中揭露。
- 嘗試取得或使用私有 test set，leaderboard 和 reproducibility 兩項都零分。

## 10. 資源
- MiniMind：https://github.com/jingyaogong/minimind
- minimind-3：https://huggingface.co/jingyaogong/minimind-3
- LongMemEval：https://arxiv.org/abs/2410.10813
- LoCoMo：https://arxiv.org/abs/2402.17753

## 建議閱讀順序
1. 先看第 3 頁到第 5 頁的資料與題型範例。
2. 再看第 7 節的計分方式，因為它決定怎麼設計 `prepare()`。
3. 最後看 Task 2 的方向清單和第 6 節的繳交清單。

Leaderboard 分數的 70% 來自記憶準確率，所以 `prepare()` 的檢索品質影響最大。Tool End-to-End 只佔 30%，但 minimind-3 現在只有 0.06，用少量 SFT 就有很大的進步空間。
