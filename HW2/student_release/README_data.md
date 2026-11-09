# HW2 資料說明

## 檔案
| 檔案 | 內容 |
|---|---|
| `train/memory.jsonl` | 300 個虛構人物，3000 題（可用於訓練） |
| `train/tool.jsonl` | 3000 筆工具呼叫範例（可直接改成 SFT 資料） |
| `val/memory.jsonl` | 30 個人物，300 題（本地評測） |
| `val/tool.jsonl` | 200 筆 |
| `tool_spec.json` | 工具定義與呼叫格式 |
| `gen_hw2_data.py`, `templates_public.py` | 生成腳本，可以自己產生更多訓練資料 |
| `check_data.py` | 統計與簡單 baseline |

私有 test set（100 人物、1000 題、300 筆工具測試）由助教保留，**會使用 train/val 沒出現過的說法、值與屬性**，背模板不會有用。

## memory.jsonl（一行一個人物）
- `sessions[].turns`：使用者依序說的話（只有使用者，agent 的回覆由你的系統產生）
- `sessions[].annotations`：哪一句是事實（`fact`）、更新（`update`）、別人的資訊（`other`），可用來造 SFT 資料
- `profile`：這個人最終的正確資訊
- `questions[]`：`type`、`question`、`options`（A–D；除了 multi 題之外都有「你沒有提過」）、`answer`、`evidence`（[session_id, turn]）

題型：`single` 單一事實、`update` 資訊更新、`multi` 多事實整合、`abstain` 沒說過要拒答、`distractor` 有別人的同類資訊干擾、`tool` 查記憶再計算。

## tool.jsonl（一行一筆）
- `context`：使用者連續說的話，最後一句是要回應的
- `expected.action`：`call` 或 `answer`（不該呼叫工具）
- `expected.accept`：評分標準。key / query 包含任一關鍵詞即算對；value 需包含正確值；calculator 比對算式的計算結果

## 評分指標
- **Memory Accuracy**：6 種題型各自準確率的平均（macro）
- **排行榜分數** = 0.7 × Memory Accuracy + 0.3 × Tool End-to-End
- 工具呼叫：Format Validity、Decision Accuracy、Tool Accuracy、Argument Match、End-to-End（格式見 `tool_spec.json`，與 MiniMind 原生的 `<tool_call>` 格式相同）

## 重現資料
資料由以下指令產生，同一個 seed 每次結果完全相同（可用 `md5sum -c MD5SUMS_public` 核對）：
```bash
python gen_hw2_data.py --split train --seed 1 --n_personas 300 --n_tool 3000 --out data/train
python gen_hw2_data.py --split val   --seed 2 --n_personas 30  --n_tool 200  --out data/val
```
