# HW2 評測程式

## 需求
`pip install torch transformers`。模型要是 Hugging Face 格式（HW1 用 MiniMind 的 `scripts/convert_model.py` 轉過的那份），
`AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)` 能載入即可。

## 學生要寫的檔案
複製 `my_agent.py` 開始改，評測：`python eval_agent.py --model_dir <模型> --agent my_agent:MyAgent --data_dir data/val`。不要修改 `eval_agent.py`、`agent_base.py`、`tool_spec.json`。

## 跑 baseline
```bash
# 無記憶：prompt 只有題目
python eval_agent.py --model_dir <你的模型> --agent agent_base:NoMemoryAgent --data_dir data/val
# 全部串接，截到最後 300 字（等於只看得到最近的對話）
python eval_agent.py --model_dir <你的模型> --agent agent_base:RecentContextAgent --data_dir data/val
# 框架寫死的關鍵字檢索
python eval_agent.py --model_dir <你的模型> --agent agent_base:KeywordRetrievalAgent --data_dir data/val
# Oracle：直接給標準答案的證據句（題目難度上限，只有助教用）
python eval_agent.py --model_dir <你的模型> --oracle --data_dir data/val
```
除錯時可加 `--limit 5 --limit_tool 50` 只跑一小部分。

## 規則
- `prepare()` 回傳的記憶內容只保留**最後 300 字**（`--max_context_chars`），題目與選項由評測程式套固定模板。
- 打分（預設 `--score option`）：比較四個選項內容的平均 log-likelihood。
  `letter`（A–D 字母機率）在 64M 模型上會一律選同一個字母；`option_calib`（減去無題目時的機率）實測效果較差，兩者都不建議。
- 工具呼叫：`next_action()` 的輸出用 `<tool_call>{"name":…,"arguments":…}</tool_call>` 解析（也接受 `<call>名稱{…}</call>`）。

## 輸出
`results/<agent>_<split>/summary.json`（總分）、`memory_pred.jsonl`（每題預測與實際放進 prompt 的記憶）、`tool_pred.jsonl`（每筆模型原始輸出）。
