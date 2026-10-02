## pretrain_t2t_mini.jsonl and sft_t2t_mini.jsonl
### Basic Info: pretrain_t2t_mini.jsonl
- **Source**: [`jingyaogong/minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
- **Version**: latest `main` (no revision pinned)
- **License**: Apache-2.0 / CC-BY-NC-2.0 (as listed on the dataset card)
- **Language**: mainly Simplified Chinese, mixed with English
- **Size**: ~1.2GB
- **Content**: lightweight pretraining corpus for quick reproduction: general text, curated dialogue text and distilled supplementary text (sources include 匠數大模型數據集 and Magpie-Align); recommended max length ~768 tokens
- **Format**: `{"text": "..."}` per line (MiniMind pretrain)
- **Processing**: none, used as-is

### Basic Info: sft_t2t_mini.jsonl
- **Source**: [`jingyaogong/minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
- **Version**: latest `main` (no revision pinned)
- **License**: Apache-2.0 / CC-BY-NC-2.0 (as listed on the dataset card)
- **Language**: mainly Simplified Chinese, mixed with English
- **Size**: ~1.6GB
- **Content**: lightweight SFT data for training a chat model: instruction data, dialogue data and model-distilled outputs, with tool-calling samples mixed in (sources include COIG and Step-3.5-Flash-SFT); recommended max length ~768 tokens
- **Format**: `{"conversations": [{"role": "...", "content": "..."}, ...]}` per line (MiniMind SFT, multi-turn)
- **Processing**: none, used as-is

### Download Command
```bash
# after the execution of setup.sh
cd <Project_Dir>
mkdir -p Trustworthy-AI-HW-1/minimind/dataset
hf download jingyaogong/minimind_dataset --repo-type dataset \
    --include "pretrain_t2t_mini.jsonl" "sft_t2t_mini.jsonl" --local-dir .
```
## zhtw_wikipedia_pretrain.jsonl
### Basic Info
- **Source**: [`zetavg/zh-tw-wikipedia`](https://huggingface.co/datasets/zetavg/zh-tw-wikipedia)
- **Version**: latest `main` (no revision pinned); Wikipedia snapshot collected 2023-05-01 ~ 2023-05-07
- **License**: not stated on the dataset card; Wikipedia text is licensed CC BY-SA 4.0
- **Language**: Traditional Chinese (Taiwan usage)
- **Size**: source 2,533,212 articles, 8.19GB (all columns, 44 parquet shards); output capped at ~2048MB
- **Content**: Traditional Chinese Wikipedia articles; only the `markdown` column is used, cleaned into plain text and split into ≤400-character chunks
- **Format**: `{"text": "..."}` per line (MiniMind pretrain)
- **Processing**: `prepare_wiki_pretrain.py` (see below)

### Data Processing
1. **Shard order shuffling**: the 44 auto-converted parquet shards are visited in an order shuffled by `--seed`.
2. **Column-pruned reading**: each shard is downloaded to a temp file and read with `pyarrow`, decoding **only** the `markdown` column (in batches of `--batch_size` rows), so the large `html` column is never loaded into memory. Each shard is deleted right after it is processed, so at most one shard (~180MB) is on disk at a time.
3. **Within-batch shuffling**: row order inside each read batch is shuffled with the same seeded RNG.
4. **Markdown to plain text**: remove headers (`#`), bold (`**`), italic (`*`), underscore emphasis (`_`), and collapse consecutive blank lines.
5. **Chunking**: MiniMind's pretrain dataset truncates every line to `max_seq_len` tokens, so writing whole articles would drop everything after the opening paragraph. Each article is therefore split into chunks of at most `--max_chars` characters:
   - whole paragraphs are greedily packed into a chunk until it would exceed `--max_chars`;
   - a single paragraph longer than `--max_chars` is cut at the last sentence-ending punctuation (`。！？；` or newline) within the window, with a hard cut only if no punctuation exists.
   - `--max_chars 400` was chosen from the measured tokenizer ratio (~1.19 tokens/char for Traditional Chinese), i.e. ~480 tokens, to fit `max_seq_len` used in pretraining.
6. **Length filter**: chunks shorter than `--min_length` characters (stubs / tail fragments) are dropped.
7. **Size cap**: writing stops once the output reaches `--target_mb` MB (UTF-8 bytes).

Parameters used:

| Argument | Value |
|---|---|
| `--target_mb` | 2048 |
| `--seed` | 824 |
| `--min_length` | 150 |
| `--max_chars` | 400 |
| `--batch_size` | 512 (default) |

### Generation Command
```bash
# after the execution of setup.sh
cd <Project_Dir>
python prepare_wiki_pretrain.py \
    --target_mb 2048 \
    --seed 824 \
    --min_length 150 \
    --max_chars 400 \
    --out_path minimind/dataset/zhtw_wikipedia_pretrain.jsonl
```

## tmmluplus_sft.jsonl
### Basic Info
- **Source**: [`ikala/tmmluplus`](https://huggingface.co/datasets/ikala/tmmluplus)
- **Version**: revision `v1.1`
- **License**: MIT
- **Language**: Traditional Chinese (Taiwan context)
- **Size**: source 66 subjects, 22,160 questions (train 321 / validation 2,202 / test 19,680); output uses train + ~80% of validation, test is never used
- **Content**: 4-option (A/B/C/D) single-choice exam questions across STEM, social sciences, humanities and professional fields, converted into single-turn Q&A
- **Format**: `{"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}` per line (MiniMind SFT)
- **Processing**: `prepare_tmmluplus_sft.py` (see below)

### Data Processing
1. **Load only train + validation**: all subjects are discovered from the repo's `data/*_train.csv` files; for each subject, `{subj}_train.csv` and `{subj}_val.csv` are downloaded. **The test split is never downloaded or used.**
2. **Stratified holdout split**: for each subject (in sorted order, with a single RNG seeded by `--seed`), the validation set is shuffled and `max(1, round(len(val) * holdout_frac))` questions are set aside as a local evaluation holdout. The holdout is never written into the SFT file.
3. **Training pool** = all train rows + the remaining (non-holdout) validation rows.
4. **Cleaning**: every field is stripped; floats that are whole numbers (e.g. `3.0` parsed by pandas) are converted back to `"3"`. Rows whose answer is not one of `A/B/C/D` are skipped.
5. **Conversion to conversations**, using the prompt template imported directly from `eval_tmmluplus.py` (`INSTRUCTION`, `ANSWER_PREFIX`, `format_question`, `SUBJECTS`, `LABELS`) so that the training format is identical to the evaluation format (chat-template mode):
   - user: `以下是關於{subject_zh}的單選題，請直接給出正確答案的選項。\n\n{question}\nA. ...\nB. ...\nC. ...\nD. ...` (everything up to, but not including, the trailing `答案：`)
   - assistant: `答案：{letter}. {option text}`, e.g. `答案：B. 選項內容` (with `--no_answer_text`, only `答案：B`)

Parameters used:

| Argument | Value |
|---|---|
| `--seed` | 824 |
| `--holdout_frac` | 0.2 |
| `--revision` | v1.1 |
| `--no_answer_text` | not set (assistant answer includes option text) |

### Generation Command
```bash
# after the execution of setup.sh
cd <Project_Dir>
python prepare_tmmluplus_sft.py \
    --seed 824 \
    --holdout_frac 0.2 \
    --revision v1.1 \
    --holdout_out_path minimind/dataset/tmmluplus_holdout_check.jsonl \
    --out_path minimind/dataset/tmmluplus_sft.jsonl
```
`--holdout_out_path` is optional; it writes the held-out questions (raw fields + `subject`) used for local evaluation, never for training.