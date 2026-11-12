# -*- coding: utf-8 -*-
"""HW2 SFT：用 build_sft_data.py 產生的資料訓練 Hugging Face 格式的 MiniMind。

訓練迴圈、學習率排程（get_lr）、混合精度、梯度累積與裁剪、DDP、log 與 swanlab 都與
minimind/trainer/train_full_sft.py 相同，工具函式直接 import 自 minimind 子模組。不同之處：
  1. 資料：每筆是 (prompt, target)，只有 target 計算 loss。prompt 用評測時相同的程式產生：
     tool    tokenizer.apply_chat_template(messages, tools, add_generation_prompt=True)，同 BaseAgent.generate()；
             target 後面接 eos（<|im_end|>），讓模型學會呼叫完就停。
     reading eval_agent.build_prompt()（以「答案：」結尾）；prompt 與選項分開 tokenize 再接起來，
             與 eval_agent.score_options() 算分的序列逐 token 相同。
  2. 模型：直接讀寫 Hugging Face 格式（from_pretrained / save_pretrained）。官方 minimind-3 與 HW1 經
     convert_model.py 轉出的模型都是 Qwen3ForCausalLM 格式，存出來的格式與起始模型相同，
     不需要 HW1 eval.sh 的 .pth → convert_model.py 轉換；輸出目錄可以直接給 eval_agent.py 使用與繳交。
     存檔後套用與 convert_model.py 相同的 transformers 5.x 相容處理，讓低版本 transformers 也能載入。
     Qwen3（非 MoE）的輸出沒有 aux_loss，此時 loss 只用 res.loss。
  3. 沒有 from_resume（資料量小，重新訓練即可）。
  4. 每個 batch 只補齊到該 batch 最長的長度（minimind 一律補到 max_seq_len）。
  5. prompt + target 超過 max_seq_len 時直接報錯；minimind 會截斷，但截斷會把 target 切掉。

用法（在 student_release/ 下執行；minimind 子模組預設在 ../../minimind，可用環境變數 MINIMIND_DIR 指定）：
  # E2：只用 tool 資料
  python train_sft.py --model_dir ../minimind-3 --data_path sft_data/sft_tool.jsonl --save_dir ../models/E2
  # E3：tool + reading 混合
  python train_sft.py --model_dir ../minimind-3 --data_path sft_data/sft_tool.jsonl sft_data/sft_reading.jsonl \
      --save_dir ../models/E3
"""
import os
import sys

MINIMIND_DIR = os.environ.get(
    "MINIMIND_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "minimind"))
sys.path.append(os.path.abspath(MINIMIND_DIR))

import argparse
import json
import time
import warnings
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from eval_agent import build_prompt
from trainer.trainer_utils import (Logger, SkipBatchSampler, get_lr, init_distributed_mode, is_main_process,
                                   setup_seed)

warnings.filterwarnings('ignore')


class HW2SFTDataset(Dataset):
    def __init__(self, paths, tokenizer, max_length):
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("tokenizer 沒有 chat_template，無法產生與評測相同的 prompt")
        self.samples, too_long = [], []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    prompt_ids, target_ids = self.encode(row, tokenizer)
                    n = len(prompt_ids) + len(target_ids)
                    if n > max_length:
                        too_long.append(n)
                        continue
                    self.samples.append((prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids))
        if too_long:
            raise ValueError(f"{len(too_long)} 筆超過 max_seq_len={max_length}（最長 {max(too_long)} tokens），"
                             f"請調大 --max_seq_len")

    @staticmethod
    def encode(row, tok):
        if row["task"] == "tool":
            prompt = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False,
                                             add_generation_prompt=True)
            target = row["target"] + tok.eos_token
        elif row["task"] == "reading":
            prompt = build_prompt(tok, row["context"], row["question"], use_chat=True)
            target = row["target"]
        else:
            raise ValueError(f"未知的 task：{row['task']}")
        return (tok(prompt, add_special_tokens=False).input_ids,
                tok(target, add_special_tokens=False).input_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(batch, pad_id):
    """右側補齊到 batch 內最長的長度；補上的位置 label 為 -100，causal 模型不會讓它影響前面的位置。"""
    length = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), length), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)
    return input_ids, labels


def save_model():
    model.eval()
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    raw_model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    # ======= 與 minimind/scripts/convert_model.py 相同：transformers-5.0 的相容低版本寫法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(args.save_dir, "tokenizer_config.json")
        config_path = os.path.join(args.save_dir, "config.json")
        with open(tokenizer_config_path, encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        tokenizer_config.update(tokenizer_class="PreTrainedTokenizerFast", extra_special_tokens={})
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
        if 'rope_parameters' in config:
            config['rope_theta'] = config['rope_parameters']['rope_theta']
            config['rope_scaling'] = None
            del config['rope_parameters']
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    model.train()


def train_epoch(epoch, loader, iters, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            aux_loss = getattr(res, "aux_loss", None)  # MiniMind MoE 才有；Qwen3 格式沒有
            loss = res.loss if aux_loss is None else res.loss + aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item() if aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / step * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, '
                   f'logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                   f'lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss,
                                 "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_model()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind SFT for HW2")
    parser.add_argument("--model_dir", type=str, required=True, help="Hugging Face 格式的起始模型目錄")
    parser.add_argument("--data_path", type=str, nargs="+", required=True,
                        help="build_sft_data.py 產生的 jsonl，可給多個（合併後每個 epoch 一起打亂）")
    parser.add_argument("--save_dir", type=str, required=True, help="輸出的 Hugging Face 模型目錄")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--max_seq_len', default=768, type=int, help="单笔 prompt + target 的最大长度")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-HW2-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录 ==========
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type="cuda", dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"HW2-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))

    # ========== 5. 定义模型、数据、优化器 ==========
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True, dtype=torch.float32)
    model = model.to(args.device)
    model.train()
    Logger(f'Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    train_ds = HW2SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    lengths = [len(ids) for ids, _ in train_ds.samples]
    Logger(f'Samples: {len(train_ds)}, max length: {max(lengths)}, mean length: {sum(lengths) / len(lengths):.0f}')
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 7. 开始训练 ==========
    collate_fn = partial(collate, pad_id=tokenizer.pad_token_id)
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, 0)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate_fn)
        train_epoch(epoch, loader, len(loader), wandb)

    # ========== 8. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
