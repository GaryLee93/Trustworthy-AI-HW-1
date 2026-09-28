"""
把 MiniMind 的簡體 pretrain 資料轉成繁體（OpenCC），再和維基繁中資料合併、打散，
輸出一份可直接給 train_pretrain.py 使用的 jsonl（每行 {"text": "..."}）。

安裝：
    pip install opencc-python-reimplemented tqdm

範例：
    python3 pretrain_to_tw.py \
        --minimind /tmp/b11902090/Trustworthy-AI-HW-1/minimind/dataset/pretrain_t2t_mini.jsonl \
        --wiki /tmp/b11902090/Trustworthy-AI-HW-1/minimind/dataset/zhtw_wikipedia_pretrain.jsonl \
        --output /tmp/b11902090/Trustworthy-AI-HW-1/minimind/dataset/pretrain_tw_merged.jsonl \
        --wiki_key text \
        --seed 824
"""
import argparse
import json
import os
import random
import tempfile
from multiprocessing import Pool

from tqdm import tqdm

_cc = None


def _init_worker(config):
    global _cc
    try:
        from opencc import OpenCC
        try:
            _cc = OpenCC(config)          # opencc-python-reimplemented: 's2twp'
        except Exception:
            _cc = OpenCC(config + '.json')  # 新版 opencc: 's2twp.json'
    except ImportError:
        raise SystemExit('請先安裝 OpenCC：pip install opencc-python-reimplemented')


def _process(args):
    """處理一行：解析 -> (可選)轉繁體 -> 回傳 json 字串。無效行回傳 None。"""
    line, text_key, convert = args
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        text = obj.get(text_key) if isinstance(obj, dict) else str(obj)
    except json.JSONDecodeError:
        text = line  # 不是 JSON，就當作純文字一行
    if not text or not text.strip():
        return None
    if convert:
        text = _cc.convert(text)
    return json.dumps({'text': text}, ensure_ascii=False)


def convert_file(path, out_f, text_key, convert, config, workers, chunksize=2000):
    """串流處理一個檔案並寫入 out_f，回傳寫入行數。"""
    def gen():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield (line, text_key, convert)

    n = 0
    with Pool(workers, initializer=_init_worker, initargs=(config,)) as pool:
        for res in tqdm(pool.imap(_process, gen(), chunksize=chunksize), desc=os.path.basename(path)):
            if res is not None:
                out_f.write(res + '\n')
                n += 1
    return n


def shuffle_file(src, dst, seed):
    """不把整份資料讀進記憶體：只記錄每行 byte offset，打散後依序讀寫。"""
    offsets = []
    with open(src, 'rb') as f:
        pos = 0
        for line in f:
            offsets.append(pos)
            pos += len(line)
    random.Random(seed).shuffle(offsets)
    with open(src, 'rb') as fin, open(dst, 'wb') as fout:
        for off in tqdm(offsets, desc='shuffle'):
            fin.seek(off)
            fout.write(fin.readline())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--minimind', required=True, help='MiniMind 簡體 pretrain jsonl')
    p.add_argument('--wiki', default=None, help='維基繁中資料（jsonl 或每行一段純文字）')
    p.add_argument('--output', required=True, help='合併後輸出的 jsonl')
    p.add_argument('--minimind_key', default='text')
    p.add_argument('--wiki_key', default='text')
    p.add_argument('--opencc_config', default='s2twp', help='s2twp=簡轉台灣繁體並轉換慣用詞；s2tw=僅字形')
    p.add_argument('--convert_wiki', action='store_true', help='維基資料若仍是簡體才加此參數')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile('w', delete=False, suffix='.jsonl', encoding='utf-8',
                                      dir=os.path.dirname(os.path.abspath(args.output)))
    try:
        with tmp as out_f:
            n1 = convert_file(args.minimind, out_f, args.minimind_key, True, args.opencc_config, args.workers)
            print(f'MiniMind（已轉繁體）: {n1} 行')
            n2 = 0
            if args.wiki:
                n2 = convert_file(args.wiki, out_f, args.wiki_key, args.convert_wiki, args.opencc_config, args.workers)
                print(f'維基: {n2} 行（轉換={args.convert_wiki}）')
        shuffle_file(tmp.name, args.output, args.seed)
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
    print(f'完成：共 {n1 + n2} 行 -> {args.output}（seed={args.seed}）')


if __name__ == '__main__':
    main()