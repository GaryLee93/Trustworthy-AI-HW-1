# -*- coding: utf-8 -*-
"""HW2 資料生成腳本：記憶測試（memory.jsonl）與工具呼叫測試（tool.jsonl）。

用法：
  python gen_hw2_data.py --split train --seed 1 --n_personas 300 --n_tool 3000 --out data/train
  python gen_hw2_data.py --split val   --seed 2 --n_personas 30  --n_tool 200  --out data/val
  python gen_hw2_data.py --split test  --seed 3 --n_personas 100 --n_tool 300  --out data/test   # 需要 templates_heldout.py

train / val 只用 templates_public.py；test 會另外混入 templates_heldout.py 的
沒公開說法、額外值池與新屬性。發給學生時請拿掉 templates_heldout.py。
"""
import argparse
import copy
import datetime as dt
import json
import os
import random

import templates_public as P

LETTERS = "ABCD"
QTYPES = ["single", "update", "multi", "abstain", "distractor", "tool"]


# ---------------------------------------------------------------- spec
def load_spec(split):
    attrs = copy.deepcopy(P.ATTRS)
    for a in attrs.values():
        for f in ("pool_h", "stmt_h", "upd_h", "q_h"):
            a[f] = []
        a["heldout_attr"] = False
    spec = dict(
        attrs=attrs,
        noise=list(P.NOISE), noise_h=[],
        multi=[dict(m, q_h=[]) for m in P.MULTI],
        calc={k: dict(v, q_h=[]) for k, v in P.CALC.items()},
        calc_direct=list(P.CALC_DIRECT), calc_direct_h=[],
        date_q=list(P.DATE_Q), date_q_h=[],
        p_held=0.0, p_pool_h=0.0,
    )
    if split == "test":
        import templates_heldout as H
        for k, extra in H.EXTRA.items():
            for f in ("pool", "stmt", "upd", "q"):
                attrs[k][f + "_h"] = list(extra.get(f, []))
        for k, a in H.NEW_ATTRS.items():
            a = copy.deepcopy(a)
            for f in ("pool_h", "stmt_h", "upd_h", "q_h"):
                a[f] = []
            a["heldout_attr"] = True
            attrs[k] = a
        spec["noise_h"] = list(H.NOISE)
        for m in spec["multi"]:
            m["q_h"] = H.MULTI_Q.get(tuple(m["attrs"]), [])
        for k, c in spec["calc"].items():
            c["q_h"] = H.CALC_Q.get(k, [])
        spec["calc_direct_h"] = list(H.CALC_DIRECT)
        spec["date_q_h"] = list(H.DATE_Q)
        spec["p_held"] = 0.6    # 句子用沒公開說法的機率
        spec["p_pool_h"] = 0.5  # 值從額外值池抽的機率
    return spec


def pick(rng, pub, held, p):
    if held and rng.random() < p:
        return rng.choice(held)
    return rng.choice(pub)


def all_values(a):
    if a["kind"] == "num":
        lo, hi, st = a["range"]
        return list(range(lo, hi + 1, st))
    return a["pool"] + a["pool_h"]


def sample_value(rng, spec, a, exclude=()):
    if a["kind"] == "num":
        cands = [v for v in all_values(a) if v not in exclude]
    else:
        use_h = a["pool_h"] and rng.random() < spec["p_pool_h"]
        cands = [v for v in (a["pool_h"] if use_h else a["pool"]) if v not in exclude]
        if not cands:
            cands = [v for v in all_values(a) if v not in exclude]
    return rng.choice(cands)


def disp(a, v):
    return a["fmt"].format(v=v) if a["kind"] == "num" else v


def canon(a, v):
    return str(v)


def stmt_text(rng, spec, a, v, kind="stmt"):
    t = pick(rng, a[kind], a[kind + "_h"], spec["p_held"])
    return t.format(v=v)


def other_text(rng, a, v):
    return a["other"].format(who=rng.choice(P.WHO), v=v)


def noise_line(rng, spec):
    return pick(rng, spec["noise"], spec["noise_h"], spec["p_held"] * 0.5)


# ---------------------------------------------------------------- persona
def make_persona(pid, rng, spec, n_sessions, turns_per_session):
    A = spec["attrs"]
    cat_keys = [k for k, a in A.items() if a["kind"] == "cat"]
    num_keys = [k for k, a in A.items() if a["kind"] == "num"]

    plan = ["single", "single", "tool", "tool", "update", "multi", "abstain", "distractor"]
    plan += rng.sample(["update", "multi", "abstain", "distractor"], 2)
    n = {t: plan.count(t) for t in QTYPES}

    # ---- 決定這個人有哪些屬性
    forced = set(num_keys)
    multis = rng.sample(spec["multi"], n["multi"])
    for m in multis:
        forced |= set(m["attrs"])
    updatable = [k for k in A if A[k]["updatable"]]
    upd_keys = rng.sample(updatable, n["update"] + (1 if rng.random() < 0.5 else 0))
    forced |= set(upd_keys)
    if "pet_name" in forced:
        forced.add("pet_type")

    n_unknown = 5
    target = len(num_keys) + len(cat_keys) - n_unknown
    rest = [k for k in cat_keys if k not in forced]
    rng.shuffle(rest)
    own = set(forced)
    for k in rest:
        if len(own) >= target:
            break
        if k == "pet_name" and "pet_type" not in own:
            continue
        own.add(k)
    unknown = [k for k in cat_keys if k not in own]

    # ---- 值
    first, latest = {}, {}
    for k in own:
        first[k] = sample_value(rng, spec, A[k])
        latest[k] = first[k]
    if latest["daughter_age"] == latest["son_age"]:
        latest["son_age"] = first["son_age"] = sample_value(
            rng, spec, A["son_age"], exclude=(latest["daughter_age"],))
    for k in upd_keys:
        latest[k] = sample_value(rng, spec, A[k], exclude=(first[k],))

    # ---- 題目要用的屬性
    upd_q_keys = upd_keys[:n["update"]]
    used = set(upd_keys)
    dist_keys = rng.sample([k for k in own if k not in used and A[k]["kind"] == "cat"], n["distractor"])
    used |= set(dist_keys)
    single_keys = rng.sample([k for k in own if k not in used], n["single"])
    abst_keys = rng.sample(unknown, n["abstain"])

    # ---- 事件
    events = []

    def add(kind, k, v, text, s):
        events.append(dict(kind=kind, key=k, value=v, text=text, session=s))

    fact_sess = {}
    for k in own:
        if k in upd_keys:
            s0 = rng.randint(0, n_sessions - 3)
            s1 = rng.randint(s0 + 1, n_sessions - 1)
            add("fact", k, first[k], stmt_text(rng, spec, A[k], first[k]), s0)
            add("update", k, latest[k], stmt_text(rng, spec, A[k], latest[k], "upd"), s1)
            fact_sess[k] = s0
        else:
            s0 = rng.randint(0, n_sessions - 1)
            add("fact", k, first[k], stmt_text(rng, spec, A[k], first[k]), s0)
            fact_sess[k] = s0

    others = {}  # key -> list of values said about other people
    for k in dist_keys:
        vs = []
        for j in range(2):
            v = sample_value(rng, spec, A[k], exclude=[first[k], latest[k]] + vs)
            vs.append(v)
            s = fact_sess[k] if j == 0 else rng.randint(0, n_sessions - 1)
            add("other", k, v, other_text(rng, A[k], v), s)
        others[k] = vs
    for k in abst_keys:
        if rng.random() < 0.5:
            v = sample_value(rng, spec, A[k])
            add("other", k, v, other_text(rng, A[k], v), rng.randint(0, n_sessions - 1))
            others[k] = [v]
    for _ in range(3):  # 背景干擾
        k = rng.choice([k for k in A if k not in dist_keys and k not in abst_keys])
        excl = [first.get(k), latest.get(k)]
        v = sample_value(rng, spec, A[k], exclude=[x for x in excl if x is not None])
        add("other", k, v, other_text(rng, A[k], v), rng.randint(0, n_sessions - 1))
        others.setdefault(k, []).append(v)

    # ---- 排成對話
    date = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 200))
    sessions, pos = [], {}
    for s in range(n_sessions):
        evs = [e for e in events if e["session"] == s]
        items = [("ev", e) for e in evs]
        n_noise = max(turns_per_session - len(evs), 3)
        noise = []
        while len(noise) < n_noise:  # 同一段對話內不重複
            x = noise_line(rng, spec)
            if x not in noise:
                noise.append(x)
        items += [("noise", x) for x in noise]
        rng.shuffle(items)
        turns, ann = [], []
        for i, (kind, x) in enumerate(items):
            if kind == "ev":
                turns.append(x["text"])
                ann.append(dict(turn=i, type=x["kind"], key=A[x["key"]]["label"],
                                value=canon(A[x["key"]], x["value"])))
                pos[id(x)] = [s + 1, i]
            else:
                turns.append(x)
        sessions.append(dict(session_id=s + 1, date=date.isoformat(), turns=turns, annotations=ann))
        date += dt.timedelta(days=rng.randint(2, 20))

    def evidence(keys, kinds=("fact", "update")):
        return [pos[id(e)] for e in events if e["key"] in keys and e["kind"] in kinds]

    # ---- 出題
    questions = []
    slots = [0, 1, 2, 3] * 3  # 讓正解位置在 A/B/C/D 間平均分布
    rng.shuffle(slots)

    def emit(qtype, qtext, correct, distractors, keys, ev):
        opts = list(distractors)
        assert len(opts) == 3 and len(set(opts + [correct])) == 4, (qtype, opts)
        rng.shuffle(opts)
        opts.insert(slots[len(questions)], correct)
        questions.append(dict(
            qid=f"{pid}_q{len(questions) + 1:02d}", type=qtype, question=qtext,
            options={L: o for L, o in zip(LETTERS, opts)},
            answer=LETTERS[opts.index(correct)],
            keys=[A[k]["label"] for k in keys], evidence=ev))

    def rand_vals(k, n_, exclude):
        a = A[k]
        excl = set(exclude)
        out = []
        while len(out) < n_:
            v = sample_value(rng, spec, a, exclude=list(excl))
            out.append(disp(a, v))
            excl.add(v)
        return out

    def known_vals(k):
        return [x for x in (first.get(k), latest.get(k)) if x is not None] + others.get(k, [])

    def q_of(k):
        return pick(rng, A[k]["q"], A[k]["q_h"], spec["p_held"])

    for k in single_keys:
        a = A[k]
        emit("single", q_of(k), disp(a, latest[k]),
             rand_vals(k, 2, known_vals(k)) + [P.ABSTAIN], [k], evidence([k]))
    for k in upd_q_keys:
        a = A[k]
        emit("update", q_of(k), disp(a, latest[k]),
             [disp(a, first[k])] + rand_vals(k, 1, known_vals(k)) + [P.ABSTAIN], [k], evidence([k]))
    for k in dist_keys:
        a = A[k]
        emit("distractor", q_of(k), disp(a, latest[k]),
             [disp(a, v) for v in others[k][:2]] + [P.ABSTAIN], [k], evidence([k]))
    for k in abst_keys:
        a = A[k]
        vs = [disp(a, v) for v in others.get(k, [])[:1]]
        vs += rand_vals(k, 3 - len(vs), others.get(k, []))
        emit("abstain", q_of(k), P.ABSTAIN, vs, [k], [])
    for m in multis:
        ka, kb = m["attrs"]
        a, b = A[ka], A[kb]

        def alt(k):
            if first[k] != latest[k]:
                return disp(A[k], first[k])
            return rand_vals(k, 1, known_vals(k))[0]
        va, vb = disp(a, latest[ka]), disp(b, latest[kb])
        f = m["opt"].format
        # 2×2 組合：每個成分都恰好出現兩次，無法靠「哪個成分出現最多次」猜答案
        aa, bb = alt(ka), alt(kb)
        emit("multi", pick(rng, m["q"], m["q_h"], spec["p_held"]), f(a=va, b=vb),
             [f(a=va, b=bb), f(a=aa, b=vb), f(a=aa, b=bb)], [ka, kb], evidence([ka, kb]))

    d, s_, r, sib = latest["daughter_age"], latest["son_age"], latest["rent"], latest["siblings"]
    calc_ans = {
        "kids_sum": (d + s_, [d + s_ + 1, d + s_ - 1, d + s_ + 2, abs(d - s_)]),
        "kids_diff": (abs(d - s_), [d + s_, abs(d - s_) + 1, abs(d - s_) + 2, abs(d - s_) - 1]),
        "annual_rent": (r * 12, ([first["rent"] * 12] if first["rent"] != r else []) + [r * 10, r * 12 + 12000, r * 11]),
        "daughter_18": (18 - d, [18 - d + 1, 18 - d - 1, d, 18 - d + 2]),
        "son_next": (s_ + 1, [s_, s_ + 2, s_ - 1, d + 1]),
        "family_kids": (sib + 1, [sib, sib + 2, sib + 3]),
    }
    for ck in rng.sample(list(spec["calc"]), n["tool"]):
        c = spec["calc"][ck]
        ans, cands = calc_ans[ck]
        ds = []
        for x in cands:
            if x > 0 and x != ans and x not in ds:
                ds.append(x)
        if ck == "annual_rent" and first["rent"] != r:
            ds = ds[:1] + rng.sample(ds[1:], 1)   # 舊房租 ×12 一定放進選項
        else:
            ds = rng.sample(ds, 2)
        u = c["unit"]
        emit("tool", pick(rng, c["q"], c["q_h"], spec["p_held"]), f"{ans}{u}",
             [f"{x}{u}" for x in ds] + [P.ABSTAIN], list(c["needs"]), evidence(c["needs"]))

    profile = {A[k]["label"]: canon(A[k], latest[k]) for k in sorted(own)}
    return dict(persona_id=pid, profile=profile, sessions=sessions, questions=questions)


# ---------------------------------------------------------------- tool-call items
def make_tool_item(tid, rng, spec):
    A = spec["attrs"]
    kinds = ["save", "update", "search", "search_calc", "calculator", "date", "answer_noise", "answer_other"]
    weights = [28, 10, 22, 6, 12, 6, 10, 6]
    kind = rng.choices(kinds, weights)[0]
    ctx = [noise_line(rng, spec) for _ in range(rng.randint(0, 2))]
    k = rng.choice(list(A))
    a = A[k]

    if kind == "save":
        v = sample_value(rng, spec, a)
        ctx.append(stmt_text(rng, spec, a, v))
        exp = dict(action="call", tool="memory_save", args=dict(key=a["label"], value=canon(a, v)),
                   accept=dict(key=a["aliases"], value=[canon(a, v)]))
    elif kind == "update":
        k = rng.choice([x for x in A if A[x]["updatable"]])
        a = A[k]
        v0 = sample_value(rng, spec, a)
        v1 = sample_value(rng, spec, a, exclude=(v0,))
        ctx = [stmt_text(rng, spec, a, v0)] + ctx + [stmt_text(rng, spec, a, v1, "upd")]
        exp = dict(action="call", tool="memory_save", args=dict(key=a["label"], value=canon(a, v1)),
                   accept=dict(key=a["aliases"], value=[canon(a, v1)]))
    elif kind == "search":
        ctx.append(pick(rng, a["q"], a["q_h"], spec["p_held"]))
        exp = dict(action="call", tool="memory_search", args=dict(query=a["label"]),
                   accept=dict(query=a["aliases"]))
    elif kind == "search_calc":
        c = rng.choice(list(spec["calc"].values()))
        ctx.append(pick(rng, c["q"], c["q_h"], spec["p_held"]))
        al = sorted({x for n_ in c["needs"] for x in A[n_]["aliases"]})
        exp = dict(action="call", tool="memory_search", args=dict(query=A[c["needs"][0]]["label"]),
                   accept=dict(query=al))
    elif kind == "calculator":
        t = pick(rng, spec["calc_direct"], spec["calc_direct_h"], spec["p_held"])
        av = rng.randint(*t["a"])
        bv = rng.randint(*t["b"]) if "b" in t else None
        ctx.append(t["t"].format(a=av, b=bv))
        expr = t["expr"].format(a=av, b=bv)
        exp = dict(action="call", tool="calculator", args=dict(expression=expr),
                   accept=dict(value=eval(expr)))
    elif kind == "date":
        ctx.append(pick(rng, spec["date_q"], spec["date_q_h"], spec["p_held"]))
        exp = dict(action="call", tool="get_date", args={}, accept={})
    elif kind == "answer_noise":
        ctx.append(noise_line(rng, spec))
        exp = dict(action="answer")
    else:  # answer_other：別人的資訊不存
        ctx.append(other_text(rng, a, sample_value(rng, spec, a)))
        exp = dict(action="answer")
    return dict(id=tid, kind=kind, context=ctx, expected=exp)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_personas", type=int, default=300)
    ap.add_argument("--n_tool", type=int, default=3000)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--turns", type=int, default=16, help="每段對話大約幾句")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    spec = load_spec(args.split)
    os.makedirs(args.out, exist_ok=True)
    prefix = args.split[:2]

    with open(os.path.join(args.out, "memory.jsonl"), "w", encoding="utf-8") as f:
        for i in range(args.n_personas):
            p = make_persona(f"{prefix}p{i + 1:04d}", rng, spec, args.sessions, args.turns)
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "tool.jsonl"), "w", encoding="utf-8") as f:
        for i in range(args.n_tool):
            t = make_tool_item(f"{prefix}t{i + 1:05d}", rng, spec)
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[{args.split}] {args.n_personas} personas, {args.n_tool} tool items -> {args.out}")


if __name__ == "__main__":
    main()
