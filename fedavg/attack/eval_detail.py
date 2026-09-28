"""
attack/eval_detail.py  —  评估细节：逐客户端记录与汇总（S9 / D-072，纯 numpy，不 import TF）

为什么要有：主列 ASR 只报「良性端的均值」，两件事因此看不见（FINDINGS F-061）：
  · 分布：G6 里与攻击者同 edge 的良性端全军覆没，均值会把它和受害 edge 摊在一起；
    阶段三评价防御要看尾部（p90、> 0.5 的比例），不是均值。
  · 饱和：ASR 接近 1 时（F-045），条件之间的差异在 ASR 上测不出来；
    触发样本的 margin（log p_t − max_{k≠t} log p_k）不饱和，还能分辨。
另外三项同样从已有的前向里顺带得到：干净样本被判成 y_t 的比例（区分「触发器特异的后门」
与「整体偏向 y_t」）、按真实类的 ASR、非目标翻转率（被推到既非真实类也非 y_t 的比例）。

**硬约束**（F-045 的确定性是「以后可带仪表重跑补记」的前提）：
  · 只读评估时已经算出的概率 / argmax，**不做任何额外前向、不碰任何 RNG**；
  · 客户端 ASR 由整数 argmax 计数得到，与日志里的主列 ASR 逐位相同
    （不从 log 概率重算：float32 下两个几乎相等的最大值会被 log 合并）。

口径：
  · 客户端级的量（asr / acc / yt_clean）在客户端之间**不加权平均**，与主列一致；
  · 样本级的分布量（margin 分位数、按类 ASR、非目标翻转率）在良性端的合格样本上**池化**；
  · 空组 → None（陷阱 #13：无定义不是 0）。
守卫：tests/test_eval_detail.py（解析值）+ tests/test_eval_detail_tf.py（与主列同一次前向）。
"""

from __future__ import annotations

import numpy as np

# softmax 输出里下溢成 0 的概率在取 log 前截到 float32 最小正规数（log ≈ −87.3）。
# margin = log p_t − max_{k≠t} log p_k 与 logit 之差严格相等（softmax 的归一化项相消），
# 只有极端饱和（差 > 约 87）会被截断 —— 分位数关心的是 0 附近，不受影响。
TINY = float(np.finfo(np.float32).tiny)
QUANTILES = (10, 50, 90)


def log_probs(probs) -> np.ndarray:
    """softmax 概率 → 截断后的自然对数（float64）。"""
    p = np.asarray(probs, dtype=np.float64)
    return np.log(np.clip(p, TINY, None))


def margins(probs, target: int) -> np.ndarray:
    """每个样本 log p_target − max_{k≠target} log p_k；> 0 ⇔ 判为 target（平局除外）。"""
    lp = log_probs(probs)
    if lp.size == 0:
        return np.zeros(0, dtype=np.float64)
    others = np.delete(lp, int(target), axis=1)
    return lp[:, int(target)] - others.max(axis=1)


def _q(values, qs=QUANTILES):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return [None] * len(qs)
    return [float(x) for x in np.percentile(v, qs)]


def _mean(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def client_record(*, client_id, edge_id, malicious, target, n_classes,
                  trig_probs=None, trig_pred=None, y_probe=None,
                  clean_pred=None, y_clean=None, keep_probs=False,
                  clean_probs=None) -> dict:
    """
    一个客户端在一个评估点上的记录。

    trig_probs / trig_pred / y_probe：主列 ASR 那一次前向的概率、argmax 与真实标签
        （four_way：探针 = 分片前 N 张、全类别；filtered：只有非目标类）。
    clean_pred / y_clean：同一个模型在整个留出分片上的干净预测（算 acc 的那一次前向）。
    keep_probs=True 时附上 fp16 对数概率与 uint8 预测（logits 存盘用；平时不留，省内存）。
    """
    t = int(target)
    rec = {"client_id": int(client_id), "edge_id": int(edge_id), "malicious": bool(malicious),
           "asr": None, "n": 0, "acc": None, "yt_clean": None, "margin_med": None,
           "margins": np.zeros(0), "flip_n": 0,
           "cls_hits": np.zeros(n_classes, dtype=np.int64),
           "cls_n": np.zeros(n_classes, dtype=np.int64)}

    if y_probe is not None and trig_pred is not None:
        y = np.asarray(y_probe).reshape(-1).astype(np.int64)
        pred = np.asarray(trig_pred).reshape(-1).astype(np.int64)
        elig = y != t
        n = int(elig.sum())
        rec["n"] = n
        if n:
            hits = int(np.sum(pred[elig] == t))
            rec["asr"] = float(hits / n)                       # 与 asr_counts 的过滤口径同式
            rec["flip_n"] = int(np.sum((pred[elig] != t) & (pred[elig] != y[elig])))
            rec["cls_hits"] = np.bincount(y[elig][pred[elig] == t], minlength=n_classes)[:n_classes]
            rec["cls_n"] = np.bincount(y[elig], minlength=n_classes)[:n_classes]
            if trig_probs is not None:
                m = margins(np.asarray(trig_probs)[elig], t)
                rec["margins"] = m
                rec["margin_med"] = float(np.median(m))

    if y_clean is not None and clean_pred is not None:
        yc = np.asarray(y_clean).reshape(-1).astype(np.int64)
        pc = np.asarray(clean_pred).reshape(-1).astype(np.int64)
        if yc.size:
            rec["acc"] = float(np.sum(pc == yc) / yc.size)
            nt = yc != t
            if nt.any():
                rec["yt_clean"] = float(np.sum(pc[nt] == t) / int(nt.sum()))

    if keep_probs:
        rec["y_probe"] = None if y_probe is None else np.asarray(y_probe).reshape(-1).astype(np.uint8)
        rec["trig_pred"] = None if trig_pred is None else np.asarray(trig_pred).reshape(-1).astype(np.uint8)
        rec["trig_logp"] = None if trig_probs is None else log_probs(trig_probs).astype(np.float16)
        rec["y_clean"] = None if y_clean is None else np.asarray(y_clean).reshape(-1).astype(np.uint8)
        rec["clean_pred"] = None if clean_pred is None else np.asarray(clean_pred).reshape(-1).astype(np.uint8)
        rec["clean_logp"] = None if clean_probs is None else log_probs(clean_probs).astype(np.float16)
    return rec


def summarize(records, target: int, n_classes: int) -> dict:
    """一组客户端记录（pooled 或一个 edge）→ 汇总。全部是 Python float / None。"""
    t = int(target)
    ben = [r for r in records if not r["malicious"]]
    mal = [r for r in records if r["malicious"]]
    ben_asr = [r["asr"] for r in ben if r["asr"] is not None]
    q_asr = _q(ben_asr)
    pooled_m = (np.concatenate([r["margins"] for r in ben]) if ben
                else np.zeros(0))
    q_m = _q(pooled_m)
    n_elig = int(sum(r["n"] for r in ben))
    cls_hits = np.sum([r["cls_hits"] for r in ben], axis=0) if ben else np.zeros(n_classes)
    cls_n = np.sum([r["cls_n"] for r in ben], axis=0) if ben else np.zeros(n_classes)
    cls_asr = [None if (c == t or cls_n[c] == 0) else float(cls_hits[c] / cls_n[c])
               for c in range(n_classes)]
    return {
        "benign_asr_p10": q_asr[0], "benign_asr_p50": q_asr[1], "benign_asr_p90": q_asr[2],
        "benign_asr_gt50": (float(np.mean([a > 0.5 for a in ben_asr])) if ben_asr else None),
        "malicious_clean_acc": _mean([r["acc"] for r in mal]),
        "yt_clean_benign": _mean([r["yt_clean"] for r in ben]),
        "margin_p10": q_m[0], "margin_p50": q_m[1], "margin_p90": q_m[2],
        "flip_other": (float(sum(r["flip_n"] for r in ben) / n_elig) if n_elig else None),
        "cls_asr": cls_asr,
        "n_benign": len(ben), "n_malicious": len(mal),
    }


# 每 edge 一行只打这几个（metrics.json 的体积预算，S9 设计复核）
EDGE_FIELDS = ("margin_p50", "benign_asr_p90", "yt_clean_benign")
# 逐客户端行的列（"/" 连接，None → na）
CLIENT_COLUMNS = (("ids", "client_id"), ("mal", "malicious"), ("asr", "asr"),
                  ("acc", "acc"), ("yt", "yt_clean"), ("mmed", "margin_med"), ("n", "n"))


def client_columns(records) -> dict:
    """一个 edge 的记录 → {列名: 值列表}（malicious 记 0/1）。"""
    out = {}
    for col, key in CLIENT_COLUMNS:
        vals = [r[key] for r in records]
        if key == "malicious":
            vals = [int(v) for v in vals]
        out[col] = vals
    return out


def pack_logits(records) -> dict:
    """keep_probs=True 的记录 → 一个 npz 的数组字典（按记录顺序拼接；每个样本带 client_id）。

    触发探针与干净分片各自拼接，两组长度可以不同（探针 = 分片前 N 张）。
    """
    def cat(key, dtype):
        parts = [r[key] for r in records if r.get(key) is not None]
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)

    def ids(key):
        parts = [np.full(len(r[key]), r["client_id"], np.int32)
                 for r in records if r.get(key) is not None]
        return np.concatenate(parts) if parts else np.zeros(0, np.int32)

    return {
        "trig_client": ids("y_probe"), "trig_y": cat("y_probe", np.uint8),
        "trig_pred": cat("trig_pred", np.uint8), "trig_logp": cat("trig_logp", np.float16),
        "clean_client": ids("y_clean"), "clean_y": cat("y_clean", np.uint8),
        "clean_pred": cat("clean_pred", np.uint8), "clean_logp": cat("clean_logp", np.float16),
        "client_ids": np.array([r["client_id"] for r in records], np.int32),
        "client_edge": np.array([r["edge_id"] for r in records], np.int16),
        "client_malicious": np.array([r["malicious"] for r in records], np.bool_),
    }
