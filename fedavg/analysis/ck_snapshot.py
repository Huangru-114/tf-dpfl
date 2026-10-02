"""
analysis/ck_snapshot.py  —  G8 快照上的 c_k 离线预检（S6a；D-085 方案 §二；TF 部分）

问题（S6a-PLAN §二）：功能分数 c_k 在「整个 body」这一级有没有信号？
  · 第 30 轮（攻击中）：E0（攻击者 edge）的 body vs E1–E3 的 body 在 c_{y_t} 上差多少；
  · 第 30 轮 vs 第 70 轮（衰减后）；各 3 seed。
  分不开 → 单个更新更分不开 → S6b 不做在线 c_k，3-D 只用几何分数。

做法：
  1. 按 G8 的配置重建数据与划分（只建客户端，不训练）；快照里的 client_edge / malicious_ids 与重建结果核对；
  2. 每个 edge：载入快照里该 edge 的权重（只读，不能续训）；
     NCM head = 本 edge **良性端留出分片**的并集上取倒数第二层特征的类均值原型
     （G8 用旧 noniid 划分，没有 edge 干净集；留出分片不在任何训练集里，陷阱 #11）；
  3. c_k = 对前 N 张留出样本做定向 PGD（推向类 k，固定 ε、少步数、**无随机起点** → 不碰任何 RNG）失败的比例；
     k = 0…K−1，重点看 c_{y_t}。ε / 步数缺省取评估 ξ 的同一组（backdoor.badpfl_epsilon，像素空间 4/255）。

用法（cwd = fedavg，经 cluster_env.sh 的容器）：
    python3 -m analysis.ck_snapshot --config ../experiments/attack/hfl-mechanism/configs/G8__a__s42.yaml \\
        --metrics ../experiments/attack/hfl-mechanism/results/P2/G8/G8__a__s42.metrics.json \\
        --dumps-root ../../tfdpfl-dumps --out /tmp/ck_s42.json
回传只有 JSON（每快照 × edge 的 c_0…c_9 与样本数），不回传任何张量（CLAUDE.md 协议）。纯算术在 functional_score.py。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from analysis import functional_score as FS


def feature_model(model):
    """倒数第二层特征（函数式模型最后一层是分类 Dense）。"""
    return tf.keras.Model(model.inputs, model.layers[-1].input)


def extract(feat, x, batch=256):
    return np.concatenate([feat(x[i:i + batch], training=False).numpy()
                           for i in range(0, len(x), batch)]).astype(np.float64)


def ck_reached(feat, protos, x, eps_in, steps, lo, hi, n_classes, batch=128):
    """
    定向 PGD（推向每个类 k）之后是否被 NCM 判成 k：返回 [n, K] 布尔。
    loss = CE(NCM 打分, k)，沿 −sign(∇) 走 steps 步、步长 2.5·ε/steps，投影到 ε 球与合法像素范围。
    没有随机起点 → 确定性、不碰任何 RNG。没有原型的类打分 −1e9（不会被选中，也不会被推到）。
    """
    p = tf.constant(np.nan_to_num(protos, nan=0.0), tf.float32)
    miss = tf.constant(np.where(np.isnan(protos[:, 0]), -1e9, 0.0), tf.float32)
    eps_t = tf.constant(np.broadcast_to(eps_in, (3,)), tf.float32)
    lo_t, hi_t = tf.constant(lo, tf.float32), tf.constant(hi, tf.float32)
    alpha = 2.5 / float(steps) * eps_t

    def logits(f):
        d2 = tf.reduce_sum(tf.square(f[:, None, :] - p[None, :, :]), axis=2)
        return -d2 + miss

    reached = np.zeros((len(x), n_classes), dtype=bool)
    for k in range(n_classes):
        if np.isnan(protos[k, 0]):
            continue
        for s in range(0, len(x), batch):
            x0 = tf.constant(x[s:s + batch], tf.float32)
            xa = x0
            tgt = tf.fill([x0.shape[0]], k)
            for _ in range(int(steps)):
                with tf.GradientTape() as tape:
                    tape.watch(xa)
                    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tgt, logits=logits(feat(xa, training=False))))
                g = tape.gradient(loss, xa)
                xa = xa - alpha * tf.sign(g)
                xa = tf.clip_by_value(xa, x0 - eps_t, x0 + eps_t)
                xa = tf.clip_by_value(xa, lo_t, hi_t)
            pred = tf.argmax(logits(feat(xa, training=False)), axis=1).numpy()
            reached[s:s + batch, k] = pred == k
    return reached


def ck_reached_batched(feat, protos, x, eps_in, steps, lo, hi, n_classes, max_batch=1024):
    """
    `ck_reached` 的批量版（S6b 在线评分用）：把「推向每个类 k」摞成一个 K·n 张的批，每步只做一次前向 + 反向，
    而不是 K 次小批（GPU 上 eager 小批的启动开销占大头）。推理模式下样本互不影响，loss 按样本求和 →
    每个 (类, 样本) 的梯度与单独算相同；ε = 0 时与 `ck_reached` 逐位相同（就是 NCM 预测）。
    K·n 超过 max_batch 时按类分块。没有原型的类整列 False。无随机起点，不碰任何 RNG。
    """
    p = tf.constant(np.nan_to_num(protos, nan=0.0), tf.float32)
    miss = tf.constant(np.where(np.isnan(protos[:, 0]), -1e9, 0.0), tf.float32)
    eps_t = tf.constant(np.broadcast_to(eps_in, (3,)), tf.float32)
    lo_t, hi_t = tf.constant(lo, tf.float32), tf.constant(hi, tf.float32)
    alpha = 2.5 / float(steps) * eps_t

    def logits(f):
        d2 = tf.reduce_sum(tf.square(f[:, None, :] - p[None, :, :]), axis=2)
        return -d2 + miss

    n = len(x)
    reached = np.zeros((n, n_classes), dtype=bool)
    valid = [k for k in range(n_classes) if not np.isnan(protos[k, 0])]
    per = max(1, int(max_batch) // max(n, 1))
    for c0 in range(0, len(valid), per):
        ks = valid[c0:c0 + per]
        x0 = tf.constant(np.tile(np.asarray(x, np.float32), (len(ks), 1, 1, 1)))
        tgt = tf.constant(np.repeat(np.asarray(ks, np.int32), n))
        xa = x0
        for _ in range(int(steps)):
            with tf.GradientTape() as tape:
                tape.watch(xa)
                loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tgt, logits=logits(feat(xa, training=False))))
            g = tape.gradient(loss, xa)
            xa = xa - alpha * tf.sign(g)
            xa = tf.clip_by_value(xa, x0 - eps_t, x0 + eps_t)
            xa = tf.clip_by_value(xa, lo_t, hi_t)
        pred = tf.argmax(logits(feat(xa, training=False)), axis=1).numpy().reshape(len(ks), n)
        for r, k in enumerate(ks):
            reached[:, k] = pred[r] == k
    return reached


def edge_ck(model, weights, x_proto, y_proto, x_att, y_att, eps_in, steps, lo, hi, n_classes):
    """一个 edge 的 body：装权重 → NCM 原型 → c_k。返回可 JSON 化的 dict。"""
    model.set_weights(weights)
    feat = feature_model(model)
    protos, counts = FS.class_prototypes(extract(feat, x_proto), y_proto, n_classes)
    pred0 = FS.ncm_predict(extract(feat, x_att), protos)
    reached = ck_reached(feat, protos, x_att, eps_in, steps, lo, hi, n_classes)
    c, n_el = FS.ck_from_reached(y_att, reached)
    return {"n_proto": int(len(y_proto)), "n_attack": int(len(y_att)),
            "proto_counts": counts.tolist(),
            "ncm_acc": float(np.mean(pred0 == np.asarray(y_att))) if len(y_att) else None,
            "c": c, "n_eligible": n_el}


def _numpy_of(ds, max_n=None):
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(np.asarray(y.numpy()).reshape(-1))
        if max_n and sum(len(a) for a in ys) >= max_n:
            break
    if not xs:
        return np.zeros((0,)), np.zeros(0, np.int64)
    x, y = np.concatenate(xs), np.concatenate(ys)
    return (x[:max_n], y[:max_n]) if max_n else (x, y)


def load_snapshot(path, n_weights, edge_ids):
    z = np.load(path)
    meta = json.loads(str(z["meta_json"]))
    edges = {int(e): [z[f"edge{int(e)}_{i:03d}"] for i in range(n_weights)] for e in edge_ids}
    return meta, edges


def run(config_path, snapshots, n_attack=500, steps=10, eps_pixel=None):
    """snapshots: {轮号: 快照路径}。返回 {"config":…, "snapshots": {轮号: {edge: {…}, "contrast": {…}}}}。"""
    import main as M
    from data.pixel_space import to_input_space, valid_range
    config = M.load_config(config_path)
    M.set_seed(config.get("seed", 42))
    x_train, y_train, x_test, y_test = _load_raw(config, M)
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    from models.cnn import build_model
    model = build_model(input_shape=(config["data"]["img_size"],) * 2 + (3,),
                        num_classes=config["data"]["num_classes"], arch=config["model"]["arch"],
                        rep_dim=int(config["model"].get("rep_dim", 64)))
    clients, baked, _ = M.build_clients(x_all, y_all, model, config, s3_out={})
    edge_servers, _ = M.build_edge_servers(clients, model, config, precomputed_assignments=baked)
    K = int(config["data"]["num_classes"])
    target = int(config["backdoor"].get("target_label", 0))
    eps_px = float(eps_pixel if eps_pixel is not None
                   else config["backdoor"].get("badpfl_epsilon", 4.0 / 255.0))
    eps_in = to_input_space(eps_px, config)
    lo, hi = valid_range(config)
    n_w = len(model.get_weights())
    out = {"config": str(config_path), "seed": int(config.get("seed", 42)), "target": target,
           "eps_pixel": eps_px, "steps": int(steps), "n_attack": int(n_attack), "snapshots": {}}
    for rnd, path in sorted(snapshots.items()):
        meta, edge_w = load_snapshot(path, n_w, [e.edge_id for e in edge_servers])
        mal = set(int(i) for i in meta["malicious_ids"])
        # 核对：重建的划分与快照一致，否则「本 edge 良性端」指的不是同一批人
        got = {int(c.client_id): int(c.assigned_edge) for c in clients}
        want = dict(zip(meta["client_ids"], meta["client_edge"]))
        if got != {int(k): int(v) for k, v in want.items()}:
            raise RuntimeError("重建的 client→edge 与快照里的 client_edge 不一致：不能继续（划分没复原）")
        res = {}
        for e in edge_servers:
            ben = sorted((c for c in e.clients if int(c.client_id) not in mal),
                         key=lambda c: int(c.client_id))
            x_all_e, y_all_e = [], []
            for c in ben:
                x, y = _numpy_of(c.test_dataset)
                x_all_e.append(x)
                y_all_e.append(y)
            xe, ye = np.concatenate(x_all_e), np.concatenate(y_all_e)
            res[str(int(e.edge_id))] = edge_ck(model, edge_w[int(e.edge_id)], xe, ye,
                                               xe[:n_attack], ye[:n_attack], eps_in, steps, lo, hi, K)
        att_edge = None
        if meta.get("eval_attacker") is not None:
            att_edge = int(got[int(meta["eval_attacker"])])
        c_by_edge = {int(e): v["c"] for e, v in res.items()}
        res["contrast"] = ({"attacker_edge": att_edge, "k": target,
                            **FS.edge_contrast(c_by_edge, att_edge, target)}
                           if att_edge is not None else None)
        out["snapshots"][str(int(rnd))] = res
    return out


def _load_raw(config, M):
    name = config["data"].get("dataset", "cifar10").lower()
    if name != "cifar10":
        raise ValueError("c_k 预检只做了 cifar10")
    _, _, x_train, y_train, x_test, y_test = M.load_cifar10(config)
    return x_train, y_train, x_test, y_test


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True, help="该 run 的 metrics.json（读 dumps.snapshots 的 manifest）")
    ap.add_argument("--dumps-root", required=True, help="tfdpfl-dumps 目录（manifest 里的路径相对它）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-attack", type=int, default=500)
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args(argv)
    m = json.loads(Path(a.metrics).read_text(encoding="utf-8"))
    snaps = {int(s["round"]): Path(a.dumps_root) / s["path"] for s in m["dumps"]["snapshots"]}
    if not snaps:
        print("[ck] metrics.json 的 dumps.snapshots 为空 —— 这个 run 没开快照", file=sys.stderr)
        return 2
    missing = [str(p) for p in snaps.values() if not p.exists()]
    if missing:
        print(f"[ck] 快照文件不存在：{missing}（存盘在集群上、不进 git）", file=sys.stderr)
        return 2
    res = run(a.config, snaps, a.n_attack, a.steps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[ck] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
