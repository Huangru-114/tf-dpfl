"""
probe/checkpoint.py  —  把一次 run 在某一轮的完整状态落盘 / 装载。

**为什么必须新建**：全仓库唯一的 `save_weights` 在 `wideres_sweep.py:189`，与 FL 主干
无关。没有 θ_t 就没有 paired trajectory，Exp 0.3 的核心量根本定义不出来。

存什么
──────
  global 权重 / 每个 edge 的权重 / 每个 client 的私有状态（FedRep 的私有 head 等）
  / 每个 client 的 RNG 状态与优化器状态 / **Bad-PFL 共享 generator 及其 Adam slot**
  / round_idx / malicious_ids / config 摘要。

**共享 generator 必须存**：它是跨轮持续累积的攻击者状态（main.py:468 起全体恶意端
共享同一对象）。不存它，从 checkpoint 复现出来的 poison twin 会拿到一个**随机初始化**
的生成器 —— 那不是 T3 时刻的攻击，是第 0 轮的攻击，ASR 会莫名其妙地低。

格式
────
  `<name>.npz`（所有张量）+ `<name>.manifest.json`（结构 + 标量 + RNG 状态）。
  按 CLAUDE.md 回程红线：**`.npz` 留集群，只有 manifest 进 git**。
"""
import json
from pathlib import Path

import numpy as np

_META_KEY = "__manifest__"


# ════════════════════════════════════════════════════════════════════════════
# 客户端私有状态的序列化
# ════════════════════════════════════════════════════════════════════════════

def _opt_vars(opt):
    v = opt.variables
    return list(v() if callable(v) else v)


def _is_optimizer(o):
    return hasattr(o, "apply_gradients") and hasattr(o, "variables")


def client_state(client) -> dict:
    """
    一个客户端需要被复现的状态。

    `probe_state` 走 `FLClientBase.get_probe_state()` 协议（默认 `{}`）——
    PFL 方法各自的个性化状态语义不同（FedRep 的私有 head、pFedMe 的 Θ 锚点、
    Ditto 的 v_k），由方法类自己声明，探针不猜。
    """
    st = {
        "client_id": int(client.client_id),
        "is_malicious": bool(getattr(client, "is_malicious", False)),
        "model": [np.asarray(w) for w in client.model.get_weights()],
        "probe_state": dict(client.get_probe_state()),
        "rng": client.rng.bit_generator.state,
        "optimizers": {},
    }
    for name, obj in client.__dict__.items():
        if _is_optimizer(obj):
            st["optimizers"][name] = {v.name: np.asarray(v.numpy()) for v in _opt_vars(obj)}
    return st


def restore_client_state(client, st: dict):
    """把 client_state 产出的状态灌回客户端。"""
    import tensorflow as tf

    client.is_malicious = bool(st["is_malicious"])
    client.model.set_weights([np.asarray(w) for w in st["model"]])
    client.set_probe_state(st.get("probe_state", {}))
    client.rng.bit_generator.state = st["rng"]
    for name, snap in st.get("optimizers", {}).items():
        opt = getattr(client, name, None)
        if opt is None:
            continue
        for v in _opt_vars(opt):
            want = snap.get(v.name)
            v.assign(tf.zeros_like(v) if want is None
                     else tf.cast(tf.convert_to_tensor(np.asarray(want)), v.dtype))


# ════════════════════════════════════════════════════════════════════════════
# 落盘 / 装载
# ════════════════════════════════════════════════════════════════════════════

def _put(arrays, meta_list, key, tensors):
    """把一串张量塞进 npz 命名空间，并在 manifest 里记下个数。"""
    for i, t in enumerate(tensors):
        arrays[f"{key}/{i}"] = np.asarray(t)
    meta_list.append({"key": key, "n": len(tensors)})


def save_checkpoint(path, *, round_idx: int, global_weights, edge_weights: dict,
                    clients, shared_generator=None, shared_gen_opt=None,
                    malicious_ids=None, run_info: dict = None) -> Path:
    """
    Args:
        path:          输出前缀（`.npz` 与 `.manifest.json` 都由它派生）
        edge_weights:  {edge_id: weights_list}
        clients:       客户端对象列表（会各自调 client_state）
        shared_generator / shared_gen_opt: Bad-PFL 的共享生成器与其 Adam
    Returns:
        写出的 `.npz` 路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, tensor_meta = {}, []

    _put(arrays, tensor_meta, "global", global_weights)
    for eid, w in edge_weights.items():
        _put(arrays, tensor_meta, f"edge/{int(eid)}", w)

    client_meta = []
    for c in clients:
        st = client_state(c)
        cid = st["client_id"]
        _put(arrays, tensor_meta, f"client/{cid}/model", st["model"])
        ps_keys = []
        for k, v in st["probe_state"].items():
            _put(arrays, tensor_meta, f"client/{cid}/probe/{k}", v)
            ps_keys.append(k)
        opt_meta = {}
        for oname, snap in st["optimizers"].items():
            names = list(snap)
            for j, vname in enumerate(names):
                arrays[f"client/{cid}/opt/{oname}/{j}"] = snap[vname]
            opt_meta[oname] = names
        client_meta.append({"client_id": cid, "is_malicious": st["is_malicious"],
                            "probe_keys": ps_keys, "optimizers": opt_meta,
                            "rng": st["rng"]})

    gen_meta = None
    if shared_generator is not None:
        _put(arrays, tensor_meta, "generator", shared_generator.get_weights())
        gen_opt_names = []
        if shared_gen_opt is not None:
            vs = _opt_vars(shared_gen_opt)
            for j, v in enumerate(vs):
                arrays[f"generator/opt/{j}"] = np.asarray(v.numpy())
            gen_opt_names = [v.name for v in vs]
        gen_meta = {"opt_names": gen_opt_names}

    manifest = {
        "round": int(round_idx),
        "tensors": tensor_meta,
        "clients": client_meta,
        "edges": sorted(int(e) for e in edge_weights),
        "generator": gen_meta,
        "malicious_ids": sorted(int(i) for i in (malicious_ids or [])),
        "run": dict(run_info or {}),
    }
    arrays[_META_KEY] = np.frombuffer(
        json.dumps(manifest, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)

    npz = path.with_suffix(".npz")
    np.savez_compressed(npz, **arrays)
    # manifest 单独再写一份纯文本：它是唯一允许进 git 的部分（CLAUDE.md 回程红线）
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz


def load_checkpoint(path) -> dict:
    """读回 `save_checkpoint` 写出的内容。Returns: {"manifest":..., "arrays": NpzFile}"""
    path = Path(path)
    npz = path if path.suffix == ".npz" else path.with_suffix(".npz")
    z = np.load(npz, allow_pickle=False)
    manifest = json.loads(bytes(z[_META_KEY]).decode("utf-8"))
    return {"manifest": manifest, "arrays": z, "path": npz}


def _get(z, key, n):
    return [z[f"{key}/{i}"] for i in range(n)]


def _n_of(manifest, key):
    for m in manifest["tensors"]:
        if m["key"] == key:
            return m["n"]
    raise KeyError(f"checkpoint 里没有 {key!r}")


def global_weights(ckpt) -> list:
    m, z = ckpt["manifest"], ckpt["arrays"]
    return _get(z, "global", _n_of(m, "global"))


def edge_weights(ckpt, edge_id: int) -> list:
    m, z = ckpt["manifest"], ckpt["arrays"]
    key = f"edge/{int(edge_id)}"
    return _get(z, key, _n_of(m, key))


def apply_to_clients(ckpt, clients):
    """把 checkpoint 里的客户端状态灌进给定的客户端对象（按 client_id 匹配）。"""
    m, z = ckpt["manifest"], ckpt["arrays"]
    by_id = {int(c.client_id): c for c in clients}
    applied = []
    for cm in m["clients"]:
        cid = int(cm["client_id"])
        c = by_id.get(cid)
        if c is None:
            continue
        st = {
            "client_id": cid,
            "is_malicious": cm["is_malicious"],
            "model": _get(z, f"client/{cid}/model", _n_of(m, f"client/{cid}/model")),
            "probe_state": {
                k: _get(z, f"client/{cid}/probe/{k}", _n_of(m, f"client/{cid}/probe/{k}"))
                for k in cm.get("probe_keys", [])
            },
            "rng": cm["rng"],
            "optimizers": {
                oname: {vname: z[f"client/{cid}/opt/{oname}/{j}"]
                        for j, vname in enumerate(names)}
                for oname, names in cm.get("optimizers", {}).items()
            },
        }
        restore_client_state(c, st)
        applied.append(cid)
    return applied


def apply_to_generator(ckpt, generator, gen_opt=None):
    """把 checkpoint 里的共享生成器权重与 Adam slot 灌回去。"""
    import tensorflow as tf

    m, z = ckpt["manifest"], ckpt["arrays"]
    if m.get("generator") is None or generator is None:
        return False
    generator.set_weights(_get(z, "generator", _n_of(m, "generator")))
    names = m["generator"].get("opt_names") or []
    if gen_opt is not None and names:
        want = {n: z[f"generator/opt/{j}"] for j, n in enumerate(names)}
        for v in _opt_vars(gen_opt):
            a = want.get(v.name)
            v.assign(tf.zeros_like(v) if a is None
                     else tf.cast(tf.convert_to_tensor(a), v.dtype))
    return True
