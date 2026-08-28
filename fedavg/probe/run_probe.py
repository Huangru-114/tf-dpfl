"""
probe/run_probe.py  —  Exp 0.3 探针入口。

    $PY -m probe.run_probe --config <yaml> --checkpoint <ckpt.npz> \
        --benign 3 --malicious 3 --out <dir> [--theta-from global|edge]

做什么
──────
用**同一份 config + seed** 重建整个 world（`main.build_world`），把 checkpoint 灌回去，
然后对选中的客户端逐个跑 paired counterfactual，产出：

    <out>/probe_rows.csv       layer 级长表（§28 Client-level schema）
    <out>/probe_summary.json   global 行 + top-k% 能量 + occupation 曲线 + **三条判据**

**绝不重建数据分区**：客户端对象、数据切分、edge 归属全部由 `build_world` 决定，
重实现一遍就等于两条不可比的数据管线，而差异不会报错。

θ_t 的语义
──────────
checkpoint 存的是「进入第 t 轮时」的状态。`--theta-from global`（缺省）表示
探针复现的是 **cloud round t 的第一个 edge round** —— 那一刻 edge 刚被 broadcast
成全局权重，client 收到的就是它。`edge_rounds > 1` 时后续 edge round 的起点各不相同，
本探针不覆盖（那属于 HFL 传播会话的 edge 层嵌套 counterfactual）。

`--determinism`（见 CLAUDE.md 陷阱 #11）
────────────────────────────────────────
paired counterfactual 要求两条 clean twin 逐比特一致，但 **GPU 上做不到**：
Bad-PFL 的 FGSM 要在 `training=False` 下对输入求梯度，模型又带 BN，而 TF 没有这个
形状的确定性 GPU kernel —— 请求确定性直接抛 UnimplementedError。所以：

    cpu（缺省）  进程内隐藏 GPU + enable_op_determinism()。已实测：不触发那个 guard，
                 且两条 clean twin maxdiff = 0.000e+00。探针是离线小负载，CPU 吃得下。
    gpu-measure  用 GPU、不开确定性。快，但 clean twin 残差非零 —— 该残差作为
                 **硬件地板**上报，判据 2 改用「信号 ≫ max(硬件地板, SGD 地板)」。

**没有 gpu-strict** —— 上面已说明它不可能成立。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from main import build_world, load_config, set_seed
from models.cnn import get_base_head_indices
from probe import checkpoint as ckpt_io
from probe.analyze import analyze_probe
from probe.layermap import weight_index_map, layer_groups
from probe.paired import assert_attack_is_hook_gated, probe_client
from probe.writer import hardware_floor, verdicts, write_rows, write_summary


def _setup_devices(mode: str):
    """
    必须在**建任何模型之前**调用（`main()` 里它排在 `build_world` 之前；
    import 顺序无关 —— `set_visible_devices` 看的是 GPU 有没有被**初始化**，
    而 import TF / keras 本身不跑任何 op）。

    隐藏 GPU 用 `tf.config.set_visible_devices` 而不是 `CUDA_VISIBLE_DEVICES=`：
    容器（apptainer）的环境变量传递有历史包袱（cluster_env.sh:57-60 为
    TFDPFL_KERAS_HOME 同时设了 APPTAINERENV_ 前缀作显式保证）。进程内设定不依赖
    任何传递机制。`enable_op_determinism()` 必须在隐藏 GPU **之后**调，
    否则又会装上 GPU 的 fused-BN guard。

    ⚠️ `enable_op_determinism()` 是**进程级且不可逆**（TF 2.15 没有 disable_*），
    并且会把所有**未播种**的随机 op 变成硬错误
    （RuntimeError: Random ops require a seed）。本函数之后紧接着就是
    `set_seed(...)`（→ tf.random.set_seed），全局种子一旦设上，后续 op 就不会再撞
    这个错。调用顺序不要动。
    """
    import tensorflow as tf

    if mode == "cpu":
        tf.config.set_visible_devices([], "GPU")
        tf.config.experimental.enable_op_determinism()
        print("[Probe] determinism=cpu —— 已隐藏 GPU 并开启 op determinism")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[Probe] determinism=gpu-measure —— 用 GPU ({len(gpus)} 个)、不开确定性；"
              f"clean twin 残差将作为硬件地板上报")
    return tf


def _pick(clients, malicious_ids, n_benign: int, n_malicious: int):
    """
    确定性地挑探测对象：各自按 client_id 升序取前 N 个。

    不用随机抽：探针每次重跑必须落在**同一批**客户端上，否则跨 seed / 跨 checkpoint
    的行对不起来，"同一个客户端在 T1 与 T3 的变化"这类比较根本做不了。
    """
    mal = sorted([c for c in clients if int(c.client_id) in malicious_ids],
                 key=lambda c: int(c.client_id))[:n_malicious]
    ben = sorted([c for c in clients if int(c.client_id) not in malicious_ids],
                 key=lambda c: int(c.client_id))[:n_benign]
    return ben, mal


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--checkpoint", required=True, help="save_checkpoint 写出的 .npz")
    ap.add_argument("--out", default="probe_out")
    ap.add_argument("--benign", type=int, default=3)
    ap.add_argument("--malicious", type=int, default=3)
    ap.add_argument("--theta-from", choices=("global", "edge"), default="global")
    ap.add_argument("--determinism", choices=("cpu", "gpu-measure"), default="cpu",
                    help="见模块 docstring 与 CLAUDE.md 陷阱 #11")
    ap.add_argument("--ratio-threshold", type=float, default=3.0)
    ap.add_argument("--floor-tol", type=float, default=1e-6,
                    help="cpu 模式下 hardware_floor 的容差；超过就判仪表坏了并早停")
    ap.add_argument("--tag", default="")
    # --config / --override / --seed 由 main.load_config 自己解析（parse_known_args）
    args, _ = ap.parse_known_args(argv)

    config = load_config()
    assert_attack_is_hook_gated(config)

    # 必须在 build_world 建模型之前定设备
    _setup_devices(args.determinism)

    set_seed(config.get("seed", 42))
    world = build_world(config)
    clients = world["clients"]
    malicious_ids = set(int(i) for i in world["malicious_ids"])
    if not malicious_ids:
        raise SystemExit("[Probe] config 里没有恶意客户端 —— paired counterfactual 无从谈起。")

    ck = ckpt_io.load_checkpoint(args.checkpoint)
    m = ck["manifest"]
    round_idx = int(m["round"])
    applied = ckpt_io.apply_to_clients(ck, clients)
    print(f"[Probe] checkpoint round={round_idx} | 灌回 {len(applied)} 个客户端状态")

    gen = next((getattr(c, "_atk_generator", None) for c in clients
                if getattr(c, "_atk_generator", None) is not None), None)
    if gen is not None:
        gen_opt = next((getattr(c, "_atk_gen_opt", None) for c in clients
                        if getattr(c, "_atk_generator", None) is not None), None)
        ok = ckpt_io.apply_to_generator(ck, gen, gen_opt)
        print(f"[Probe] 共享 generator 装载: {ok}")
    elif m.get("generator") is not None:
        print("[Probe] ⚠️ checkpoint 里有 generator，但当前 world 里没有 —— "
              "config 的 badpfl_shared_generator 与产出 checkpoint 时不一致？")

    tag = args.tag or f"r{round_idx:04d}"
    seed = int(config.get("seed", 42))
    benign, malicious = _pick(clients, malicious_ids, args.benign, args.malicious)
    print(f"[Probe] 探测对象: benign={[int(c.client_id) for c in benign]} "
          f"malicious={[int(c.client_id) for c in malicious]}")

    # 层分组只需算一次（所有客户端的模型同构）
    ref_model = clients[0].model
    imap = weight_index_map(ref_model)
    split = get_base_head_indices(ref_model, int(config["data"]["num_classes"]))
    base_idx, head_idx = split["base_weight_indices"], split["head_weight_indices"]
    tensor_groups = layer_groups(imap)
    ref_weights = ref_model.get_weights()
    print(f"[Probe] 层分组 {len(tensor_groups)} 层 | backbone {len(base_idx)} 张量 / "
          f"head {len(head_idx)} 张量 | BN state 张量 {len(imap['bn_state_indices'])}")

    def _scoped(summs):
        """判据只看 upload×backbone —— 那是真正参与聚合、决定后门如何传播的权重。"""
        return [s for s in summs
                if s["scope"] == "backbone" and s["weight_space"] == "upload"]

    rows, summaries, failures = [], [], []
    aborted = None
    # 良性端**先跑**：它们是仪表本身。仪表坏了就不必把剩下的客户端跑完再报一个
    # 谁也看不懂的 FAIL —— 那在锚点上是几十分钟的白烧。
    for i, c in enumerate(benign + malicious):
        if i == len(benign) and args.determinism == "cpu":
            hw = hardware_floor(_scoped(summaries))
            if hw["max"] is not None and hw["max"] >= args.floor_tol:
                aborted = (
                    f"仪表未通过：determinism=cpu 下良性端 ‖Δθ_BD‖ max={hw['max']:.3e} "
                    f"≥ tol={args.floor_tol:.0e}。良性端两条 twin 走的是逐字节相同的"
                    f"代码路径，非零只能是批序冻结或状态还原漏了东西 —— 先修仪表，"
                    f"恶意端跑出来也不可读。")
                print(f"\n[Probe] ✗ {aborted}\n[Probe] 提前中止，跳过恶意端。")
                break
        cid = int(c.client_id)
        theta_t = (ckpt_io.global_weights(ck) if args.theta_from == "global"
                   else ckpt_io.edge_weights(ck, int(getattr(c, "assigned_edge", 0))))
        # 每个客户端一对独立、可复现的 order seed（与训练用的 rng 流互不干扰）
        sa = int(np.random.default_rng([seed, cid, 1]).integers(1, 2 ** 31))
        sb = int(np.random.default_rng([seed, cid, 2]).integers(1, 2 ** 31))
        role = "malicious" if cid in malicious_ids else "benign"
        print(f"[Probe] client {cid} ({role}) | seed_a={sa} seed_b={sb}")
        try:
            res = probe_client(c, theta_t, round_idx, seed_a=sa, seed_b=sb,
                               global_weights=ckpt_io.global_weights(ck))
        except Exception as e:
            # 客户端异常在训练里会被 _collect_updates_parallel 吞掉；探针里必须**显式**回传
            print(f"[Probe] ⚠️ client {cid} 失败: {type(e).__name__}: {e}")
            failures.append({"client_id": cid, "role": role,
                             "error": f"{type(e).__name__}: {e}"})
            continue
        out = analyze_probe(res, ref_weights, tensor_groups, base_idx, head_idx,
                            checkpoint=tag)
        rows.extend(out["rows"])
        summaries.extend(out["summaries"])

    outdir = Path(args.out)
    write_rows(outdir / f"probe_rows_{tag}.csv", rows)

    v = verdicts(_scoped(summaries), ratio_threshold=args.ratio_threshold,
                 determinism=args.determinism, tol=args.floor_tol)
    payload = {
        "run": {
            "checkpoint": str(args.checkpoint), "round": round_idx, "tag": tag,
            "theta_from": args.theta_from, "seed": seed,
            "determinism": args.determinism,
            "method": config["training"].get("drift_correction"),
            "strategy": config.get("backdoor", {}).get("malicious_strategy"),
            "arch": config.get("model", {}).get("arch"),
            "n_benign": len(benign), "n_malicious": len(malicious),
            "benign_ids": [int(c.client_id) for c in benign],
            "malicious_ids": [int(c.client_id) for c in malicious],
            "ckpt_run_info": m.get("run", {}),
        },
        "verdicts": v,
        "aborted": aborted,
        "client_failures": failures,
        "summaries": summaries,
    }
    write_summary(outdir / f"probe_summary_{tag}.json", payload)

    print("\n" + "=" * 60)
    print(f"[Probe] determinism = {args.determinism}")
    print(f"[Probe] 判据 1 仪表  | hardware_floor(良性端 ‖Δθ_BD‖) max = "
          f"{v['instrument_ok']['hardware_floor']['max']} | pass={v['instrument_ok']['pass']}"
          f"{'' if v['instrument_ok']['asserted'] else ' (gpu-measure：不断言，只作地板)'}")
    print(f"[Probe] 判据 2 信号  | 恶意端 ‖Δθ_BD‖ / max(硬件地板, SGD地板) min = "
          f"{v['signal']['signal_over_floor']['min']} "
          f"(阈值 {args.ratio_threshold}) | pass={v['signal']['pass']}")
    print(f"[Probe] SGD 噪声地板 | {v['noise_floor']}")
    if failures:
        print(f"[Probe] ⚠️ 失败客户端 {len(failures)} 个: {failures}")
    print(f"[Probe] 产物 -> {outdir}")
    print("=" * 60)
    if aborted:
        print(f"[Probe] ✗ {aborted}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
