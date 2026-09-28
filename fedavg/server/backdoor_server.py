"""
server/backdoor_server.py  –  后门感知的 Cloud 服务器

继承 CloudServer，不改其核心训练/聚合逻辑；只在每 bd_eval_interval 轮（及最后一轮）
额外计算并记录**分层后门指标**（任务2/3）：
  - global / edge / local 三层 ASR 与 ACC
  - local 模型 same_edge vs diff_edge ASR（判断后门是否溢出到其他边缘节点）
  - 特征空间分离度（global + 抽样 local）
  - 后门遗忘曲线（仅个性化/微调方法、仅评估轮、抽样 1 个良性客户端）
所有指标通过 FLLogger.log_round_metrics 记入 wandb。
"""

import json
import time

import numpy as np

from server.server import CloudServer
from server.stopping import ASR_KEYS
from attack.backdoor_eval import (evaluate_hierarchical_asr,
                                   evaluate_forgetting_curve,
                                   evaluate_feature_separation,
                                   evaluate_drift,
                                   evaluate_local_asr,
                                   dataset_to_numpy)
from attack.eval_detail import EDGE_FIELDS, client_columns, pack_logits, summarize
from alignment import get_switch
from utils.dumps import dump_line, dump_root, parse_rounds, run_dir_name, write_npz
from utils.kvline import fmt_list, format_kv
from models.cnn import get_base_head_indices
from defense import create_post_hoc_defense
from utils.logger import FLLogger


# 个性化 / 微调方法：遗忘曲线对这些方法有意义（client.model 训练后停在个性化模型）
_PERSONALIZED_METHODS = {
    "hierpfedme", "pfedme", "hier_ditto", "hier_ditto_rep", "hier_pfedme_rep",
    "hier_fedrep",
}


class BackdoorCloudServer(CloudServer):

    def __init__(self, *args, bd_cfg=None, x_test=None, y_test=None,
                 trigger_fn=None, malicious_ids=None, eval_attacker=None, **kwargs):
        super().__init__(*args, **kwargs)
        # ── 评估口径的对齐开关（A02 / A06；A28 的 pm_kind 在 CloudServer 里）────
        #   eval_xi_model=fixed_attacker：主 ASR 的 ξ 在 setup 时按 seed 固定选中的一个
        #   攻击者（eval_attacker，main.select_eval_attacker）的 PM 上求（D-015 + D-033），
        #   δ 用它的生成器（共享时就是那一个）；白盒列（ξ 在受害者上求）另打 [ASRwb]。
        self.eval_attacker = eval_attacker
        self.eval_xi_model = get_switch(self.config, "backdoor.eval_xi_model")
        self.asr_columns = get_switch(self.config, "evaluation.asr_columns")
        # 评估降频（D-050）：白盒列可关（白盒 ≈ 主列，F-051）；陈旧 ASR 每 stale_asr_every 个
        # 评估点算一次，与陈旧 pm_acc 共用 CloudServer._eval_seq（同一批点）。
        self.whitebox_asr = bool(get_switch(self.config, "evaluation.whitebox_asr"))
        self.stale_asr_every = int(get_switch(self.config, "evaluation.stale_asr_every"))
        self._seed = int(self.config.get("seed", 42))
        self.bd_cfg = bd_cfg or {}
        self.x_test = x_test
        self.y_test = np.asarray(y_test).reshape(-1) if y_test is not None else None
        self.trigger_fn = trigger_fn
        self.malicious_ids = set(int(i) for i in (malicious_ids or set()))
        self.bd_eval_interval = int(self.bd_cfg.get("eval_interval", 50))
        self.bd_asr_max = int(self.bd_cfg.get("asr_max_samples", 0))  # 0 = 用全部非目标类
        self.bd_target = int(self.bd_cfg.get("target_label", 9))
        self.bd_verbose = bool(self.bd_cfg.get("verbose_clients", True))
        self._all_clients = [c for e in self.edge_servers for c in e.clients]
        self.history.setdefault("bd_c_acc", [])
        self.history.setdefault("bd_asr", [])
        self._last_bd_metrics = None

        # ── S9：评估细节（常开）+ 两个存盘开关（D-072 / D-073；utils/dumps.py）────────
        #   细节只读主列那一次前向的结果，不多做前向、不碰 RNG → 已有数值逐位不变。
        #   dump_logits_every=k：每第 k 个后门评估点存一次逐样本对数概率（0 = 关）。
        #   snapshot_rounds="30/70"：这些 cloud 轮末存分析快照（只供评估，不能续训）。
        self._n_classes = int(self.config["data"]["num_classes"])
        self.dump_logits_every = int(get_switch(self.config, "evaluation.dump_logits_every"))
        self.snapshot_rounds = parse_rounds(
            get_switch(self.config, "evaluation.snapshot_rounds"))
        self._bd_eval_count = 0
        self._last_probe_order = None
        self._dump_root = dump_root()
        self._dump_dir = run_dir_name(self.config)

        # 特征分离度 / 遗忘曲线相关配置
        # 漂移测量（Experiment 3C）：默认关闭，零开销；3C 的 config 打开。
        self.drift_eval = bool(self.bd_cfg.get("drift_eval", False))
        self.drift_probe_samples = int(self.bd_cfg.get("drift_probe_samples", 256))
        self._drift_base_idx = None   # 懒缓存 backbone 索引

        self.feature_eval = bool(self.bd_cfg.get("feature_eval", True))
        self.feature_eval_samples = int(self.bd_cfg.get("feature_eval_samples", 200))
        self.feature_clean_per_class = int(self.bd_cfg.get("feature_clean_per_class", 50))
        self.forgetting_epochs = int(self.bd_cfg.get(
            "forgetting_epochs", self.config["training"].get("local_epochs", 5)))
        self._feat_data = None       # 懒惰构建并缓存

        # ── 后处理防御（Simple-Tuning）：对每个良性 client 的本地模型评估前做后处理。──
        # name 非后处理类时为 None；为 None 时分层评估直接用 client.model（行为不变）。
        self.post_defense = create_post_hoc_defense(
            self.config,
            num_classes=int(self.config["data"]["num_classes"]),
            lr_default=float(self.config["training"].get("learning_rate", 0.02)))

    def run_round(self, round_idx: int):
        n_rounds = int(self.config["federation"]["n_rounds"])
        do_eval  = (round_idx % self.bd_eval_interval == 0) or (round_idx == n_rounds)
        # 漂移的 anchor = 本轮起点（broadcast/训练之前）的全局权重。必须在 super() 前深拷贝，
        # 因为 super().run_round 会就地更新 self.global_model。
        anchor = None
        if self.drift_eval and do_eval:
            anchor = [w.copy() for w in self.global_model.get_weights()]

        metrics = super().run_round(round_idx)
        # CerP：cloud 层协调器，按上一轮快照给恶意客户端互相分发 peer 权重
        self._coordinate_cerp_peers()
        if do_eval:
            self._backdoor_eval(round_idx, anchor=anchor)
        # S9：分析快照放在本轮全部评估之后（edge 模型此刻 = 本轮 fresh-PM 用的 body）
        if round_idx in self.snapshot_rounds:
            self._snapshot(round_idx, evaluated=do_eval)
        return metrics

    def _stopping_signals(self, metrics: dict) -> dict:
        """
        基类的 `pm_acc` 之外，补上三层 ASR —— 判据 A（θ 全越过）读的就是它们。

        `_last_bd_metrics` 只在**后门评估轮**更新；非评估轮沿用上一次会让同一个值
        被重复喂进序列，把去抖计数和斜率都算错。所以喂完就清空：非评估轮这三项
        是 `None`，规则会跳过（而不是当 0）。
        """
        sig = super()._stopping_signals(metrics)
        bd = self._last_bd_metrics
        self._last_bd_metrics = None
        for k in ASR_KEYS:
            sig[k] = None if bd is None else bd.get(k)
        return sig

    # ══════════════════════════════════════════════════════════════════════
    # 评估触发器与副列（A02 / A06 / A28）
    # ══════════════════════════════════════════════════════════════════════

    def _eval_rng(self, round_idx, column):
        """评估期 PGD 起点噪声：按 (seed, 轮, 列) 键控 —— 不消耗训练流，列之间互不影响。"""
        return np.random.default_rng([self._seed, 0xE7A1, int(round_idx), int(column)])

    def _attacker_trigger(self, round_idx, column, pm_kind):
        """ξ 在固定攻击者的 PM（pm_kind）上求，δ 用它的生成器；**忽略**被评估的模型。"""
        att = self.eval_attacker
        xi_model = self.pm_model(att, pm_kind, slot="attacker")
        rng = self._eval_rng(round_idx, column)

        def trig(model, x, y=None):
            x = np.asarray(x, np.float32)
            return x + att.eval_xi(xi_model, x, y, rng=rng) + att.eval_delta(x)
        return trig

    def _side_columns(self, round_idx, metrics, fixed):
        """
        副列，各打一条独立的 key=value 行（不扩 [Backdoor] 的正则）：
          [ASR4]     A06：过滤 / 不过滤 × 仅良性 / 全体（主口径的模型与 ξ）
          [ASRwb]    A02：白盒 ξ（在受害者自己的 PM 上求）的良性过滤 ASR —— 上界
          [StaleASR] A28：陈旧 PM（client.model）上的良性 / 恶意过滤 ASR；
                     ξ 用攻击者**自己的陈旧模型**（受害者与攻击者同一定义，A4 用户拍板）

        返回两个副列各自的耗时 {"whitebox": s, "stale": s}（没算的记 None），
        由 _backdoor_eval 打进 [TimingASR]（D-047：副列降频之前先量它们占多少）。
        [ASR4] 只是打印主列已算好的数，不单独计时。

        降频（D-050）：evaluation.whitebox_asr=false → 不算白盒；evaluation.stale_asr_every=k
        → 陈旧 ASR 只在第 0、k、2k… 个评估点算（与陈旧 pm_acc 同一个序号）。没算的列不打行。
        """
        t_side = {"whitebox": None, "stale": None}
        if self.asr_columns == "four_way":
            print(format_kv("[ASR4]", {
                "benign_filtered":   metrics["local_asr_benign_mean"],
                "benign_unfiltered": metrics["local_asr_benign_unfiltered_mean"],
                "all_filtered":      metrics["local_asr_mean"],
                "all_unfiltered":    metrics["local_asr_unfiltered_mean"],
                "global_unfiltered": metrics["global_asr_unfiltered"],
            }, round_idx=round_idx))
        benign = [c for c in self._all_clients if int(c.client_id) not in self.malicious_ids]
        if fixed and getattr(self, "whitebox_asr", True):
            _t0 = time.perf_counter()
            att, rng = self.eval_attacker, self._eval_rng(round_idx, 1)
            wb = evaluate_local_asr(
                benign, self.main_pm,
                lambda model, x, y=None: att.eval_trigger(model, x, y, rng=rng),
                self.bd_target, self.malicious_ids, fallback_test_ds=self.test_dataset,
                asr_max_samples=self.bd_asr_max)
            metrics["local_asr_benign_whitebox"] = wb["benign_mean"]
            print(format_kv("[ASRwb]", {"local_benign": wb["benign_mean"]},
                            round_idx=round_idx))
            t_side["whitebox"] = time.perf_counter() - _t0
        if self.pm_kind == "fresh" and self._side_due(getattr(self, "stale_asr_every", 1)):
            _t0 = time.perf_counter()
            trig = (self._attacker_trigger(round_idx, 2, "stale") if fixed else self.trigger_fn)
            st = evaluate_local_asr(
                self._all_clients, lambda c: c.model, trig, self.bd_target,
                self.malicious_ids, fallback_test_ds=self.test_dataset,
                asr_max_samples=self.bd_asr_max)
            metrics["local_asr_benign_stale"] = st["benign_mean"]
            metrics["local_asr_malicious_stale"] = st["malicious_mean"]
            print(format_kv("[StaleASR]", {"local_benign": st["benign_mean"],
                                           "local_malicious": st["malicious_mean"]},
                            round_idx=round_idx))
            t_side["stale"] = time.perf_counter() - _t0
        return t_side

    # ══════════════════════════════════════════════════════════════════════
    # S9：评估细节行 + logits 存盘 + 分析快照（D-072 / D-073）
    # ══════════════════════════════════════════════════════════════════════

    def _logits_due(self) -> bool:
        """本次后门评估要不要存 logits：每第 dump_logits_every 个评估点（第 0、k、2k… 个）。"""
        k = self.dump_logits_every
        due = k > 0 and self._bd_eval_count % k == 0
        self._bd_eval_count += 1
        return due

    def _write_dump(self, round_idx, kind, rel, arrays):
        """写盘 + 打 [Dump] manifest 行。写盘失败不拖垮 run，但要在日志里留下 error。"""
        try:
            info = write_npz(self._dump_root, f"{self._dump_dir}/{rel}", arrays)
        except OSError as e:
            msg = f"{type(e).__name__}: {e}".replace("|", "/").replace("=", ":")
            print(format_kv("[Dump]", {"kind": kind, "path": rel, "error": msg[:200]},
                            round_idx=round_idx))
            return
        print(dump_line(round_idx, kind, info))

    def _emit_detail(self, round_idx, metrics, keep_probs):
        """
        [EvalDetail]（良性端池化）+ 每 edge 的 [EvalDetailEdge] / [ClientEval]，
        到期时再存 logits。原始数组用完即丢：metrics 之后会进 _last_bd_metrics。
        """
        recs = metrics.pop("client_detail", None) or []
        self._last_probe_order = metrics.pop("probe_order", None)
        if not recs:
            return
        pooled = summarize(recs, self.bd_target, self._n_classes)
        fields = {k: v for k, v in pooled.items() if k not in ("n_benign", "n_malicious")}
        fields["cls_asr"] = fmt_list(pooled["cls_asr"])
        print(format_kv("[EvalDetail]", fields, round_idx=round_idx))
        by_edge = {}
        for r in recs:
            by_edge.setdefault(r["edge_id"], []).append(r)
        for eid in sorted(by_edge):
            s = summarize(by_edge[eid], self.bd_target, self._n_classes)
            print(format_kv("[EvalDetailEdge]", {k: s[k] for k in EDGE_FIELDS},
                            round_idx=round_idx, edge_id=eid))
            cols = client_columns(by_edge[eid])
            print(format_kv("[ClientEval]", {k: fmt_list(v) for k, v in cols.items()},
                            round_idx=round_idx, edge_id=eid))
        if keep_probs:
            self._write_dump(round_idx, "logits", f"logits_r{round_idx:03d}.npz",
                             pack_logits(recs))

    def _snapshot(self, round_idx, evaluated):
        """
        分析快照（只供评估，不能续训）：全局模型、每 edge 的模型、每端私有部分
        （private_state：FedRep 的私有 head + 私有 BN 统计量；从未被选中的端为空 →
        其 fresh-PM 就是 edge 模型）、评估攻击者的生成器（与其 Adam 状态，若已建）。
        fp32 原样存：重载后应逐位复现本轮的 fresh-PM 评估。
        """
        arrays = {}

        def put(prefix, ws):
            for i, w in enumerate(ws):
                arrays[f"{prefix}_{i:03d}"] = np.asarray(w)

        put("global", self.global_model.get_weights())
        cache = getattr(self, "_edge_w_cache", None)
        matches = None
        for e in self.edge_servers:
            ws = e.model.get_weights()
            put(f"edge{int(e.edge_id)}", ws)
            if evaluated and cache is not None and int(e.edge_id) in cache:
                same = (len(ws) == len(cache[int(e.edge_id)]) and all(
                    np.array_equal(a, b) for a, b in zip(ws, cache[int(e.edge_id)])))
                matches = same if matches is None else (matches and same)
        has_private = []
        for c in self._all_clients:
            idx, val = c.private_state()
            cid = int(c.client_id)
            arrays[f"client{cid}_idx"] = np.asarray(idx, dtype=np.int32)
            put(f"client{cid}", val)
            has_private.append(bool(len(idx)))
        att = self.eval_attacker
        gen = getattr(att, "_atk_generator", None) if att is not None else None
        n_gen_opt = 0
        if gen is not None:                         # 不调 _atk_ensure_generator：只读已有的
            put("gen", gen.get_weights())
            opt = getattr(att, "_atk_gen_opt", None)
            if opt is not None:
                v = opt.variables
                v = v() if callable(v) else v
                opt_ws = [np.asarray(x.numpy()) for x in v]
                put("genopt", opt_ws)
                n_gen_opt = len(opt_ws)
        R = int(self.config["federation"].get("edge_rounds", 1) or 1)
        meta = {
            "round": int(round_idx), "effective_round": int(round_idx) * R,
            "seed": self._seed, "run_id": (self.config.get("meta") or {}).get("run_id"),
            "evaluated": bool(evaluated), "edge_matches_eval": matches,
            "edge_ids": [int(e.edge_id) for e in self.edge_servers],
            "client_ids": [int(c.client_id) for c in self._all_clients],
            "client_edge": [int(getattr(c, "assigned_edge", -1)) for c in self._all_clients],
            "has_private": has_private,
            "malicious_ids": sorted(self.malicious_ids),
            "eval_attacker": None if att is None else int(att.client_id),
            "has_generator": gen is not None, "n_genopt": n_gen_opt,
            "probe_order": self._last_probe_order if evaluated else None,
            "resumable": False,
        }
        arrays["meta_json"] = np.array(json.dumps(meta))
        self._write_dump(round_idx, "snapshot", f"snapshot_r{round_idx:03d}.npz", arrays)

    def _coordinate_cerp_peers(self):
        """收集 CerP 恶意客户端最新权重，互相分发（排除自身），供下一轮 cos 正则使用。"""
        cerp = [c for c in self._all_clients
                if getattr(c, "is_malicious", False)
                and hasattr(c, "set_peer_malicious_weights")]
        if not cerp:
            return
        snaps = [(c, c.model.get_weights()) for c in cerp]
        for c in cerp:
            peers = [w for oc, w in snaps if oc is not c]
            c.set_peer_malicious_weights(peers)

    # ── 特征评估用数据：固定一次，避免每轮重建 ──────────────────────────────
    def _prepare_feature_data(self):
        if self._feat_data is not None:
            return self._feat_data
        x, y = self.x_test, self.y_test
        # 干净样本按类抽样
        clean_by_class = {}
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) > self.feature_clean_per_class:
                idx = idx[:self.feature_clean_per_class]
            clean_by_class[int(c)] = x[idx]
        # 后门样本：非目标类抽样后加 trigger，保留真实标签
        non_target = np.where(y != self.bd_target)[0]
        if len(non_target) > self.feature_eval_samples:
            non_target = non_target[:self.feature_eval_samples]
        # 评估侧 trigger 约定 (model, x, y)；静态触发器忽略 model/y，这里用 global_model。
        # （Phase 2 的 model-dependent 触发器若需 per-model 特征样本，再单独处理。）
        poisoned = self.trigger_fn(self.global_model, x[non_target], y[non_target])
        poisoned_true = y[non_target]
        self._feat_data = (poisoned, poisoned_true, clean_by_class)
        return self._feat_data

    def _backdoor_eval(self, round_idx: int, anchor=None):
        # ASR 的探针是**留出分片**，不是原始 x_test。
        #
        # 本仓库按 PFLlib 口径把 train+test 合并后分区，所以官方 test split 里的图
        # 已经分给客户端、其中约 1−per_client_test_ratio 进了训练集。再拿同一份
        # x_test 当探针，等于在训练过的图上测攻击成功率。
        # 而分区使每个 index 只归属一个客户端、再在客户端内部切 train/test，
        # 于是「所有留出分片的并集」与「所有训练数据」**全局不相交** —— 那才是
        # 真正的留出集，而且 ASR 与 pm_acc 这下测在同一个 population 上。
        #
        #   global ASR ← self.test_dataset（= merge_test_datasets(所有 edge)）
        #   edge   ASR ← edge.get_test_dataset()
        #   client ASR ← client.test_dataset
        #
        # 子采样改为「取前 N 个合格样本」：测试集的 dataset 是 shuffle=False 建的，
        # 顺序固定，所以这是确定性的，不需要像旧路径那样缓存一份随机索引。
        # （x_test/y_test 只留给可选的 feature_eval，见 _prepare_feature_data；
        #  那条路径在 exp3 的全部配置里都是关的。）

        print(f"\n[Backdoor] ===== Round {round_idx} hierarchical backdoor eval =====")

        # ── 分阶段计时 ───────────────────────────────────────────────────
        # 为什么需要：`[Cloud] … time=Xs` 测的是 CloudServer.run_round 的
        # t0→elapsed，而 _backdoor_eval 是在 super().run_round() **返回之后**
        # 才调用的（见 run_round），所以那个数**不含**后门评估。
        # 于是「一轮到底花了多少、其中多少花在评估上」在 metrics.json 里
        # 此前完全看不到 —— 标定 (local_epochs, n_rounds, eval_interval)
        # 时这正是要读的那个数。
        # 未启用的阶段记 None → 打 "n/a"，不是 0.0（陷阱 #13 的同一约定：
        # 「没跑」与「跑了但是很快」不能在数值上长得一样）。
        _t_all = time.perf_counter()
        t_feature = t_forget = t_drift = None

        # ── Simple-Tuning 防御：良性 client 本地模型评估前做「重置头 + 干净微调」──
        # （恶意 client 不设防，保持 c.model 原样，仅用于报告 local_asr_malicious）
        # A28：本地模型一律经 main_pm 取（与 pm_acc 同一个入口 → 同一个 PM）；
        # stale 口径下 main_pm(c) 就是 c.model，与改动前逐字相同。
        if self.pm_kind == "fresh":
            self.begin_pm_eval()

        def local_model_fn(c):
            return self.main_pm(c)
        if self.post_defense is not None:
            def local_model_fn(c):
                if int(c.client_id) in self.malicious_ids:
                    return self.main_pm(c)
                return self.post_defense.tune(self.main_pm(c), getattr(c, "dataset", None))
            print(f"[Defense] applying Simple-Tuning to benign local models "
                  f"before ASR eval (round {round_idx})")

        fixed = (self.eval_xi_model == "fixed_attacker" and self.eval_attacker is not None)
        main_trigger = (self._attacker_trigger(round_idx, 0, self.pm_kind) if fixed
                        else self.trigger_fn)

        # ── 任务2：分层 ASR（global/edge/local 三层 + same/diff edge） ────────
        keep_probs = self._logits_due()
        _t0 = time.perf_counter()
        metrics = evaluate_hierarchical_asr(
            self.global_model, self.edge_servers, self._all_clients,
            self.test_dataset, main_trigger, self.bd_target,
            self.malicious_ids, fallback_test_ds=self.test_dataset,
            local_model_fn=local_model_fn,
            asr_max_samples=self.bd_asr_max,
            asr_columns=self.asr_columns,
            keep_probs=keep_probs,
            n_classes=self._n_classes,
        )
        t_asr_main = time.perf_counter() - _t0
        t_side = self._side_columns(round_idx, metrics, fixed)
        t_asr = time.perf_counter() - _t0
        # S9：细节行与 logits 存盘放在计时之后 —— [Timing] asr / [TimingASR] main 与改动前可比
        self._emit_detail(round_idx, metrics, keep_probs)

        # ── 任务2：特征空间分离度（global + 抽样 1 个良性 local） ─────────────
        if self.feature_eval:
            _t0 = time.perf_counter()
            poisoned, poisoned_true, clean_by_class = self._prepare_feature_data()
            sep_g = evaluate_feature_separation(
                self.global_model, poisoned, poisoned_true,
                clean_by_class, self.bd_target)
            metrics["sep_score_global"] = sep_g["separation_score"]

            benign = [c for c in self._all_clients
                      if int(c.client_id) not in self.malicious_ids]
            if benign:
                sep_l = evaluate_feature_separation(
                    benign[0].model, poisoned, poisoned_true,
                    clean_by_class, self.bd_target)
                metrics["sep_score_local"] = sep_l["separation_score"]
            t_feature = time.perf_counter() - _t0

        # ── 任务2：后门遗忘曲线（仅个性化/微调方法、抽样 1 个良性客户端） ─────
        method = self.config["training"].get("drift_correction", "fedavg")
        do_forget = (method in _PERSONALIZED_METHODS
                     or self.config.get("evaluation", {}).get("final_finetune", False))
        if do_forget:
            _t0 = time.perf_counter()
            benign = [c for c in self._all_clients
                      if int(c.client_id) not in self.malicious_ids]
            if benign:
                c0 = benign[0]
                lr = float(self.config["training"].get("learning_rate", 0.02))
                # 探针同样用**这个客户端自己**的留出分片。
                # evaluate_forgetting_curve 还吃 numpy（内部自己调 compute_asr、
                # 自己做目标类过滤），所以这里摊平一份。
                fx, fy = dataset_to_numpy(
                    c0.test_dataset if getattr(c0, "test_dataset", None) is not None
                    else self.test_dataset, self.bd_asr_max)
                if fx is not None:
                    metrics["forgetting_curve"] = evaluate_forgetting_curve(
                        c0.model, self.forgetting_epochs, c0.dataset,
                        fx, fy, self.trigger_fn, self.bd_target, lr=lr)
            t_forget = time.perf_counter() - _t0

        # ── 历史 + 控制台 ─────────────────────────────────────────────────
        self.history["bd_c_acc"].append(metrics["local_acc_mean"])
        self.history["bd_asr"].append(metrics["local_asr_benign_mean"])
        # 供自适应轮数的停止规则读（见 _stopping_signals）。只存最近一次评估的
        # 三层 ASR —— 规则自己维护序列，这里不做累积。
        self._last_bd_metrics = metrics
        # 无定义的分组打 "n/a"（不是 0.000）——见 backdoor_eval._mean 的说明。
        # 回程解析器 harness/collect_metrics.py 认这个记号并写成 JSON 的 null。
        def _f3(v):
            return "n/a" if v is None else f"{v:.3f}"

        print(f"[Backdoor] Round {round_idx} | "
              f"GM_ASR={_f3(metrics['global_asr'])} | "
              f"EM_ASR={_f3(metrics['edge_asr_mean'])} | "
              f"local_benign={_f3(metrics['local_asr_benign_mean'])} "
              f"(same_edge={_f3(metrics['local_asr_same_edge'])}, "
              f"diff_edge={_f3(metrics['local_asr_diff_edge'])}) | "
              f"local_malicious={_f3(metrics['local_asr_malicious_mean'])}\n")

        # 逐 edge 面板（Experiment 3：per-edge 传播路径。聚合均值会抹平 amplification/
        # dilution/cross-edge cancellation，所以每个 edge 单独打一条可解析行）。
        for pe in metrics.get("per_edge", []):
            print(f"[Backdoor] Round {round_idx} | edge{pe['edge_id']} | "
                  f"edge_asr={_f3(pe['edge_asr'])} | "
                  f"client_benign={_f3(pe['client_benign'])} | "
                  f"client_malicious={_f3(pe['client_malicious'])} | "
                  f"n_benign={pe['n_benign']} | n_malicious={pe['n_malicious']} | "
                  f"has_malicious={pe['has_malicious']}")

        # ── 漂移测量（Experiment 3C）：edge 相对本轮起点全局的参数/表示漂移 ──────
        if self.drift_eval and anchor is not None:
            _t0 = time.perf_counter()
            self._eval_drift(round_idx, anchor)
            t_drift = time.perf_counter() - _t0

        # ── 计时行（可解析；harness/collect_metrics.py 的 RE_TIMING 认它）─────
        def _t3(v):
            return "n/a" if v is None else f"{v:.1f}"

        print(f"[Timing] Round {round_idx} | asr={_t3(t_asr)}s | "
              f"feature={_t3(t_feature)}s | forgetting={_t3(t_forget)}s | "
              f"drift={_t3(t_drift)}s | total={_t3(time.perf_counter() - _t_all)}s")
        # asr 的分项：主列 / 白盒副列 / 陈旧副列（D-047：副列降频之前先量）。
        # 独立一行，不往 [Timing] 里加字段 —— RE_TIMING 是全或无的正则，
        # 格式一变原有六个字段会一起变 None（[设定2] 同一教训）。
        def _r2(v):
            return None if v is None else round(v, 2)

        print(format_kv("[TimingASR]", {"main": _r2(t_asr_main),
                                         "whitebox": _r2(t_side["whitebox"]),
                                         "stale": _r2(t_side["stale"])},
                        round_idx=round_idx, digits=2))

        # ── 任务3：wandb 记录全部分层指标 ─────────────────────────────────
        FLLogger.log_round_metrics(round_idx, metrics)
        # 兼容旧看板字段
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"backdoor/c_acc": metrics["local_acc_mean"],
                           "backdoor/asr": metrics["local_asr_benign_mean"]},
                          step=round_idx)
        except Exception:
            pass

    def _eval_drift(self, round_idx: int, anchor):
        """Experiment 3C：每 edge 相对 anchor（本轮起点全局）的参数/表示漂移，打一条可解析行。"""
        if self.x_test is None or not self.edge_servers:
            return
        if self._drift_base_idx is None:
            num_classes = int(self.config["data"]["num_classes"])
            split = get_base_head_indices(self.global_model, num_classes)
            self._drift_base_idx = split["base_weight_indices"]
        n = min(self.drift_probe_samples, len(self.x_test))
        x_probe = self.x_test[:n]                       # 固定干净探针（确定性、跨轮跨格可比）
        d = evaluate_drift(self.edge_servers, self.global_model, anchor,
                           self._drift_base_idx, x_probe)
        print(f"[Drift] Round {round_idx} | "
              f"param_abs={d['param_abs']:.4f} | param_rel={d['param_rel']:.4f} | "
              f"repr_mean={d['repr_mean']:.4f} | repr_median={d['repr_median']:.4f} | "
              f"n_edges={len(self.edge_servers)}")
