"""
server/backdoor_server.py  –  后门感知的 Cloud 服务器

继承 CloudServer，不改其核心训练/聚合逻辑；只在每 bd_eval_interval 轮（及最后一轮）
额外计算并记录各良性客户端的 C-Acc / ASR（训练中周期记录）。
"""

import numpy as np

from server.server import CloudServer
from attack.backdoor_eval import evaluate_backdoor


class BackdoorCloudServer(CloudServer):

    def __init__(self, *args, bd_cfg=None, x_test=None, y_test=None,
                 trigger_fn=None, malicious_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
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

    def run_round(self, round_idx: int):
        metrics = super().run_round(round_idx)
        n_rounds = int(self.config["federation"]["n_rounds"])
        if (round_idx % self.bd_eval_interval == 0) or (round_idx == n_rounds):
            self._backdoor_eval(round_idx)
        return metrics

    def _backdoor_eval(self, round_idx: int):
        xt, yt = self.x_test, self.y_test
        if self.bd_asr_max and yt is not None and len(yt) > self.bd_asr_max:
            idx = np.random.choice(len(yt), self.bd_asr_max, replace=False)
            xt, yt = xt[idx], yt[idx]

        print(f"\n[Backdoor] ===== Round {round_idx} backdoor evaluation =====")
        res = evaluate_backdoor(
            self._all_clients, xt, yt, self.trigger_fn, self.bd_target,
            self.malicious_ids, fallback_test_ds=self.test_dataset,
            round_idx=round_idx, verbose=self.bd_verbose,
        )
        self.history["bd_c_acc"].append(res["c_acc"])
        self.history["bd_asr"].append(res["asr"])
        print(f"[Backdoor] Round {round_idx} | benign avg "
              f"C-Acc={res['c_acc']:.3f} | ASR={res['asr']:.3f}\n")

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"backdoor/c_acc": res["c_acc"],
                           "backdoor/asr": res["asr"]}, step=round_idx)
        except Exception:
            pass
