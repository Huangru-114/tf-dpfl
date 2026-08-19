# tf-dpfl —— 分层联邦学习 后门攻防 hub

TensorFlow 实现的分层（client → edge → cloud）个性化联邦学习，
用于研究后门攻击与鲁棒聚合防御。

活跃代码只有 **`fedavg/`**。

## 快速开始

```bash
# L1：算法不变量（本地，秒级）
bash run_l1.sh

# L2：端到端 smoke（集群，约 3 分钟）→ 产出小 metrics.json
bash run_smoke.sh attack neurotoxin neurotoxin none exp001

# 完整实验（集群）
cd fedavg && python main.py --config config/config.yaml
```

## 怎么读这个仓库

| 文件 | 作用 |
|---|---|
| [`SETUP.md`](SETUP.md) | 仓库组织与集群交互标准、两级验收、整合循环 |
| [`CLAUDE.md`](CLAUDE.md) | 会话约定 + **已确认陷阱清单**（先读这个） |
| [`methods-registry.md`](methods-registry.md) | 所有方法的台账 / 研究看板 |
| `experiments/<axis>/<method>/current-focus.md` | 每个方法当前唯一要回答的问题 |

## 正交实验轴

| 轴 | config 字段 | 取值 |
|---|---|---|
| PFL 方法 | `training.drift_correction` | `hierfedavg` / `hierpfedme` / `hier_ditto` / `hier_fedrep` / … |
| 攻击 | `backdoor.malicious_strategy` × `backdoor.trigger` | `vanilla`/`neurotoxin`/`cerp`/`badpfl` × `badnet`/`blended`/`dba` |
| 防御 | `defense.name` | `none`/`trimmed_mean`/`median`/`multi_krum`/`flame`/`dnc`/`simple_tuning` |
| 数据分布 | `federation.partition` | `iid`/`noniid`/`pathological`/`superclass_pathological`/`hierarchical` |

> ⚠️ 攻击轴与 PFL 方法轴目前**尚未真正正交**（`main.py:395` 用替换而非组合）。
> 见 CLAUDE.md 陷阱 #1 与 `tests/test_attack_method_orthogonality.py`。

## 红线

`*.h5 / *.pt / *.ckpt / 大 npy / *.log / wandb/ / runs/ / venv` **永不进 git**。
checkpoint 留集群，git 里只放 manifest（见 `results/README.md`）。

refactor 前的完整状态保留在分支 `claude/repo-refactor-review-xgyi0k`（commit `8ae9b48`），
含 `fedavg_v2/`、`HierDP-FL/`、sweep 权重与 wandb 记录。要取回其中某个文件：

```bash
git checkout claude/repo-refactor-review-xgyi0k -- <path>
```
