"""
攻击基线第一档（PREREG-v2 第七节 A1；D-141）：静态触发器逐批投毒（`malicious_strategy: static`）的本地侧。不需要 TF。

  · 纯算术 attack/static_poison.apply_static_poison：选中行 = trigger(x) + 目标标签，其余逐位不变，输入不被改；
  · mixin（client/client_static.py）的结构：闸门 / 先 super() / 用 self.rng + poison_mask / on_upload 纯函数；
  · main.py 的接线：static → StaticPoisonMixin、保留干净数据 + 强制串行（与 Bad-PFL 同列）、
    触发器注入与旧路径同一个表达式（DBA = 本端局部触发器）；
  · config_validate：static 只配 per_epoch、vanilla + per_epoch 仍拒绝、云聚合后评估点放开、冻结触发器列仍只给
    Bad-PFL、攻击时间窗可用、抽样开关与云聚合后评估点不再误报「只对 Bad-PFL」；
  · 登记表：ATKP / ATK 与 SNAP__collocated__s42 只差攻击设置（DBA 是官方 4 块）；ATK 等探路过闸；
  · 打印 ↔ 解析：[设定15] → run.static / trigger / …；[Poison] → poison_counts；老日志 → None / []；
  · status 只对 static 核对触发器；runs_table 把三种静态攻击分成三格、其余文件因素键不变。
"""

import ast
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
FEDAVG = ROOT / "fedavg"
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(FEDAVG))

import collect_metrics as CM                                    # noqa: E402
import registry as R                                            # noqa: E402
import runs_table as T                                          # noqa: E402
import status as ST                                             # noqa: E402
from attack.poison_mask import poison_mask                      # noqa: E402
from attack.static_poison import apply_static_poison            # noqa: E402
from config_validate import ConfigError, static_settings_row, validate_config   # noqa: E402
from utils.kvline import format_kv                              # noqa: E402

REG = ROOT / "experiments" / "defense" / "edge-native" / "registry.yaml"
OFFICIAL_DBA = [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
                [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]],
                [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
                [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]]
META = {"meta.group", "meta.run_id"}


# ══════════════════════════════════════════════════════════════════════════
# 1. 纯算术
# ══════════════════════════════════════════════════════════════════════════
def _white_corner(x):
    x = np.array(x, np.float32, copy=True)
    x[:, -3:, -3:, :] = 7.0
    return x


def _batch(n=8, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, 6, 6, 3)).astype(np.float32), (np.arange(n) % 10).astype(np.int64)


def test_selected_rows_get_the_trigger_and_target_others_bitwise_unchanged():
    x, y = _batch()
    x0, y0 = x.copy(), y.copy()
    mask = np.array([1, 0, 0, 1, 0, 1, 0, 0], bool)
    xp, yp, k = apply_static_poison(x, y, mask, _white_corner, target=0)
    assert k == 3
    assert np.array_equal(xp[mask], _white_corner(x[mask]))
    assert np.all(yp[mask] == 0)
    assert np.array_equal(xp[~mask], x[~mask]) and np.array_equal(yp[~mask], y[~mask])
    assert np.array_equal(x, x0) and np.array_equal(y, y0)          # 输入不被就地改
    assert yp.dtype == y.dtype and xp.dtype == np.float32


def test_trigger_on_selected_rows_equals_trigger_on_whole_batch_then_select():
    """静态触发器逐样本独立 → 只喂选中的行与整批喂进去再挑，逐位相同（Bad-PFL 是整批算 ξ/δ、mask 只做选择）。"""
    x, y = _batch(seed=1)
    mask = np.array([0, 1, 1, 0, 0, 0, 1, 1], bool)

    def blend(z):
        return 0.8 * np.asarray(z, np.float32) + 0.2 * np.float32(0.5)
    xp, _, _ = apply_static_poison(x, y, mask, blend, target=3)
    assert np.array_equal(xp, np.where(mask[:, None, None, None], blend(x), x))


def test_empty_mask_is_the_identity_and_never_calls_the_trigger():
    x, y = _batch()

    def boom(_):
        raise AssertionError("一张都没选中时不该调触发器")
    xp, yp, k = apply_static_poison(x, y, np.zeros(8, bool), boom, target=0)
    assert k == 0 and np.array_equal(xp, x) and np.array_equal(yp, y)


def test_mask_shape_mismatch_is_rejected():
    x, y = _batch()
    with pytest.raises(ValueError):
        apply_static_poison(x, y, np.zeros(7, bool), _white_corner, target=0)


def test_bernoulli_rate_is_the_bad_pfl_meaning_of_rho():
    """用户拍板：逐批抽、与 Bad-PFL 同一个函数 —— ρ=0.2 时被投毒的比例 ≈ 0.2（批 32 下 exact_k 是 0.1875）。"""
    rng = np.random.default_rng(0)
    n = sum(poison_mask(rng, 32, 0.2, "bernoulli").sum() for _ in range(5000))
    assert abs(n / (32 * 5000) - 0.2) < 0.005
    assert poison_mask(np.random.default_rng(0), 32, 0.2, "exact_k").sum() == 6


# ══════════════════════════════════════════════════════════════════════════
# 2. mixin 的结构（AST）—— 闸门与 super() 的顺序另由 test_attack_window 守着
# ══════════════════════════════════════════════════════════════════════════
def _fn_src(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]                                  # 去掉 docstring（文档里提到的名字不算代码读了它）
    return "\n".join(ast.unparse(s) for s in body)


def test_on_batch_gates_before_drawing_and_uses_the_client_rng():
    src = _fn_src(FEDAVG / "client" / "client_static.py", "on_batch")
    i_super, i_gate = src.find("super().on_batch"), src.find("if not self._attack_active")
    i_mask = src.find("poison_mask(self.rng")
    assert -1 < i_super < i_gate < i_mask, src           # 闸门关时一个投毒随机数都不消耗
    assert "np.random." not in src and "random." not in src.replace("poison_mask", "")


def test_on_upload_is_pure():
    src = _fn_src(FEDAVG / "client" / "client_static.py", "on_upload")
    assert "set_weights" not in src and "return upload" in src
    assert "self._attack_active" in src


def test_missing_trigger_is_an_error_not_a_silent_benign_client():
    src = (FEDAVG / "client" / "client_static.py").read_text(encoding="utf-8")
    assert "raise RuntimeError" in src and "set_static_trigger" in src


# ══════════════════════════════════════════════════════════════════════════
# 3. main.py 接线（AST / 源码）
# ══════════════════════════════════════════════════════════════════════════
MAIN = (FEDAVG / "main.py").read_text(encoding="utf-8")


def test_main_maps_static_to_the_mixin():
    assert '"static":     StaticPoisonMixin' in MAIN
    assert "from client.client_static      import StaticPoisonMixin" in MAIN


def test_static_keeps_clean_data_and_is_forced_serial_like_bad_pfl():
    """dynamic_poison 一个开关管两件事：不调 build_poisoned_dataset + 强制串行（用户定串行）。"""
    assert 'dynamic_poison = strategy in ("cerp", "badpfl", "static")' in MAIN
    i_dyn = MAIN.find("if bd_enabled and dynamic_poison:")
    assert i_dyn != -1 and 'config["federation"]["n_workers"] = 1' in MAIN[i_dyn:i_dyn + 1200]
    assert "if not dynamic_poison:" in MAIN               # 静态投毒数据集只在 vanilla 下构造


def test_trigger_injection_is_the_old_path_expression():
    assert "client.set_static_trigger(dba_triggers.get(i, bd_trigger))" in MAIN
    assert "poison_trig = dba_triggers.get(i, bd_trigger)" in MAIN       # 旧路径同一个表达式


# ══════════════════════════════════════════════════════════════════════════
# 4. 登记表 + config_validate
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def reg():
    return R.Registry(REG)


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def _diff(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def _cfg(reg, rid):
    return reg.declared_config(next(r for r in reg.runs() if r["run_id"] == rid))


def test_groups_sizes_and_gates(reg):
    runs = {r["run_id"]: r for r in reg.runs() if r["group"] in ("ATKP", "ATK")}
    assert sorted(runs) == sorted(["ATKP__badnet__s42", "ATKP__badnet-rep__s42", "ATKP__blended__s42",
                                   "ATKP__dba__s42", "ATK__badnet__s42", "ATK__blended__s42",
                                   "ATK__dba__s42"])
    assert "atk-static-code" in reg.available
    assert reg.unmet_requires("ATKP") == [] or all(t.startswith("audit") for t in reg.unmet_requires("ATKP"))
    assert "atkp-ok" in reg.unmet_requires("ATK")          # 正式组等探路过闸


@pytest.mark.parametrize("group,trig", [("ATK", "badnet"), ("ATK", "blended"), ("ATK", "dba"),
                                        ("ATKP", "badnet"), ("ATKP", "blended"), ("ATKP", "dba")])
def test_cells_differ_from_the_bad_pfl_control_only_in_the_attack(reg, group, trig):
    a = _cfg(reg, f"{group}__{trig}__s42")
    snap = _cfg(reg, "SNAP__collocated__s42")
    want = {"backdoor.malicious_strategy", "evaluation.frozen_trigger", "evaluation.snapshot_rounds"} | META
    if trig != "badnet":
        want.add("backdoor.trigger")                       # base 写的就是 badnet
    if trig == "dba":
        want.add("backdoor.dba_patterns")
    if group == "ATKP":
        want.add("federation.n_rounds")
    assert _diff(a, snap) == want
    assert a["backdoor"]["malicious_strategy"] == "static" and a["backdoor"]["trigger"] == trig
    assert a["evaluation"]["frozen_trigger"] is False and a["evaluation"]["post_agg_eval"] is True
    assert a["backdoor"]["malicious_per_edge"] == [10, 0, 0, 0] and a["backdoor"]["poison_ratio"] == 0.2
    assert a["federation"]["n_rounds"] == (4 if group == "ATKP" else 60)


def test_dba_cells_use_the_four_official_patterns(reg):
    """AI-secure/DBA utils/cifar_params.yaml 的 0–3_poison_pattern（11-24 核对）；fedavg/config/config.yaml 同值。"""
    for g in ("ATK", "ATKP"):
        assert _cfg(reg, f"{g}__dba__s42")["backdoor"]["dba_patterns"] == OFFICIAL_DBA
    old = yaml.safe_load((FEDAVG / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert old["backdoor"]["dba_patterns"] == OFFICIAL_DBA
    # base 只有前 2 块（改 base 会改所有已有配置的 sha → 只在格子里写全）
    assert _cfg(reg, "SNAP__collocated__s42")["backdoor"]["dba_patterns"] == OFFICIAL_DBA[:2]


def test_probe_repeat_is_the_same_config(reg):
    a, b = _cfg(reg, "ATKP__badnet__s42"), _cfg(reg, "ATKP__badnet-rep__s42")
    assert _diff(a, b) == {"meta.run_id"}


@pytest.fixture(scope="module")
def atk_cfgs(reg):
    return {r["run_id"]: reg.intended_config(r) for r in reg.runs() if r["group"] in ("ATKP", "ATK")}


def test_every_cell_validates_and_describes_its_trigger(atk_cfgs, capsys):
    for rid, cfg in atk_cfgs.items():
        warnings = validate_config(copy.deepcopy(cfg))
        out = capsys.readouterr().out
        line = [ln for ln in out.splitlines() if ln.startswith("[设定15]")]
        trig = cfg["backdoor"]["trigger"]
        assert line and "static=true" in line[0] and f"trigger={trig}" in line[0], rid
        assert "poison_sampling=bernoulli" in line[0], rid               # P2 模板 = 与 Bad-PFL 同一个值
        assert ("dba_n_patterns=4" in line[0]) is (trig == "dba"), rid
        joined = "\n".join(warnings)
        assert "poison_sampling" not in joined and "post_agg_eval" not in joined, rid   # 不再误报
        assert "badpfl_xi" in joined                                                    # 真只属于 Bad-PFL 的照样提醒


def _atk(atk_cfgs):
    return copy.deepcopy(atk_cfgs["ATK__badnet__s42"])


def _no_recording(c):
    """关掉网格与云聚合后评估点 —— 让下面各条撞上要测的那一道检查，而不是更早的网格 / 记录开关检查。"""
    c["evaluation"].update(eval_grid=None, post_agg_eval=False)


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: (_no_recording(c), c.pop("meta"), c["data"].update(batch_pipeline="legacy")),   # 不是 P2 配置
     "只配 data.batch_pipeline = 'per_epoch'"),
    (lambda c: (_no_recording(c), c["backdoor"].update(malicious_strategy="vanilla")), "请用 static"),
    (lambda c: (_no_recording(c), c["backdoor"].update(malicious_strategy="neurotoxin")), "per_epoch"),
    (lambda c: c["backdoor"].update(malicious_strategy="vanilla"),
     "post_agg_eval 要求 backdoor.malicious_strategy ∈ \\{badpfl, static\\}"),
    (lambda c: c["evaluation"].update(frozen_trigger=True), "frozen_trigger 要求 backdoor.malicious_strategy = badpfl"),
    (lambda c: c["backdoor"].update(trigger="kitty"), "trigger"),
    (lambda c: c["backdoor"].update(generator_schedule="always"), "generator_schedule"),
])
def test_config_validate_rejects(atk_cfgs, mutate, needle):
    c = _atk(atk_cfgs)
    mutate(c)
    with pytest.raises(ConfigError, match=needle):
        validate_config(c)


def test_without_recording_switches_static_still_validates(atk_cfgs):
    c = _atk(atk_cfgs)
    _no_recording(c)
    validate_config(c)


def test_attack_window_is_accepted_for_static(atk_cfgs):
    c = _atk(atk_cfgs)
    c["backdoor"].update(attack_start_round=5, attack_stop_round=30)
    validate_config(c)


def test_post_agg_eval_still_needs_a_fixed_attacker_for_bad_pfl(reg):
    c = reg.intended_config(next(r for r in reg.runs() if r["run_id"] == "SNAP__collocated__s42"))
    c["backdoor"]["eval_xi_model"] = "victim"
    c["evaluation"]["frozen_trigger"] = False
    with pytest.raises(ConfigError, match="fixed_attacker"):          # 网格那道检查先拦住；放开 static 没放开它
        validate_config(c)


def test_settings_row_is_static_false_for_every_other_strategy(reg):
    for r in reg.runs():
        if r["group"] in ("ATK", "ATKP"):
            continue
        assert static_settings_row(reg.intended_config(r)) == {"static": False}, r["run_id"]


# ══════════════════════════════════════════════════════════════════════════
# 5. 打印 ↔ 解析，status，runs_table
# ══════════════════════════════════════════════════════════════════════════
HEAD = ["[配置校验] 通过 | method=hier_fedrep | defense=none | attack=static | 0 个警告"]


def test_settings15_round_trips_into_the_run_block(atk_cfgs):
    for rid, cfg in atk_cfgs.items():
        row = static_settings_row(cfg)
        m = CM.collect("\n".join(HEAD + [format_kv("[设定15]", row)]))
        run = m["run"]
        assert run["static"] is True and run["trigger"] == cfg["backdoor"]["trigger"], rid
        assert run["poison_sampling"] == "bernoulli"
        assert run["dba_n_patterns"] == (4 if cfg["backdoor"]["trigger"] == "dba" else None)
        assert m["schema_version"] == 16


def test_old_and_non_static_logs_parse_to_none():
    old = CM.collect("\n".join(HEAD))["run"]
    assert all(old[k] is None for k in ("static", "trigger", "dba_n_patterns", "poison_sampling"))
    bad = CM.collect("\n".join(HEAD + [format_kv("[设定15]", {"static": False})]))["run"]
    assert bad["static"] is False and bad["trigger"] is None
    assert CM.collect("\n".join(HEAD))["poison_counts"] == []


def test_poison_lines_accumulate_per_client():
    lines = HEAD + [
        format_kv("[Poison]", {"client": 7, "n_poisoned": 13, "n_seen": 64, "n_batches": 2}, round_idx=1),
        format_kv("[Poison]", {"client": 3, "n_poisoned": 0, "n_seen": 32, "n_batches": 1}, round_idx=1),
        format_kv("[Poison]", {"client": 7, "n_poisoned": 5, "n_seen": 32, "n_batches": 1}, round_idx=2),
    ]
    assert CM.collect("\n".join(lines))["poison_counts"] == [
        {"client": 3, "reports": 1, "n_poisoned": 0, "n_seen": 32},
        {"client": 7, "reports": 2, "n_poisoned": 18, "n_seen": 96},
    ]


def test_status_checks_the_trigger_only_for_static(reg):
    runs = {r["run_id"]: r for r in reg.runs()}
    exp = reg.expected_run_block(runs["ATK__dba__s42"])
    assert exp["trigger"] == "dba"
    got = dict(exp, trigger="badnet")
    assert ST.run_block_mismatches(exp, got) == [["trigger", "dba", "badnet"]]
    assert "trigger" not in reg.expected_run_block(runs["SNAP__collocated__s42"])   # 老 run 不会因此变 mismatch


def test_runs_table_separates_the_three_triggers_and_leaves_old_keys_alone():
    base = {"attack": "static", "method": "hier_fedrep", "malicious_per_edge": [10, 0, 0, 0]}
    keys = {T.factor_key(dict(base, trigger=t)) for t in ("badnet", "blended", "dba")}
    assert len(keys) == 3
    old = {"attack": "badpfl", "method": "hier_fedrep"}
    assert T.factor_key(old) == T.factor_key(dict(old, trigger=None))
