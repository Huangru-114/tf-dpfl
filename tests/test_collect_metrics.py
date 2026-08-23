"""
tests/test_collect_metrics.py  —  日志 → metrics.json 解析器的不变量

**为什么这个解析器需要 L1**：它是回程协议的唯一出口。集群跑完只回传这一个 json，
所有 L2 判据都从它读数。解析器错一个正则，回来的就是一串**看起来很正常的错数字**
—— 这正是本仓库最贵的失败模式（config_validate 存在的同一个理由）。
它此前一条测试都没有。

纯 Python + 内联日志夹具，本地秒级。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

from collect_metrics import collect          # noqa: E402


# ── 夹具：一段结构与真实 run 完全一致的最小日志 ──────────────────────────────
def _log(strategy="vanilla", attack_lines=False, n_rounds=3,
         malicious=(0, 9), defense="none", with_pm=True,
         settings=True, client_fraction=0.1, poison_ratio=0.2,
         n_clients=10, n_edges=2, arch="fedavg_cnn", forced_participation=False,
         per_edge=False):
    L = [
        "[Config] loading ../experiments/smoke-base.yaml",
        "[Config] run_name = hier_pfedme_noniid_badnet_seed42",
        f"[配置校验] 通过 | method=hierpfedme | defense={defense} | "
        f"attack={strategy} | 0 个警告",
    ]
    if settings:
        L.append(
            f"[设定] client_fraction={client_fraction} | poison_ratio={poison_ratio} | "
            f"n_clients={n_clients} | n_edges={n_edges} | "
            f"n_malicious={len(malicious)} | forced_participation={forced_participation} | "
            f"arch={arch}")
    L.append(f"[Backdoor] resolved malicious clients (placement=spread): "
             f"{list(malicious)}")
    for c in malicious:
        L.append(f"[Backdoor] Client {c} MALICIOUS | strategy={strategy} | "
                 f"trigger=badnet | target=9 | poison=0.5")

    for r in range(1, n_rounds + 1):
        L.append(f"[Round   {r}] Broadcasting to 2 edges...")
        # 全部 10 个客户端都打印方法行（与真实日志一致）
        for c in range(10):
            L.append(f"  [Client  {c}] Round {r} | Hier-pFedMe(λ2=15.0, K=5) | "
                     f"loss=1.{r}00")
            if attack_lines and c in malicious:
                L.append(f"  [Client  {c}] Round {r} | "
                         f"Neurotoxin(mask=top-5%, ref=edge_weights)")
        if defense in ("median", "trimmed_mean"):
            # 坐标类防御：没有客户端级判决
            L.append(f"    [Decision] edge0 | {defense} | coordinate-wise | n=10")
        elif defense != "none":
            L.append(f"    [Decision] edge0 | {defense} | admitted {8 - r}/10 | "
                     f"rejected=[{r}, 9]")
        pm = f" PM=0.7{r}00" if with_pm else ""
        L.append(f"  [Cloud] GM=0.1{r}00 | EM=0.5{r}00{pm} | loss=2.0 | "
                 f"time=1.0s | comm=1.0MB (total=1MB)")
        L.append(f"[Backdoor] Round {r} | GM_ASR=0.0{r}0 | EM_ASR=0.1{r}0 | "
                 f"local_benign=0.2{r}0 (same_edge=0.3{r}0, diff_edge=0.4{r}0) | "
                 f"local_malicious=0.9{r}0")
        if per_edge:
            # edge0 含恶意、edge1 不含（collocated 拓扑），逐 edge 面板
            L.append(f"[Backdoor] Round {r} | edge0 | edge_asr=0.8{r}0 | "
                     f"client_benign=0.7{r}0 | client_malicious=0.9{r}0 | "
                     f"n_benign=4 | n_malicious=2 | has_malicious=True")
            L.append(f"[Backdoor] Round {r} | edge1 | edge_asr=0.1{r}0 | "
                     f"client_benign=0.0{r}0 | client_malicious=0.000 | "
                     f"n_benign=4 | n_malicious=0 | has_malicious=False")
    L.append("[Final PM] weighted C-Acc = 0.8136 over 10 clients / 14996 samples")
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════
# 1. 自描述：这份数字到底跑了什么
# ══════════════════════════════════════════════════════════════════════════
def test_run_info_is_self_describing():
    """
    `--config` 曾经被静默忽略（见 test_config_cli.py），跑出来的 metrics.json
    与「按预期跑」的那份长得一模一样。所以每份 metrics 必须自己写明跑了什么，
    否则事后无法判读。
    """
    m = collect(_log(defense="flame"))
    run = m["run"]
    assert run["config_path"] == "../experiments/smoke-base.yaml"
    assert run["run_name"] == "hier_pfedme_noniid_badnet_seed42"
    assert run["method"] == "hierpfedme"
    assert run["defense"] == "flame"
    assert run["attack"] == "vanilla"
    assert run["n_rounds"] == 3
    assert run["malicious_ids"] == [0, 9]


def test_settings_are_self_describing():
    """
    收口陷阱 #7 的同类：client_fraction / poison_ratio 此前**不进** run 块，
    「改成论文值」无法从 artifact 证实（exp006 即栽在这里：participation 反证
    实为全参与，而 metrics.json 里根本看不到 fraction）。现在必须能读出来。
    """
    m = collect(_log(client_fraction=0.1, poison_ratio=0.2, n_clients=100,
                     n_edges=2, arch="resnet10", malicious=tuple(range(10)),
                     forced_participation=True))
    run = m["run"]
    assert run["client_fraction"] == pytest.approx(0.1)
    assert run["poison_ratio"] == pytest.approx(0.2)
    assert run["n_clients"] == 100
    assert run["n_edges"] == 2
    assert run["n_malicious"] == 10
    assert run["forced_participation"] is True
    assert run["arch"] == "resnet10"


def test_settings_absent_in_legacy_log_is_none_not_zero():
    """
    老日志没有 [设定] 行时，这几个字段必须是 None（未知）而不是 0/默认值 ——
    0.0 的 fraction 会被误读成「跑了论文设定」，正是要防的误判。
    """
    m = collect(_log(settings=False))
    run = m["run"]
    assert run["client_fraction"] is None
    assert run["poison_ratio"] is None
    assert run["n_clients"] is None
    assert run["forced_participation"] is None
    assert run["arch"] is None


def test_per_edge_panel_is_collected():
    """
    Experiment 3：逐 edge 面板必须解析成 {round: [per-edge...]}，聚合均值会抹平
    amplification/dilution/cancellation。edge_id 升序，末轮快照单列。
    """
    m = collect(_log(n_rounds=3, per_edge=True))
    per = m["per_edge_rounds"]
    assert set(per.keys()) == {1, 2, 3}
    r3 = per[3]
    assert [e["edge_id"] for e in r3] == [0, 1], "edge 必须按 id 升序"
    e0, e1 = r3
    assert e0["edge_asr"] == pytest.approx(0.830)
    assert e0["client_benign"] == pytest.approx(0.730)
    assert e0["n_malicious"] == 2 and e0["has_malicious"] is True
    assert e1["has_malicious"] is False and e1["client_malicious"] == pytest.approx(0.0)
    assert m["per_edge_final"] == r3, "per_edge_final 必须是末轮快照"


def test_per_edge_absent_is_empty_not_crash():
    """老日志没有逐 edge 行时，per_edge_rounds 为空、per_edge_final 为 []，不报错。"""
    m = collect(_log(n_rounds=2, per_edge=False))
    assert m["per_edge_rounds"] == {}
    assert m["per_edge_final"] == []


def test_malicious_ids_fallback_to_declaration_lines():
    """resolved 那行缺失时（老日志），要能从逐条 MALICIOUS 声明里回退解析。"""
    text = _log()
    text = "\n".join(ln for ln in text.splitlines()
                     if "resolved malicious" not in ln)
    assert collect(text)["run"]["malicious_ids"] == [0, 9]


# ══════════════════════════════════════════════════════════════════════════
# 2. 准确率：CLAUDE.md 要求 metrics.json 含 acc，旧实现一个都没采
# ══════════════════════════════════════════════════════════════════════════
def test_accuracy_is_collected_per_round():
    m = collect(_log(n_rounds=3))
    assert [a["round"] for a in m["acc_rounds"]] == [1, 2, 3]
    assert m["acc_rounds"][0]["gm_acc"] == pytest.approx(0.1100)
    assert m["acc_rounds"][2]["em_acc"] == pytest.approx(0.5300)
    assert m["acc_rounds"][2]["pm_acc"] == pytest.approx(0.7300)


def test_pm_is_optional():
    """PM 只在 eval_interval 轮打印，缺失时必须是 None 而不是让整行解析失败。"""
    m = collect(_log(with_pm=False))
    assert len(m["acc_rounds"]) == 3
    assert all(a["pm_acc"] is None for a in m["acc_rounds"])
    assert all(a["gm_acc"] is not None for a in m["acc_rounds"])


def test_final_acc_includes_weighted_pm():
    m = collect(_log(n_rounds=3))
    assert m["final_acc"]["round"] == 3
    assert m["final_acc"]["gm_acc"] == pytest.approx(0.1300)
    assert m["final_acc"]["final_pm_weighted"] == pytest.approx(0.8136)


def test_acc_round_numbers_come_from_round_headers():
    """
    轮号必须来自 `[Round N]` 行，不能靠「第 k 条 Cloud 行 = 第 k 轮」。
    中途崩掉时后者会整体错位，而错位的曲线看起来完全正常。
    """
    text = _log(n_rounds=3)
    # 模拟第 1 轮的 Cloud 行缺失（例如那一轮评估抛异常被吞）
    lines = [ln for ln in text.splitlines()
             if not (ln.strip().startswith("[Cloud] GM=0.1100"))]
    m = collect("\n".join(lines))
    assert [a["round"] for a in m["acc_rounds"]] == [2, 3], (
        "轮号没有跟随 [Round N] 头 —— 缺一条 Cloud 行就会让整条 acc 曲线错位")


# ══════════════════════════════════════════════════════════════════════════
# 3. 恶意参与度：vanilla 策略下旧实现恒读 0
# ══════════════════════════════════════════════════════════════════════════
def test_participation_counted_for_vanilla_strategy():
    """
    这是旧实现的真 bug：vanilla（badnet/blended/dba）的恶意客户端打印的是
    普通方法行，没有 Neurotoxin/Bad-PFL/CerP 标记。旧正则只认那三种标记，
    于是所有 vanilla 配置恒读 0 —— 而 0 同时也是「攻击根本没被选中」的取值。
    """
    m = collect(_log(strategy="vanilla", attack_lines=False, n_rounds=5))
    assert m["malicious_selected_rounds"] == [1, 2, 3, 4, 5]
    assert m["n_malicious_participations"] == 5
    assert m["n_malicious_participations"] == m["run"]["n_rounds"], (
        "L2-a 判据（n_malicious_participations == n_rounds）必须能从这份 json 直接读出")


def test_participation_counted_for_mixin_strategy():
    """mixin 策略（每轮多打一行）不能因为重复行把轮次数算多。"""
    m = collect(_log(strategy="neurotoxin", attack_lines=True, n_rounds=5))
    assert m["malicious_selected_rounds"] == [1, 2, 3, 4, 5]
    assert m["n_malicious_participations"] == 5


def test_participation_ignores_benign_clients():
    """只统计恶意 id；良性客户端每轮也在打印同样格式的行。"""
    m = collect(_log(malicious=(3,), n_rounds=2))
    assert m["run"]["malicious_ids"] == [3]
    assert m["malicious_participation_by_client"] == {"3": [1, 2]}


def test_partial_participation_is_visible():
    """
    fix-frequency（Q>1）下恶意客户端只在部分轮参与，必须如实反映 ——
    这正是用来核对 Q 是否生效的数。
    """
    text = _log(malicious=(0,), n_rounds=4)
    # 抹掉第 2、3 轮里 client 0 的行（模拟被换出）
    lines = [ln for ln in text.splitlines()
             if not (("[Client  0] Round 2" in ln) or ("[Client  0] Round 3" in ln))]
    m = collect("\n".join(lines))
    assert m["malicious_selected_rounds"] == [1, 4]
    assert m["n_malicious_participations"] == 2
    assert m["n_malicious_participations"] != m["run"]["n_rounds"]


# ══════════════════════════════════════════════════════════════════════════
# 4. 防御判决
# ══════════════════════════════════════════════════════════════════════════
def test_admitted_is_none_when_no_defense():
    """defense=none 时没有判决行，均值必须是 None 而不是 0（0 会被读成「全剔除」）。"""
    m = collect(_log(defense="none"))
    assert m["admitted"] == []
    assert m["admitted_count_mean"] is None


def test_admitted_count_mean():
    m = collect(_log(defense="flame", n_rounds=3))
    assert [a["admitted"] for a in m["admitted"]] == [7, 6, 5]
    assert m["admitted_count_mean"] == pytest.approx(6.0)
    assert all(a["total"] == 10 for a in m["admitted"])
    assert all(a["layer"] == "edge0" for a in m["admitted"])


def test_rejected_ids_are_aggregated():
    """TPR/FPR 的原料：到底剔掉了哪些 client_id（不是位置索引）。"""
    m = collect(_log(defense="flame", n_rounds=3))
    assert m["rejected_ids"] == [1, 2, 3, 9]


@pytest.mark.parametrize("defense", ["median", "trimmed_mean"])
def test_coordinate_wise_defense_has_no_client_level_decision(defense):
    """
    坐标类防御不做客户端级筛选，`admitted` 必须是 None 而不是 0 ——
    0 会被读成「本轮全部客户端被剔除」，那是完全不同的结论。
    """
    m = collect(_log(defense=defense, n_rounds=3))
    assert len(m["admitted"]) == 3
    assert all(a["admitted"] is None for a in m["admitted"])
    assert all(a["total"] == 10 for a in m["admitted"])
    assert m["admitted_count_mean"] is None
    assert m["rejected_ids"] == []


def test_uniform_decision_line_covers_every_defense():
    """
    统一判决行由 robust_mean 这个唯一收口处发出，所以任何防御名都能解析 ——
    包括本测试写这行时还不存在的防御。
    旧实现只认 flame 自己那句，multi_krum / dnc / trimmed_mean / median 全瞎。
    """
    for name in ["flame", "multi_krum", "dnc", "some_future_defense"]:
        m = collect(_log(defense=name, n_rounds=2))
        assert len(m["admitted"]) == 2, f"{name} 的判决行没被解析到"
        assert all(a["defense"] == name for a in m["admitted"])


def test_legacy_flame_line_still_parses():
    """老日志里只有 flame 专属行，仍要能读出来（回程数据不能因为改格式作废）。"""
    text = _log(defense="none", n_rounds=2)
    text += "\n    [Defense:flame] admitted 7/10 updates (cos_thresh=0.512)"
    m = collect(text)
    assert m["admitted"] == [{"layer": None, "defense": "flame",
                              "admitted": 7, "total": 10, "rejected": None}]
    assert m["admitted_count_mean"] == pytest.approx(7.0)


# ══════════════════════════════════════════════════════════════════════════
# 5. 被吞掉的客户端异常 —— 「ASR=0」最常见的真实原因
# ══════════════════════════════════════════════════════════════════════════
def test_client_failures_are_surfaced():
    """
    _collect_updates_parallel 会 catch 掉客户端异常、把它踢出聚合，run 照常跑完。
    如果恰好是恶意客户端每轮都抛异常，ASR 必然是 0，而日志之外没有任何线索。
    """
    text = _log() + (
        "\n    [ERROR] Client 9: Incompatible shapes: [8,8,8,3] vs. [8,16,16,3]"
        "\n    [Edge 0] 1 个客户端本轮失败并被丢弃: [9]")
    m = collect(text)
    assert any(f.get("client_id") == 9 for f in m["client_failures"])
    assert any(f.get("dropped") == [9] for f in m["client_failures"])


def test_no_failures_on_clean_log():
    assert collect(_log())["client_failures"] == []


# ══════════════════════════════════════════════════════════════════════════
# 6. 后门指标（旧字段，不能因为重构而漂）
# ══════════════════════════════════════════════════════════════════════════
def test_backdoor_rounds_unchanged():
    m = collect(_log(n_rounds=3))
    assert [r["round"] for r in m["rounds"]] == [1, 2, 3]
    assert m["final"]["global_asr"] == pytest.approx(0.030)
    assert m["final"]["local_malicious_asr"] == pytest.approx(0.930)
    assert m["final"]["same_edge_asr"] == pytest.approx(0.330)


def test_empty_log_does_not_crash():
    """跑崩了的 run 也要能压出 json —— 否则连 traceback 都回传不了。"""
    m = collect("Traceback (most recent call last):\nValueError: boom")
    assert m["final"] is None
    assert m["acc_rounds"] == []
    assert m["n_malicious_participations"] == 0
    assert m["errors"]
