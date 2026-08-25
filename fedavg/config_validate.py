"""
config_validate.py  –  启动时的配置兼容性校验（fail fast）

**存在理由**：这个仓库里最贵的一类错误不是崩溃，而是**静默跑错**——
config 被接受、日志一切正常、跑完 24 小时才发现那一格结果没有意义。
已确认的两种：

  1. `drift_correction` 拼错 / 用了已移除的值 → `_select_method_classes` 的
     `.get(method, PFedMeClient)` 让它**静默退化成 pFedMe**。
     你以为在跑 fedprox，实际跑的是 pFedMe。
  2. 某些 PFL 方法的 edge 聚合不走 `robust_mean` → `defense` 轴静默失效，
     而日志照样打印 `[Defense] enabled: flame`。

三维实验矩阵里这两种错误会成片出现且没有征兆，所以在 run 开始前挡住。

用法（main.py 在构建任何对象之前调用）：
    from config_validate import validate_config
    validate_config(config)          # 不兼容直接 raise ConfigError 退出
"""


class ConfigError(ValueError):
    """配置不兼容。消息里必须写清楚：哪里不对、合法值是什么、怎么改。"""


# ── PFL 方法轴 ───────────────────────────────────────────────────────────
# 键 = config["training"]["drift_correction"] 的合法取值
# 值 = 该方法的 edge 聚合是否经过 EdgeServerBase.robust_mean（即 defense 轴是否生效）
#
# 新增方法时**必须**在这里登记，否则 validate_config 会拒绝启动。
# 这张表就是「防御轴覆盖」的单一事实来源。
METHOD_SUPPORTS_DEFENSE = {
    "fedavg":          True,   # FedAvgEdgeServer
    "hierfedavg":      True,   # FedAvgEdgeServer
    "pfedme":          True,   # PFedMeEdgeServer
    "hierpfedme":      True,   # PFedMeEdgeServer
    "hier_ditto":      True,   # HierDittoEdgeServer
    "hier_fedrep":     True,   # HierFedRepEdgeServer
    "hier_ditto_rep":  True,   # HierDittoRepEdgeServer
    "hier_pfedme_rep": True,   # 继承 HierDittoRepEdgeServer
}

# 已移除的方法 → 给出明确的迁移说明，而不是「未知取值」
REMOVED_METHODS = {
    "hier_perfedavg": (
        "Hier-PerFedAvg 已移除：它的 edge 层聚合的是**元梯度**而非权重，"
        "与 defense 接口（对 W_i − G_{t-1} 做范数裁剪）语义不兼容，"
        "在三维矩阵里只会产出无法解释的格子。"),
    "perfedavg": "同 hier_perfedavg，已移除。",
}

# 曾经在 config 注释里出现、但当前类分发**不实现**的取值。
# 它们会落到 .get(method, PFedMeClient) 的默认分支 → 静默变成 pFedMe。
UNIMPLEMENTED_METHODS = {
    "fedprox": "当前 client/edge 类分发不实现 fedprox，会静默退化成 pFedMe。",
    "feddyn":  "当前 client/edge 类分发不实现 feddyn，会静默退化成 pFedMe。",
    "scaffold": "当前 client/edge 类分发不实现 scaffold，会静默退化成 pFedMe。",
}

# ── 防御轴 ───────────────────────────────────────────────────────────────
AGGREGATION_DEFENSES = {"trimmed_mean", "median", "multi_krum", "flame", "dnc"}
POST_HOC_DEFENSES    = {"simple_tuning"}
VALID_DEFENSES       = {"none"} | AGGREGATION_DEFENSES | POST_HOC_DEFENSES

# ── 攻击轴 ───────────────────────────────────────────────────────────────
VALID_STRATEGIES = {"vanilla", "neurotoxin", "cerp", "badpfl"}
VALID_TRIGGERS   = {"badnet", "blended", "dba"}

# 需要行为 mixin 的攻击策略（vanilla 只换数据集，不需要 mixin）。
# CLAUDE.md 陷阱 #1 已修复：这些策略现在是 mixin，与 PFL 方法类**组合**而非替换
# （main.py:resolve_client_classes + client/compose.py），因此配任何方法都可解释。
# 守卫：tests/test_attack_method_orthogonality.py
MIXIN_STRATEGIES = {"neurotoxin", "cerp", "badpfl"}

# 防御可作用的层。用户在 defense.layers 里选，缺省 ["edge"]。
VALID_DEFENSE_LAYERS = {"client", "edge", "cloud", "post_hoc"}


def _fail(msg: str):
    raise ConfigError("\n[配置校验失败] " + msg + "\n")


# ── 防御层的读取辅助（懒导入：defense 包是纯 numpy，但没必要在模块顶层拉进来）──
def _configured_layers(dcfg: dict) -> tuple:
    from defense import configured_layers
    return configured_layers({"defense": dcfg})


def _declared_layers(name: str) -> set:
    from defense import defense_class
    cls = defense_class(name)
    return set(getattr(cls, "layers", ())) if cls is not None else set()


def _client_mixin_of(name: str):
    from defense import defense_class
    cls = defense_class(name)
    return getattr(cls, "client_mixin", None) if cls is not None else None


def validate_config(config: dict, strict_orthogonality: bool = False) -> list:
    """
    校验 config 的跨轴兼容性。

    Args:
        strict_orthogonality: **已失效的参数**，保留只为兼容老 config。
                              陷阱 #1 已修复（攻击是 mixin，与方法类组合而非替换），
                              不再存在「不正交的格子」，因此这个开关无事可做。
                              置 True 时会给一条提示，让人去删掉这条配置。
    Returns:
        warnings: 非致命问题的文字列表（同时已打印）。
    Raises:
        ConfigError: 存在会导致「静默跑错」的组合。
    """
    warnings = []

    if strict_orthogonality:
        warnings.append(
            "experiment.strict_orthogonality 已失效：陷阱 #1 已修复（攻击策略是 mixin，"
            "与 PFL 方法类组合而非替换），不再有「不正交的格子」需要拦。可以删掉这条配置。")

    method   = str(config.get("training", {}).get("drift_correction", "")).lower()
    dcfg     = config.get("defense", {}) or {}
    defense  = str(dcfg.get("name", "none")).lower()
    bd       = config.get("backdoor", {}) or {}
    fed      = config.get("federation", {}) or {}
    data     = config.get("data", {}) or {}

    # ── 1. PFL 方法轴 ────────────────────────────────────────────────────
    if method in REMOVED_METHODS:
        _fail(f"training.drift_correction = {method!r}\n"
              f"  {REMOVED_METHODS[method]}\n"
              f"  合法取值：{sorted(METHOD_SUPPORTS_DEFENSE)}")
    if method in UNIMPLEMENTED_METHODS:
        _fail(f"training.drift_correction = {method!r}\n"
              f"  {UNIMPLEMENTED_METHODS[method]}\n"
              f"  合法取值：{sorted(METHOD_SUPPORTS_DEFENSE)}")
    if method not in METHOD_SUPPORTS_DEFENSE:
        _fail(f"未知的 training.drift_correction = {method!r}\n"
              f"  它会落到 _select_method_classes 的默认分支，**静默变成 pFedMe**。\n"
              f"  合法取值：{sorted(METHOD_SUPPORTS_DEFENSE)}\n"
              f"  新增方法请先登记到 config_validate.METHOD_SUPPORTS_DEFENSE。")

    # ── 2. 防御轴 ────────────────────────────────────────────────────────
    if defense not in VALID_DEFENSES:
        _fail(f"未知的 defense.name = {defense!r}\n"
              f"  合法取值：{sorted(VALID_DEFENSES)}")

    if defense in AGGREGATION_DEFENSES and not METHOD_SUPPORTS_DEFENSE[method]:
        _fail(f"defense.name = {defense!r} 与 drift_correction = {method!r} 不兼容。\n"
              f"  该方法的 edge 聚合不经过 robust_mean，防御会**静默失效**"
              f"（日志仍会打印 [Defense] enabled）。\n"
              f"  要么换方法，要么把该方法的 edge 聚合改走 robust_mean 并更新"
              f" config_validate.METHOD_SUPPORTS_DEFENSE。")

    # 2b. 防御的「声明层」与「配置层」必须对得上 —— 否则又是一种静默失效：
    #     用户写了 layers: [client] 但该防御根本没有客户端侧实现，
    #     create_defense 会在那一层返回 None，日志却看不出少了什么。
    cfg_layers = _configured_layers(dcfg)
    bad_layers = set(cfg_layers) - VALID_DEFENSE_LAYERS
    if bad_layers:
        _fail(f"未知的 defense.layers 取值：{sorted(bad_layers)}\n"
              f"  合法取值：{sorted(VALID_DEFENSE_LAYERS)}")
    if defense in AGGREGATION_DEFENSES:
        declared = _declared_layers(defense)
        unsupported = set(cfg_layers) - declared
        if unsupported:
            _fail(f"defense.name = {defense!r} 配置在 {sorted(unsupported)} 层，"
                  f"但该防御类只声明支持 {sorted(declared)}。\n"
                  f"  create_defense 会在那些层返回 None → 防御**静默失效**。\n"
                  f"  要么改 defense.layers，要么在该防御类的 `layers` 类属性里登记"
                  f"并把那一层真正接线。")
        if "client" in cfg_layers and _client_mixin_of(defense) is None:
            _fail(f"defense.name = {defense!r} 配了 client 层，但该防御类没有给出"
                  f" `client_mixin`。\n"
                  f"  主动防御必须提供客户端侧行为 mixin，否则客户端什么也不会做。")

    # ── 3. 攻击轴 ────────────────────────────────────────────────────────
    bd_enabled = bool(bd.get("enabled", False))
    if bd_enabled:
        strategy = str(bd.get("malicious_strategy", "vanilla")).lower()
        trigger  = str(bd.get("trigger", "badnet")).lower()
        if strategy not in VALID_STRATEGIES:
            _fail(f"未知的 backdoor.malicious_strategy = {strategy!r}\n"
                  f"  合法取值：{sorted(VALID_STRATEGIES)}")
        if trigger not in VALID_TRIGGERS:
            _fail(f"未知的 backdoor.trigger = {trigger!r}\n"
                  f"  合法取值：{sorted(VALID_TRIGGERS)}")

        # 陷阱 #1 已修复：攻击策略是 mixin，与方法类组合而非替换
        # （main.py:resolve_client_classes）。这里不再告警。
        # 守卫在 tests/test_attack_method_orthogonality.py —— 若组合退化回替换，
        # 那条测试会失败，而不是靠这里的一句 warning 提醒。

        # 数量 / 取值 sanity
        n_clients = int(fed.get("n_clients", 0) or 0)
        n_mal     = int(bd.get("n_malicious", 0) or 0)
        if n_mal > n_clients:
            _fail(f"backdoor.n_malicious={n_mal} > federation.n_clients={n_clients}")
        n_classes = int(data.get("num_classes", 0) or 0)
        target    = int(bd.get("target_label", 0))
        if n_classes and not (0 <= target < n_classes):
            _fail(f"backdoor.target_label={target} 超出 data.num_classes={n_classes} 的范围")
        pr = float(bd.get("poison_ratio", 0.0))
        if not (0.0 <= pr <= 1.0):
            _fail(f"backdoor.poison_ratio={pr} 不在 [0, 1]")
        if trigger == "dba" and not bd.get("dba_patterns"):
            _fail("backdoor.trigger='dba' 需要非空的 backdoor.dba_patterns")

        # by_edge 布点（Experiment 3）：早筛 malicious_per_edge 的长度/容量，别等训练中途才炸。
        placement = str(bd.get("malicious_placement", "spread")).lower()
        if placement == "by_edge":
            per_edge = bd.get("malicious_per_edge", None)
            n_edges  = int(fed.get("n_edges", 0) or 0)
            if not per_edge:
                _fail("malicious_placement=by_edge 需要 backdoor.malicious_per_edge"
                      "（按 edge id 索引的恶意端个数列表，如 [4,0,0,0]）")
            elif len(per_edge) != n_edges:
                _fail(f"backdoor.malicious_per_edge 长度 {len(per_edge)} != "
                      f"federation.n_edges {n_edges}")
            elif any(int(k) < 0 for k in per_edge):
                _fail(f"backdoor.malicious_per_edge 含负数：{per_edge}")
            elif sum(int(k) for k in per_edge) > n_clients:
                _fail(f"backdoor.malicious_per_edge 求和 {sum(int(k) for k in per_edge)} "
                      f"> n_clients {n_clients}")
            elif str(fed.get("edge_assignment", "random")).lower() != "block":
                warnings.append(
                    "malicious_placement=by_edge 建议配 edge_assignment=block（确定性连续分块），"
                    "否则 edge 成员是随机的，布点虽仍精确但不可从 id 直观预期。")

    elif defense != "none":
        warnings.append(
            f"backdoor.enabled=false 但 defense.name={defense!r}。"
            f"这是「无攻击下的防御误伤」对照组——如果不是故意的，请确认。")

    # ── 4. 联邦拓扑 sanity ───────────────────────────────────────────────
    n_clients = int(fed.get("n_clients", 0) or 0)
    n_edges   = int(fed.get("n_edges", 0) or 0)
    if n_edges > n_clients:
        _fail(f"federation.n_edges={n_edges} > n_clients={n_clients}")
    frac = float(fed.get("client_fraction", 1.0))
    if not (0.0 < frac <= 1.0):
        _fail(f"federation.client_fraction={frac} 不在 (0, 1]")

    # ── 5. 可复现性 ─────────────────────────────────────────────────────
    if "seed" not in config:
        warnings.append("config 缺 seed，实验不可复现。建议显式写死。")

    for w in warnings:
        print(f"[配置校验警告] {w}")
    print(f"[配置校验] 通过 | method={method} | defense={defense} | "
          f"attack={'off' if not bd_enabled else bd.get('malicious_strategy', 'vanilla')} | "
          f"{len(warnings)} 个警告")

    # ── 6. 设定自描述（收口陷阱 #7 的同类）──────────────────────────────────
    #   metrics.json 的 run 块此前只记录 method/defense/attack/n_rounds/malicious_ids，
    #   **恰恰不含 client_fraction / poison_ratio / n_clients / n_edges / arch** —— 而这几个
    #   正是「设定是否对齐论文」的判据。历史 exp006 因此无法证实自己跑的 fraction，
    #   participation 反证它实为全参与（见 experiments/attack/bad-pfl/current-focus.md）。
    #   在此打一条**可解析**的自描述行，collect_metrics.py 解析进 run 块，从此每个 run 自证。
    print(f"[设定] client_fraction={float(fed.get('client_fraction', 1.0))} | "
          f"poison_ratio={float(bd.get('poison_ratio', 0.0)) if bd_enabled else 0.0} | "
          f"n_clients={int(fed.get('n_clients', 0) or 0)} | "
          f"n_edges={int(fed.get('n_edges', 0) or 0)} | "
          f"edge_rounds={int(fed.get('edge_rounds', 1) or 1)} | "
          f"n_malicious={int(bd.get('n_malicious', 0) or 0) if bd_enabled else 0} | "
          f"forced_participation={bool(bd.get('forced_participation', False)) if bd_enabled else False} | "
          f"arch={config.get('model', {}).get('arch', '?')}")
    return warnings
