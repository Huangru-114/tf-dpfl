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

# 继承 FedAvgClient、会**替换掉** PFL 方法类的攻击策略（CLAUDE.md 陷阱 #1）。
# 正交化修好之前，这些策略配非 fedavg 方法时结果不可解释。
NON_ORTHOGONAL_STRATEGIES = {"neurotoxin", "cerp", "badpfl"}
FEDAVG_LIKE = {"fedavg", "hierfedavg"}


def _fail(msg: str):
    raise ConfigError("\n[配置校验失败] " + msg + "\n")


def validate_config(config: dict, strict_orthogonality: bool = False) -> list:
    """
    校验 config 的跨轴兼容性。

    Args:
        strict_orthogonality: True 时把「攻击轴 × 方法轴不正交」也当成错误。
                              默认 False（只告警），因为正交化尚未完成，
                              置 True 可用于矩阵跑之前拦掉不可解释的格子。
    Returns:
        warnings: 非致命问题的文字列表（同时已打印）。
    Raises:
        ConfigError: 存在会导致「静默跑错」的组合。
    """
    warnings = []

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
              f"  该方法的 edge 聚合不经过 EdgeServerBase.robust_mean，防御会**静默失效**"
              f"（日志仍会打印 [Defense] enabled）。\n"
              f"  要么换方法，要么把该方法的 edge 聚合改走 robust_mean 并更新"
              f" config_validate.METHOD_SUPPORTS_DEFENSE。")

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

        # 陷阱 #1：攻击类替换 PFL 方法类
        if strategy in NON_ORTHOGONAL_STRATEGIES and method not in FEDAVG_LIKE:
            msg = (f"backdoor.malicious_strategy = {strategy!r} 与 "
                   f"drift_correction = {method!r} **尚未正交**。\n"
                   f"  {strategy} 的客户端类继承 FedAvgClient，且在 main.py 里"
                   f"**替换**掉方法类 → 恶意客户端跑朴素 FedAvg、良性客户端跑 {method}，\n"
                   f"  上传语义不一致，这一格的 ASR 数字不可解释（CLAUDE.md 陷阱 #1）。")
            if strict_orthogonality:
                _fail(msg)
            warnings.append(msg)

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
    return warnings
