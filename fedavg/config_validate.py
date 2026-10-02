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


def _compact_list(v) -> str:
    """把布点向量打成 `[10,0,0,0]`（无空格）—— 正则好写、一行放得下。

    `None` / 空 → `n/a`（**不是 `[]`**）：「本 run 不按 edge 布点」与
    「按 edge 布点但每个 edge 零个」是两件事，数值上不能长一样（铁律：无定义留空）。
    """
    if not v:
        return "n/a"
    return "[" + ",".join(str(int(x)) for x in v) + "]"


def _stop_round_str(bd: dict) -> str:
    """攻击退出轮 → `n/a`（从不停止 / 无攻击）或十进制轮号。

    与 `_compact_list` 同一条铁律：「从不停止」不能打成 `0` ——
    `0` 在语义上是「一轮都不投毒」，与「一直投到跑完」正好相反。
    """
    v = (bd or {}).get("attack_stop_round", None)
    return "n/a" if v is None else str(int(v))


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

        # ── 攻击时间窗（Neurotoxin 式持久性协议）──────────────────────────
        #   T = backdoor.attack_stop_round：恶意端在第 T 个 **cloud round** 及之后
        #   停止投毒，之后只观察衰减。三种写错都会静默变成「从不停止」，
        #   而日志与正常 run 长得一模一样 —— 全部在这里拦掉。
        stop_round = bd.get("attack_stop_round", None)
        if stop_round is not None:
            n_rounds = int(fed.get("n_rounds", 0) or 0)
            try:
                T = int(stop_round)
            except (TypeError, ValueError):
                T = None
                _fail(f"backdoor.attack_stop_round={stop_round!r} 不是整数。"
                      f"（null = 从不停止）")
            if T is not None and T <= 0:
                _fail(f"backdoor.attack_stop_round={T} ≤ 0 —— 恶意端一轮都不投毒，"
                      f"等价于无攻击对照。要做无攻击对照请用 malicious_per_edge=[0,...]，"
                      f"别用退出轮伪装。")
            if T is not None and n_rounds and T >= n_rounds:
                _fail(f"backdoor.attack_stop_round={T} ≥ federation.n_rounds={n_rounds}"
                      f" —— 攻击者到跑完都没退出，**静默等于「从不停止」**，"
                      f"而 metrics.json 会写着有退出轮。持久性格取 n_rounds//2。")
            # vanilla 的投毒是在 build_clients 里**静态**改数据集的，钩子拦不住它。
            if T is not None and strategy == "vanilla":
                _fail("backdoor.attack_stop_round 对 malicious_strategy='vanilla' 无效："
                      "vanilla 在 build_clients 阶段就把恶意端的数据集**静态**投毒了，"
                      "时间窗只作用于 on_round_start/on_batch/on_upload 钩子 → 会静默无效。"
                      "持久性实验请用 neurotoxin / cerp / badpfl。")
            if T is not None and bool(bd.get("forced_participation", False)):
                warnings.append(
                    f"attack_stop_round={T} 同时开着 forced_participation —— 恶意端在 T 之前"
                    f"每轮被强制选入、之后仍占着名额（补位循环只砍良性端）。"
                    f"衰减段的参与分布因此与主干格不同，跨格比较前先确认这是有意的。")

        # ── 投毒窗口起点 + 生成器语义（S4，D-078 / D-079）────────────────────
        #   S = backdoor.attack_start_round：第 S 个 cloud round 起投毒（含），窗口 [S, T)。
        #   写错同样都是静默的：S ≥ T 是空窗口（一轮都不投毒）、S ≥ n_rounds 是跑完都没开始，
        #   而日志与正常 run 一样 —— 在这里拦掉。
        from attack.attack_window import SCHEDULES as GEN_SCHEDULES
        start_round = bd.get("attack_start_round", None)
        S = None
        if start_round is not None:
            n_rounds = int(fed.get("n_rounds", 0) or 0)
            try:
                S = int(start_round)
            except (TypeError, ValueError):
                _fail(f"backdoor.attack_start_round={start_round!r} 不是整数。"
                      f"（null = 从第一轮起投毒）")
            if S is not None and S < 1:
                _fail(f"backdoor.attack_start_round={S} < 1 —— cloud round 从 1 起；"
                      f"从第一轮起投毒请写 null。")
            if S is not None and stop_round is not None and S >= int(stop_round):
                _fail(f"backdoor.attack_start_round={S} ≥ attack_stop_round={int(stop_round)}"
                      f" —— 投毒窗口 [start, stop) 是空的，恶意端一轮都不投毒，"
                      f"而 metrics.json 会写着有窗口。")
            if S is not None and n_rounds and S >= n_rounds:
                _fail(f"backdoor.attack_start_round={S} ≥ federation.n_rounds={n_rounds}"
                      f" —— 跑完都没开始投毒（或只投最后一轮）。")
            if S is not None and strategy == "vanilla":
                _fail("backdoor.attack_start_round 对 malicious_strategy='vanilla' 无效："
                      "vanilla 在 build_clients 阶段就把恶意端的数据集**静态**投毒了，"
                      "时间窗只作用于钩子 → 会静默无效。请用 neurotoxin / cerp / badpfl。")
            if S is not None and bool(bd.get("forced_participation", False)):
                warnings.append(
                    f"attack_start_round={S} 同时开着 forced_participation —— 窗口外恶意端"
                    f"仍被强制选入，参与分布与主干格不同，跨格比较前先确认这是有意的。")
        gen_sched = bd.get("generator_schedule", None)
        if gen_sched is not None:
            gs = str(gen_sched).lower()
            if gs not in GEN_SCHEDULES:
                _fail(f"backdoor.generator_schedule={gen_sched!r}：合法取值 {list(GEN_SCHEDULES)}"
                      f"（window = 生成器跟着投毒窗口走；always = 每轮都训）。")
            elif gs == "always" and strategy != "badpfl":
                _fail(f"backdoor.generator_schedule=always 只对 badpfl 有意义"
                      f"（malicious_strategy={strategy!r} 没有触发器生成器）→ 会静默无效。")
            elif gs == "always" and S is None and stop_round is None:
                warnings.append(
                    "generator_schedule=always 但没有投毒窗口（start / stop 都没写）——"
                    "全程都在投毒，与 window 完全等价。")

    # ── 3b. 自适应轮数（stopping）────────────────────────────────────────
    #   写错的三种方式都会静默改变实验长度，而日志与正常 run 一模一样：
    #   floor ≥ cap（永不延长/立刻停）、窗口比总点数还大（判据永远不满足）、
    #   θ 超出 [0,1]（永远越不过 → 每格都跑到 cap）。全部在这里拦掉。
    stop_cfg = config.get("stopping") or {}
    if stop_cfg:
        from server.stopping import VALID_CRITERIA
        er = int(fed.get("edge_rounds", 1) or 1)
        n_rounds = int(fed.get("n_rounds", 0) or 0)
        floor = int(stop_cfg.get("floor_effective", 0))
        cap = int(stop_cfg.get("cap_effective", n_rounds * er))
        bad = [c for c in stop_cfg.get("criteria", VALID_CRITERIA)
               if c not in VALID_CRITERIA]
        if bad:
            _fail(f"未知的 stopping.criteria：{bad}\n"
                  f"  合法取值：{sorted(VALID_CRITERIA)}")
        if floor >= cap:
            _fail(f"stopping.floor_effective={floor} ≥ cap_effective={cap} —— "
                  f"地板不低于上限，「按需延长」没有余地。")
        if n_rounds * er != cap:
            _fail(f"federation.n_rounds × edge_rounds = {n_rounds*er} 与 "
                  f"stopping.cap_effective = {cap} 不一致。\n"
                  f"  n_rounds 是循环上界，必须正好等于 cap ÷ edge_rounds，"
                  f"否则真正的上限是两者里小的那个，而 metrics.json 写着另一个。")
        if floor % er:
            warnings.append(
                f"stopping.floor_effective={floor} 不是 edge_rounds={er} 的整数倍，"
                f"实际地板会落在 {(floor + er - 1)//er * er} 个有效轮。")
        thetas = [float(t) for t in stop_cfg.get("thetas", (0.25, 0.5, 0.75))]
        if any(not (0.0 < t < 1.0) for t in thetas):
            _fail(f"stopping.thetas={thetas} 含超出 (0,1) 的值 —— "
                  f"越不过的阈值会让每格都跑满 cap。")
        W = int(stop_cfg.get("pm_window", 10))
        if W < 5:
            _fail(f"stopping.pm_window={W} < 5 —— 斜率的 SE 会大到判据形同虚设。")
        if float(stop_cfg.get("pm_slope_tol", 0.001)) <= 0:
            _fail("stopping.pm_slope_tol 必须 > 0。")
        # bd.eval_interval 在下面第 6 节才取，这里就地读一次（别用还没定义的名字 ——
        # 陷阱 #18 就是这么来的；守卫 tests/test_main_names_are_bound.py）。
        _ev = int(bd.get("eval_interval", 0) or 0) if bd_enabled else 0
        n_points = (cap // er) // _ev if _ev else 0
        # S5：开了统一网格，轻评估点也喂 pm_acc 平台判据 → 点数 = cap ÷ G
        _G = (config.get("evaluation") or {}).get("eval_grid", None)
        if isinstance(_G, int) and not isinstance(_G, bool) and _G >= 1:
            n_points = cap // _G
        if "pm_acc_plateau" in tuple(stop_cfg.get("criteria", VALID_CRITERIA)) \
                and n_points and n_points < W:
            warnings.append(
                f"本格最多 {n_points} 个评估点 < pm_window={W} —— "
                f"pm_acc 平台判据永远算不出来，该格会跑满 cap 并报 grid_too_coarse。")

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

    # ── 4b. 评估网格 ─────────────────────────────────────────────────────
    #   `backdoor.eval_interval`（backdoor_server.py:45，默认 **50**）与
    #   `evaluation.eval_interval`（server.py:258，默认 **10**）是**两个独立的键**，
    #   默认值还不一样。只写一个，ASR 与 PM 精度就落在不同的轮上 —— 于是
    #   metrics.json 的 `rounds[]`（ASR）与 `acc_rounds[]`（精度）无法逐点配对，
    #   而日志里没有任何异常。这不是致命错误（有意设成不同密度是合理的），
    #   所以只警告，不拦。
    ev_cfg = config.get("evaluation", {}) or {}
    bd_ev = bd.get("eval_interval", None) if bd_enabled else None
    acc_ev = ev_cfg.get("eval_interval", None)
    if bd_enabled and bd_ev is None:
        warnings.append(
            "backdoor.eval_interval 未显式设置，回退默认 50 —— 而 "
            "evaluation.eval_interval 的默认是 10。两者不同会让 ASR 与精度落在不同轮上。")
    if acc_ev is None:
        warnings.append(
            "evaluation.eval_interval 未显式设置，回退默认 10。建议显式写死。")
    if bd_ev is not None and acc_ev is not None and int(bd_ev) != int(acc_ev):
        warnings.append(
            f"backdoor.eval_interval={int(bd_ev)} != evaluation.eval_interval={int(acc_ev)}："
            f"ASR 与 PM 精度会落在不同的轮上，metrics.json 的 rounds[] 与 acc_rounds[] "
            f"无法逐点配对。若是有意的（ASR 评估更贵），忽略本条。")

    # ── 4b'. 统一评估网格（S5；规则在 server/eval_grid.py，D-055 / D-084）────────
    #   写错的每一种都会静默跑错：网格与 eval_interval 对不上 → 全量点不在网格上；
    #   顺序调度 → 没有「所有 edge 都跑完第 er 轮」的时刻；陈旧 PM → 轻评估测的不是主列。
    #   其余几条把网格收窄到 P2 的代码路径（G1 / G2 都在里面），不去证明别的路径也安全：
    #   legacy 管线用 Python random 洗牌、非共享生成器惰性创建、非固定攻击者的 ξ 退回持久的
    #   _atk_eval_rng —— 轻评估都可能碰到它们（D-084）。
    _grid = ev_cfg.get("eval_grid", None)
    if _grid is not None:
        if isinstance(_grid, bool) or not isinstance(_grid, int) or _grid < 1:
            _fail(f"evaluation.eval_grid 必须是 ≥ 1 的整数（有效轮），收到 {_grid!r}")
        # get_switch 在下面 §4c 才导入；这里先绑定，否则是「先用后绑」（陷阱 #18 同类）
        from alignment import get_switch
        from server import eval_grid as EG
        _R = int(fed.get("edge_rounds", 1) or 1)
        _need = EG.full_interval(_grid, _R)
        _period = EG.period_eff(_grid, _R)
        for _name, _sec, _dflt, _on in (("backdoor", bd, 50, bd_enabled),
                                        ("evaluation", ev_cfg, 10, True)):
            if not _on:
                continue
            _got = int(_sec.get("eval_interval", _dflt))
            if _got != _need:
                _fail(f"evaluation.eval_grid={_grid}、edge_rounds={_R} 要求 {_name}.eval_interval"
                      f" = lcm({_grid},{_R})/{_R} = {_need}，收到 {_got}：全量评估会落在网格外"
                      f"（D-055）。")
        _nr = int(fed.get("n_rounds", 0) or 0)
        if _nr and (_nr * _R) % _period:
            _fail(f"federation.n_rounds × edge_rounds = {_nr * _R} 不是 lcm({_grid},{_R}) = {_period}"
                  f" 的倍数：强制的末轮评估会落在网格外。")
        if _R > 1 and get_switch(config, "federation.edge_schedule") != "interleaved":
            _fail("evaluation.eval_grid 要求 federation.edge_schedule = interleaved：顺序调度下"
                  "没有「所有 edge 都跑完第 er 个 edge 轮」的时刻，轻评估点无从插入。")
        if get_switch(config, "evaluation.pm_model") != "fresh":
            _fail("evaluation.eval_grid 要求 evaluation.pm_model = fresh：轻评估只算主列"
                  "（fresh-PM），陈旧口径下它测的不是主列。")
        if get_switch(config, "data.batch_pipeline") != "per_epoch":
            _fail("evaluation.eval_grid 要求 data.batch_pipeline = per_epoch：legacy 管线训练时用"
                  " Python random 洗牌，轻评估若动到它就会改变训练（D-084，只开放 P2 路径）。")
        if defense in POST_HOC_DEFENSES:
            _fail(f"evaluation.eval_grid 与后处理防御 {defense!r} 不能同开：Simple-Tuning 会在每个"
                  f"轻评估点微调模型，且可能改到 edge 模型本身（未实现）。")
        if bd_enabled and str(bd.get("malicious_strategy", "vanilla")).lower() == "badpfl":
            if get_switch(config, "backdoor.eval_xi_model") != "fixed_attacker":
                _fail("evaluation.eval_grid + Bad-PFL 要求 backdoor.eval_xi_model = fixed_attacker："
                      "否则评估的 ξ 退回持久的 _atk_eval_rng，轻评估会消耗它 → 全量点的 ξ 随之错位，"
                      "开 / 关网格的全量数值不可比。")
            if not bool(bd.get("badpfl_shared_generator", False)):
                _fail("evaluation.eval_grid + Bad-PFL 要求 backdoor.badpfl_shared_generator = true："
                      "非共享生成器是惰性创建的，轻评估可能把创建时刻提前（D-084）。")
        _floor = int((config.get("stopping") or {}).get("floor_effective", 0) or 0)
        # period == R 时 §3b 的「floor 不是 edge_rounds 的整数倍」已经警告过，不重复
        if config.get("stopping") and _floor % _period and _period != _R:
            warnings.append(
                f"stopping.floor_effective={_floor} 不是 lcm({_grid},{_R}) = {_period} 的倍数："
                f"网格下只在全量点做停止决定，实际地板推迟到 "
                f"{(_floor + _period - 1) // _period * _period} 个有效轮。")
        _side = [k for k, on in (
            ("stale_asr_every", int(ev_cfg.get("stale_asr_every", 1) or 1) > 1),
            ("stale_pm_every", int(ev_cfg.get("stale_pm_every", 1) or 1) > 1),
            ("dump_logits_every", int(ev_cfg.get("dump_logits_every", 0) or 0) > 0),
            ("snapshot_rounds", ev_cfg.get("snapshot_rounds") is not None)) if on]
        if _side:
            warnings.append(
                f"evaluation.eval_grid 下 {', '.join(_side)} 只按全量点计数（间隔 = 每 "
                f"{_period} 个有效轮的倍数，仍随 edge_rounds 变）：副列与存盘不纳入网格（D-084）。")

    # ── 4g. S6a 记录开关（D-085）：update_geometry / post_agg_eval / frozen_trigger ──────
    #   都是只读记录，写错的每一种都会静默记错或静默不记：类型不对（1 == True）、没开网格
    #   （轻评估的围栏与草稿槽是网格那套）、顺序调度（没有「所有 edge 都跑完第 er 轮」的时刻）、
    #   方法不是 hier_fedrep（update_geometry 要 body 索引）、post_agg / frozen 要 Bad-PFL 固定攻击者。
    #   另：停止判据的容差是按 5 有效轮标定的（F-052）→ 开着停止判据时网格只能是 5。
    from alignment import get_switch as _gs
    _s6 = {k: ev_cfg.get(k) for k in ("update_geometry", "post_agg_eval", "frozen_trigger")
           if ev_cfg.get(k) is not None}
    for _k, _v in _s6.items():
        if not isinstance(_v, bool):
            _fail(f"evaluation.{_k} 必须是 bool，收到 {_v!r}")
    _dim = ev_cfg.get("update_sketch_dim", None)
    if _dim is not None and (isinstance(_dim, bool) or not isinstance(_dim, int) or _dim < 1):
        _fail(f"evaluation.update_sketch_dim 必须是 ≥ 1 的整数，收到 {_dim!r}")
    _s6_on = [k for k in ("update_geometry", "post_agg_eval", "frozen_trigger") if _s6.get(k)]
    if _s6_on:
        _names = ", ".join(f"evaluation.{k}" for k in _s6_on)
        if _gs(config, "evaluation.eval_grid") is None:
            _fail(f"{_names} 要求 evaluation.eval_grid 已开：记录开关复用网格那套轻评估的围栏"
                  f"（评估不得改变训练）。")
        if _gs(config, "federation.edge_schedule") != "interleaved":
            _fail(f"{_names} 要求 federation.edge_schedule = interleaved：逐 edge 轮的记录要在"
                  f"「所有 edge 都跑完第 er 个 edge 轮」的时刻做。")
    if _s6.get("update_geometry") and method != "hier_fedrep":
        _fail("evaluation.update_geometry 只对 hier_fedrep 实现（需要 body 索引）。")
    if (_s6.get("post_agg_eval") or _s6.get("frozen_trigger")):
        if not (bd_enabled and str(bd.get("malicious_strategy", "vanilla")).lower() == "badpfl"):
            _fail("evaluation.post_agg_eval / frozen_trigger 要求 backdoor.malicious_strategy = badpfl。")
        if _gs(config, "backdoor.eval_xi_model") != "fixed_attacker":
            _fail("evaluation.post_agg_eval / frozen_trigger 要求 backdoor.eval_xi_model = fixed_attacker。")
    _grid_now = _gs(config, "evaluation.eval_grid")
    if config.get("stopping") and _grid_now is not None and _grid_now != 5:
        _fail(f"stopping 与 evaluation.eval_grid={_grid_now} 不能同开：停止判据的斜率容差 0.0010 是按"
              f" 5 有效轮的网格标定的（F-052）；G=1 时等于放宽 5 倍，G>5 时更严。")

    # ── 4h. S6b 在线 c_k（D-087）：evaluation.update_ck(+ _every / _n / _steps) ─────────────
    #   只读记录，写错的每一种都会静默记错：类型不对（1 == True）、没有 edge 干净集（旧划分不切）、
    #   没开网格 / 不是交错调度 / 不是 hier_fedrep（评分挂在 edge 轮的上传收齐处，要 body 索引）、
    #   PGD 图片数超过干净集。
    _ck_ints = {k: ev_cfg.get(k) for k in ("update_ck_every", "update_ck_n", "update_ck_steps")
                if ev_cfg.get(k) is not None}
    for _k, _v in _ck_ints.items():
        if isinstance(_v, bool) or not isinstance(_v, int) or _v < 1:
            _fail(f"evaluation.{_k} 必须是 ≥ 1 的整数，收到 {_v!r}")
    _ck_on = ev_cfg.get("update_ck")
    if _ck_on is not None and not isinstance(_ck_on, bool):
        _fail(f"evaluation.update_ck 必须是 bool，收到 {_ck_on!r}")
    if _ck_on:
        if _gs(config, "evaluation.eval_grid") is None:
            _fail("evaluation.update_ck 要求 evaluation.eval_grid 已开（评分复用网格那套「评估不得改变训练」的围栏）。")
        if _gs(config, "federation.edge_schedule") != "interleaved":
            _fail("evaluation.update_ck 要求 federation.edge_schedule = interleaved。")
        if method != "hier_fedrep":
            _fail("evaluation.update_ck 只对 hier_fedrep 实现（需要 body 索引）。")
        if str(fed.get("partition", "")) not in ("designed", "hdir", "equal_random"):
            _fail("evaluation.update_ck 要求 S3 划分（federation.partition ∈ designed / hdir / equal_random）："
                  "只有它们会切出每个 edge 的干净集（D-064）；旧划分没有干净集可用。")
        _clean = int((fed.get("design") or {}).get("clean_per_edge", 0) or 0)
        _n_ck = int(_gs(config, "evaluation.update_ck_n"))
        if _clean < 1:
            _fail("evaluation.update_ck 要求 federation.design.clean_per_edge ≥ 1（edge 干净集）。")
        if _n_ck > _clean:
            _fail(f"evaluation.update_ck_n={_n_ck} 超过 federation.design.clean_per_edge={_clean}：PGD 的图片取自 edge 干净集。")

    # ── 4c. 对齐开关（A4；fedavg/alignment.py + config/alignment_p2.yaml）──
    #   三种写错都会静默跑错：取值拼错（回退旧行为）、P2 配置少开一项（与其他
    #   P2 格子不可比）、方法专属开关开在别的方法上（什么也不发生）。
    from alignment import (invalid_values, p2_mismatches, get_switch,
                           SWITCHES, EXTRA_SWITCHES, PM_FRESH_METHODS)
    bad_sw = invalid_values(config)
    if bad_sw:
        _fail("对齐开关取值不合法：\n" + "\n".join(
            f"  {k} = {v!r}，合法取值 {list(c)}" for k, v, c in bad_sw))
    protocol = str((config.get("meta") or {}).get("protocol", "")).upper()
    if protocol == "P2":
        mism = p2_mismatches(config)
        if mism:
            _fail("meta.protocol = P2，但以下对齐开关与模板 fedavg/config/alignment_p2.yaml"
                  " 不一致（P2 = 模板全开）：\n" + "\n".join(
                      f"  {k}: 模板 {want!r}，实际 {got!r}" for k, want, got in mism)
                  + "\n  消融请登记在 pilot 表（P1 口径），不要写成 P2。")
    _strategy = str(bd.get("malicious_strategy", "vanilla")).lower() if bd_enabled else None
    for sw in SWITCHES + EXTRA_SWITCHES:
        if sw.scope is None:
            continue
        val = get_switch(config, sw.key)
        if val == sw.legacy:
            continue
        if sw.scope == "hier_fedrep" and method != "hier_fedrep":
            warnings.append(f"{sw.key}={val!r} 只对 hier_fedrep 生效，"
                            f"当前方法 {method!r} 下无效（AUDIT {sw.row}）。")
        if sw.scope == "badpfl" and _strategy != "badpfl":
            warnings.append(f"{sw.key}={val!r} 只对 Bad-PFL 攻击生效，"
                            f"当前攻击 {_strategy!r} 下无效（AUDIT {sw.row}）。")
    if (get_switch(config, "data.batch_pipeline") == "per_epoch" and bd_enabled
            and _strategy != "badpfl"):
        _fail(f"data.batch_pipeline = 'per_epoch' 目前只接了 Bad-PFL 的取数"
              f"（malicious_strategy={_strategy!r}）：静态投毒（vanilla / neurotoxin）会被"
              f"per_epoch 绕过（它从原始数组取数，不读 build_poisoned_dataset 替换的数据集）"
              f" → 恶意端静默变成良性端；CerP 的触发器训练仍读 tf.data（AUDIT A25）。")
    if get_switch(config, "evaluation.pm_model") == "fresh" and method not in PM_FRESH_METHODS:
        _fail(f"evaluation.pm_model = 'fresh' 需要方法定义「私有部分」（client.private_state），"
              f"目前只有 {sorted(PM_FRESH_METHODS)}；{method!r} 没有定义 → "
              f"fresh-PM 无从组装（AUDIT A28 / D-033）。")

    # ── 4d. 评估降频（D-050 / D-054；开关在 alignment.EXTRA_SWITCHES）──────────
    #   类型显式查：Python 里 1 == True，choices 挡不住 `whitebox_asr: 1` 这类写法。
    #   两个 *_every > 1 时，陈旧 ASR 与陈旧 pm_acc 共用一个评估点序号（server._eval_seq）
    #   → 要求 ASR 与精度落在同一批评估点上，否则「隔点」隔的不是同一批点。
    _wb = ev_cfg.get("whitebox_asr", None)
    if _wb is not None and not isinstance(_wb, bool):
        _fail(f"evaluation.whitebox_asr 必须是 true / false，收到 {_wb!r}")
    _every = {}
    for _k in ("stale_asr_every", "stale_pm_every"):
        _v = ev_cfg.get(_k, None)
        if _v is not None and (isinstance(_v, bool) or not isinstance(_v, int) or _v < 1):
            _fail(f"evaluation.{_k} 必须是 ≥ 1 的整数，收到 {_v!r}")
        _every[_k] = 1 if _v is None else _v
    if max(_every.values()) > 1 and bd_enabled:
        _bd_i = int(bd.get("eval_interval", 50))
        _acc_i = int(ev_cfg.get("eval_interval", 10))
        if _bd_i != _acc_i:
            _fail(f"evaluation.stale_*_every > 1 要求 backdoor.eval_interval（{_bd_i}）"
                  f"== evaluation.eval_interval（{_acc_i}）：两个陈旧列共用一个评估点序号，"
                  f"网格不同就隔不到同一批点上（D-054）。")

    # ── 4d'. 两个存盘开关（S9 / D-073；utils/dumps.py）────────────────────────
    #   写错都会静默：dump_logits_every: true 在 Python 里等于 1；snapshot_rounds 写成
    #   YAML 列表会在 [设定4] 里变成字符串；超过 n_rounds 的轮号永远不会到。全部拦掉。
    _dl = ev_cfg.get("dump_logits_every", None)
    if _dl is not None and (isinstance(_dl, bool) or not isinstance(_dl, int) or _dl < 0):
        _fail(f"evaluation.dump_logits_every 必须是 ≥ 0 的整数（0 = 关），收到 {_dl!r}")
    _snap = ev_cfg.get("snapshot_rounds", None)
    if _snap is not None:
        from utils.dumps import MAX_SNAPSHOTS, estimate_snapshot_bytes, parse_rounds
        try:
            _rounds = parse_rounds(_snap)
        except ValueError as e:
            _fail(f"evaluation.{e}")
        _nr = int(fed.get("n_rounds", 0) or 0)
        if len(_rounds) > MAX_SNAPSHOTS:
            _fail(f"evaluation.snapshot_rounds={_snap!r}：每个 run 最多 {MAX_SNAPSHOTS} 个快照"
                  f"（D-073，磁盘是组内共享的）")
        if _nr and any(r > _nr for r in _rounds):
            _fail(f"evaluation.snapshot_rounds={_snap!r} 里有轮号 > federation.n_rounds={_nr}"
                  f" —— 那一轮永远不会到，快照静默缺失")
        if config.get("stopping"):
            warnings.append(
                f"evaluation.snapshot_rounds={_snap!r} 与自适应停轮同开：提前停下时，"
                f"停轮之后的快照轮会被静默跳过。固定长度的组请写 stopping: null。")
        if str((config.get("model") or {}).get("arch", "")).startswith("resnet10"):
            _est = estimate_snapshot_bytes(len(_rounds), int(fed.get("n_edges", 1) or 1) + 1,
                                           4_909_002, int(fed.get("n_clients", 0) or 0), 10_890)
            if _est > 1 << 30:
                warnings.append(f"evaluation.snapshot_rounds={_snap!r}：估计 {_est / 2**30:.1f} GiB"
                                f" / run（> 1 GiB），项目总预算 20 GB（D-073）")

    # ── 4e. 三层个性化（S8 / 3-E；utils/tier_split.py，D-057）──────────────
    #   四种写错都是「静默不生效或静默泄漏」：取值拼错（True == 1）、方法不是 FedRep
    #   （别的 edge server 广播照样整体覆盖 → edge 段每轮被冲掉，而 cloud 那边照样不聚合）、
    #   arch 没有按块命名的层、cloud 层防御（距离里带着本不上云的 edge 段坐标，陷阱 #9 同类）。
    from utils.tier_split import (EDGE_SHARED_KEY, TIER_ARCHS, TIER_METHODS,
                                  edge_shared_blocks)
    try:
        _k_edge = edge_shared_blocks(config)
    except ValueError as e:
        _fail(str(e))
    if _k_edge:
        if method not in TIER_METHODS:
            _fail(f"{EDGE_SHARED_KEY}={_k_edge} 只在 {sorted(TIER_METHODS)} 下实现"
                  f"（HierFedRepEdgeServer.set_weights 保留 edge 段）；{method!r} 的 edge server"
                  f" 会被 cloud 广播整体覆盖 → edge 段每轮被冲掉，三层个性化静默不生效。")
        _arch = str((config.get("model") or {}).get("arch", ""))
        if _arch not in TIER_ARCHS:
            _fail(f"{EDGE_SHARED_KEY}={_k_edge} 按残差块的层名（stage1_…stage4_）切分，"
                  f"只支持 {sorted(TIER_ARCHS)}；model.arch={_arch!r} 没有这套层名。")
        if defense in AGGREGATION_DEFENSES and "cloud" in cfg_layers:
            _fail(f"{EDGE_SHARED_KEY}={_k_edge} 与 cloud 层防御（defense.layers 含 cloud）不能同开："
                  f"edge 段不上云，但 cloud 的鲁棒聚合会拿它的坐标算距离（陷阱 #9 同类）。")
        if n_edges <= 1:
            warnings.append(f"{EDGE_SHARED_KEY}={_k_edge} 而 n_edges={n_edges}：只有一个 edge 时"
                            f"「edge 内共享」与「上云共享」等价，(a)/(b)/(c) 三臂没有区别。")

    # ── 4f. S3 新划分（data/designed_partition.py；D-062 … D-068）──────────────
    #   写错的后果都是「划分静默不是声明的那个」：比例表是 4 edge 的、特殊列是 class 0、
    #   等大小要求 n_per_client × test_ratio 为整数、需求不能超过每类 6000 张的供给（F-057）。
    from data.designed_partition import S3_PARTITIONS, config_errors
    _s3_errs = config_errors(config)
    if _s3_errs:
        _fail("S3 划分配置不合法（federation.partition="
              f"{fed.get('partition')!r}）：\n" + "\n".join(f"  {e}" for e in _s3_errs))
    if str(fed.get("partition", "")) not in S3_PARTITIONS and (fed.get("design") or {}):
        warnings.append(f"federation.design 只对 S3 划分 {S3_PARTITIONS} 生效；"
                        f"当前 partition={fed.get('partition')!r} 下被忽略。")

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

    # ── 6b. 布点与训练量自描述（[设定2]）──────────────────────────────────
    #   **为什么另起一行而不是扩 [设定]**：`RE_SETTINGS`（collect_metrics.py:110）
    #   是一条**全或无**的正则 —— 格式一旦对不上，client_fraction / poison_ratio /
    #   n_clients / n_edges / edge_rounds / n_malicious / forced_participation / arch
    #   **八个字段一起变成 None**，而日志毫无异常。扩它等于把八个已经能用的字段押上去。
    #   独立成行：老格式日志照常解析，新字段解析不出来就单独是 None。
    #
    #   记的是此前 metrics.json 里**完全没有**的四类量：
    #     - `malicious_per_edge` —— Experiment 3A 的**自变量本身**。没有它，
    #       「这份 json 是哪一格」只能靠文件名，而 metrics.json 必须自证。
    #     - `edge_assignment` —— 把 client_id 映射到 edge 的规则（block = 连续分块）。
    #       没有它，回程分析拿 malicious_participation_by_client 聚合到 edge 时
    #       只能靠「我记得配置是 block」。
    #     - `local_epochs` / `plocal_epochs` —— Stage B 标定之后会变，变了没法从文件区分。
    #     - 两个 eval_interval —— 决定评估网格的粗细（R40 最细只能到 40 有效轮）。
    #       下游要靠它判断「这一格能不能算到阈值的轮数」，粗到一定程度就该拒绝插值。
    train_cfg = config.get("training", {}) or {}
    print(f"[设定2] malicious_per_edge={_compact_list(bd.get('malicious_per_edge') if bd_enabled else None)} | "
          f"placement={str(bd.get('malicious_placement', 'spread')).lower() if bd_enabled else 'n/a'} | "
          f"edge_assignment={str(fed.get('edge_assignment', 'random')).lower()} | "
          f"local_epochs={int(train_cfg.get('local_epochs', 0) or 0)} | "
          f"plocal_epochs={int(train_cfg.get('plocal_epochs', 0) or 0)} | "
          f"seed={config.get('seed', 'n/a')} | "
          f"bd_eval_interval={int(bd_ev) if bd_ev is not None else 'n/a'} | "
          f"acc_eval_interval={int(acc_ev) if acc_ev is not None else 'n/a'} | "
          f"attack_stop_round={_stop_round_str(bd if bd_enabled else {})}")

    # ── 6c. 自适应轮数自描述（[设定3]）──────────────────────────────────
    #   **又是独立一行**，理由同 [设定2]：全或无的正则，往已有行里加字段一旦
    #   格式对不上，原有字段会一起变 None 而日志毫无异常。
    #   停在第 32 轮的 run 与跑满的 run 长得一模一样 —— 没有这一行事后无法判读。
    _sc = config.get("stopping") or {}
    print(f"[设定3] stopping={'on' if _sc else 'off'} | "
          f"floor_effective={_sc.get('floor_effective', 'n/a')} | "
          f"cap_effective={_sc.get('cap_effective', 'n/a')} | "
          f"criteria={','.join(_sc.get('criteria', [])) or 'n/a'} | "
          f"thetas={','.join(str(t) for t in _sc.get('thetas', [])) or 'n/a'} | "
          f"pm_window={_sc.get('pm_window', 'n/a')} | "
          f"pm_slope_tol={_sc.get('pm_slope_tol', 'n/a')}")

    # ── 6d. 对齐开关自描述（[设定4]）──────────────────────────────────
    #   又一条**独立**的行，而且按 key=value 逐字段解析（utils/kvline.py）：
    #   开关会随审计继续增加，全或无的正则扛不住。
    #   template=p2 / legacy / mixed：一眼看出这一格是 P2、旧协议还是消融。
    from alignment import describe_fields
    from utils.kvline import format_kv
    print(format_kv("[设定4]", describe_fields(config)))

    # ── 6e. 投毒窗口起点 + 生成器语义（[设定7]，S4）────────────────────────
    #   独立的 key=value 行，**不扩 [设定2]**（那条是全或无的正则，末尾的
    #   attack_stop_round 已经是可选组）。起点缺省 → n/a（= 从第一轮起），
    #   generator_schedule 打**有效值**（没写 → window），无攻击 → 两个都是 n/a。
    _sched = (str(bd.get("generator_schedule", None) or "window").lower()
              if bd_enabled else None)
    _start = bd.get("attack_start_round", None) if bd_enabled else None
    print(format_kv("[设定7]", {
        "attack_start_round": None if _start is None else int(_start),
        "generator_schedule": _sched,
    }))

    # ── 6f. 统一评估网格（[设定8]，S5）──────────────────────────────────
    #   独立的 key=value 行，不扩已有的 [设定*]。没开网格：eval_grid=n/a、slope_axis=cloud
    #   （停止判据横轴是云轮号）、gm_em=every_round —— 与 S5 之前的 run 是同一种 run。
    from server.eval_grid import describe as _grid_describe
    print(format_kv("[设定8]", _grid_describe(config)))
    # ── 6g. S6a 记录开关（[设定9]，D-085）────────────────────────────────
    #   独立 kv 行，不扩 [设定8]。没写 = 全关（与 S6a 之前的 run 是同一种 run）。
    print(format_kv("[设定9]", {
        "update_geometry": bool(_gs(config, "evaluation.update_geometry")),
        "update_sketch_dim": int(_gs(config, "evaluation.update_sketch_dim")),
        "post_agg_eval": bool(_gs(config, "evaluation.post_agg_eval")),
        "frozen_trigger": bool(_gs(config, "evaluation.frozen_trigger")),
    }))

    # ── 6h. S6b 在线 c_k（[设定10]，D-087）──────────────────────────────
    #   独立 kv 行，不扩 [设定9]。没写 = update_ck 关（与 S6b 之前的 run 是同一种 run）。
    print(format_kv("[设定10]", {
        "update_ck": bool(_gs(config, "evaluation.update_ck")),
        "update_ck_every": int(_gs(config, "evaluation.update_ck_every")),
        "update_ck_n": int(_gs(config, "evaluation.update_ck_n")),
        "update_ck_steps": int(_gs(config, "evaluation.update_ck_steps")),
    }))
    return warnings
