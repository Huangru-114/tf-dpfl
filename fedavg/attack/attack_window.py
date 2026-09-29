"""
fedavg/attack/attack_window.py  —  攻击时间窗与生成器语义的判定（S4，D-078 / D-079）

纯算术，**不 import TF**：本地 L1 秒级覆盖（tests/test_attack_window.py）。
`FLClientBase.attacking` / `generating` 只调这里，判定规则只定义一次。

轮号 = global(cloud) round，从 1 起（`CloudServer` 的主循环），与
`attack_stop_round` / `eval_interval` 同一把尺。

  投毒窗口   [start, stop)：start 为 None → 不设下限；stop 为 None → 从不停止。
             两者都为 None = 全程投毒（= 加 S4 之前的行为，逐字节一致）。
  生成器语义（`backdoor.generator_schedule`，只对 Bad-PFL 有意义）：
    window   生成器只在投毒窗口内训练，窗口外冻结（缺省；= G8 的语义）；
    always   窗口只管投毒，生成器每一轮都训（窗口外等于 ρ=0 的影子攻击者）。
"""

SCHEDULES = ("window", "always")
DEFAULT_SCHEDULE = "window"


def in_window(round_idx, start=None, stop=None) -> bool:
    """第 round_idx 个 cloud round 是否在投毒窗口 [start, stop) 内。"""
    r = int(round_idx)
    if start is not None and r < int(start):
        return False
    return stop is None or r < int(stop)


def generator_on(round_idx, start=None, stop=None, schedule=DEFAULT_SCHEDULE) -> bool:
    """这一轮（恶意端）要不要训练触发器生成器。"""
    s = DEFAULT_SCHEDULE if schedule is None else str(schedule).lower()
    if s == "always":
        return True
    if s == "window":
        return in_window(round_idx, start, stop)
    raise ValueError(f"backdoor.generator_schedule={schedule!r}：合法取值 {SCHEDULES}")
