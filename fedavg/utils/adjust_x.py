"""
将 wandb 中已有 runs 的 x 轴（step）整体平移，使所有曲线从 epoch=0 开始。

原因：原脚本第 1 个 cloud round 被 log 在 epoch = ER×LE 而非 0，
      导致不同实验起点不同。

修复方法：新 step = old_step - min_step（即减去每个 run 自己的第一个 step）

用法：
  pip install wandb pandas
  python shift_epochs.py
"""

import wandb
import pandas as pd

# ============================================================
# 🔧 配置区
# ============================================================
WANDB_ENTITY   = "3445339138-kth-royal-institute-of-technology"
SOURCE_PROJECT = "HierFL-cifar10"   # 当前有问题的 project
TARGET_PROJECT = "HierFL-cifar10_v2"  # 修复后的新 project
# ============================================================


def fix_run(run, target_project: str):
    history: pd.DataFrame = run.history(samples=10_000, pandas=True)
    if history.empty:
        print("  ⚠️  空 run，跳过")
        return

    # 只处理有 total_epoch 列的 run
    if "total_epoch" not in history.columns:
        print("  ⏭️  无 total_epoch 列，跳过")
        return

    # 用 total_epoch 作为新的 step，而不是 _step
    valid_steps = history["total_epoch"].dropna()
    if valid_steps.empty:
        print("  ⚠️  无有效 step，跳过")
        return
    offset = int(valid_steps.min())
    print(f"  offset = {offset}（所有 step 减去此值）")

    new_run = wandb.init(
        project=target_project,
        entity=WANDB_ENTITY,
        name=run.name,
        config=dict(run.config),
        tags=list(run.tags) + ["epoch_fixed"],
        reinit=True,
    )

    logged = 0
    for _, row in history.iterrows():
        old_step = row.get("total_epoch")
        if old_step is None or pd.isna(old_step):
            continue

        new_step = int(old_step) - offset  # 平移，第一个点变成 0

        metrics = {
            k: v
            for k, v in row.items()
            if not k.startswith("_") and pd.notna(v)
        }
        wandb.log(metrics, step=new_step)
        logged += 1

    wandb.finish()
    print(f"  ✅ 完成，log 了 {logged} 个点，x 轴范围：0 → {logged - 1} × step_size")


def main():
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{SOURCE_PROJECT}")
    total = len(runs)
    print(f"找到 {total} 个 runs\n")

    for i, run in enumerate(runs, 1):
        print(f"[{i}/{total}] {run.name}")
        fix_run(run, TARGET_PROJECT)
        print()

    print("全部完成！")


if __name__ == "__main__":
    main()