import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)


def generate_report(model=None, test_dataset=None,
                    all_labels=None, all_preds=None, all_probs=None,
                    class_names=None, save_path=None):
    """
    支持两种调用方式：
      1. 传 model + test_dataset → 内部推理
      2. 直接传 all_labels / all_preds / all_probs → 跳过推理
    """
    if all_labels is None:
        # 内部推理模式（GM / EM）
        all_labels, all_preds, all_probs = [], [], []
        for x, y in test_dataset:
            probs = model(x, training=False).numpy()
            all_probs.append(probs)
            all_preds.append(np.argmax(probs, axis=1))
            all_labels.append(y.numpy())
        all_labels = np.concatenate(all_labels)
        all_preds  = np.concatenate(all_preds)
        all_probs  = np.concatenate(all_probs)

    n_classes = all_probs.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    # 显式给出全部类别标签：非-IID 下某个 edge/client 的测试集可能只含部分类别，
    # 若不指定 labels，sklearn 会按实际出现的类别数推断，与长度为 n_classes 的
    # target_names 不匹配而报错（ValueError: Number of classes ... does not match
    # size of target_names）。缺席类别的 support 记为 0。
    labels_full = list(range(n_classes))

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(" Final Classification Report")
    lines.append("=" * 60)

    # ── 1. 分类报告（precision / recall / f1 / support）──────────
    lines.append("\n[ Per-Class Report ]")
    lines.append(classification_report(
        all_labels, all_preds,
        labels=labels_full,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    # ── 2. Macro / Weighted F1 ───────────────────────────────────
    f1_macro    = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    lines.append(f"[ F1 Score ]")
    lines.append(f"  Macro    F1 : {f1_macro:.4f}")
    lines.append(f"  Weighted F1 : {f1_weighted:.4f}")

    # ── 3. AUC（one-vs-rest）────────────────────────────────────
    try:
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
            lines.append(f"\n[ AUC ]")
            lines.append(f"  Binary AUC : {auc:.4f}")
        else:
            auc_macro = roc_auc_score(
                all_labels, all_probs,
                multi_class='ovr', average='macro', labels=labels_full
            )
            auc_weighted = roc_auc_score(
                all_labels, all_probs,
                multi_class='ovr', average='weighted', labels=labels_full
            )
            lines.append(f"\n[ AUC (OvR) ]")
            lines.append(f"  Macro    AUC : {auc_macro:.4f}")
            lines.append(f"  Weighted AUC : {auc_weighted:.4f}")
    except Exception as e:
        lines.append(f"\n[ AUC ] 计算失败: {e}")

    # ── 4. 混淆矩阵 ─────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=labels_full)
    lines.append(f"\n[ Confusion Matrix ] (shape: {cm.shape})")
    # 类别多时只打印对角线统计，不然太宽
    if n_classes <= 20:
        lines.append(str(cm))
    else:
        diag = np.diag(cm)
        lines.append(f"  Per-class correct: min={diag.min()} "
                     f"max={diag.max()} mean={diag.mean():.1f}")
        lines.append(f"  Overall accuracy : {diag.sum()/cm.sum():.4f}")

    lines.append("=" * 60)
    report_str = "\n".join(lines)
    print(report_str)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report_str)
            # 额外保存完整混淆矩阵为 npy
            np.save(save_path.replace(".txt", "_cm.npy"), cm)
            np.save(save_path.replace(".txt", "_probs.npy"), all_probs)
            np.save(save_path.replace(".txt", "_labels.npy"), all_labels)
        print(f"  Report saved to {save_path}")

    return {
        "f1_macro":    f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "all_probs":   all_probs,
        "all_labels":  all_labels,
    }