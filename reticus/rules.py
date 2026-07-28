from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _flag(
    flag_id: str,
    severity: str,
    message: str,
    suggestion: str,
    stage: str,
) -> Dict[str, str]:
    return {
        "flag_id": flag_id,
        "severity": severity,
        "message": message,
        "suggestion": suggestion,
        "stage": stage,
    }


def evaluate_data_flags(dataset_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []

    distribution = dataset_analysis.get("class_distribution", {})
    total_counts: Dict[str, int] = {}
    for split_counts in distribution.values():
        for class_name, count in split_counts.items():
            total_counts[class_name] = total_counts.get(class_name, 0) + int(count)

    if total_counts:
        mean_count = sum(total_counts.values()) / float(len(total_counts))
        for class_name, count in sorted(total_counts.items()):
            if count < mean_count * 0.2:
                flags.append(_flag(
                    "CRITICAL_IMBALANCE", "CRITICAL",
                    "Class '%s' has only %d samples - far below the dataset mean." % (class_name, count),
                    "Collect more data or use oversampling for '%s'." % class_name,
                    "data",
                ))
            elif count < mean_count * 0.5:
                flags.append(_flag(
                    "IMBALANCE_WARNING", "WARNING",
                    "Class '%s' has %d samples - below half the dataset mean." % (class_name, count),
                    "Use a weighted sampler or augment minority class '%s'." % class_name,
                    "data",
                ))

    duplicate_count = int(dataset_analysis.get("duplicate_count", 0))
    if duplicate_count > 0:
        flags.append(_flag(
            "DUPLICATES_FOUND", "WARNING",
            "%d duplicate image pairs detected in the dataset." % duplicate_count,
            "Remove duplicate images before training to avoid data leakage.",
            "data",
        ))

    corrupted_count = int(dataset_analysis.get("corrupted_count", 0))
    if corrupted_count > 0:
        flags.append(_flag(
            "CORRUPTED_IMAGES", "WARNING",
            "%d images were unreadable or smaller than 32x32 pixels." % corrupted_count,
            "Inspect and remove corrupted files from the dataset.",
            "data",
        ))

    for class_name in dataset_analysis.get("od_low_box_classes", []):
        flags.append(_flag(
            "OD_LOW_BOX_COUNT", "WARNING",
            "OD class '%s' has fewer than 100 annotated bounding boxes." % class_name,
            "Collect more annotated examples for '%s'." % class_name,
            "data",
        ))

    return flags


def evaluate_training_flags(
    recent_epochs: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Evaluate training-stage flags from a window of recent epoch records.
    Caller should pass only the last 10 rows - not the full history.
    """
    flags: List[Dict[str, str]] = []
    if not recent_epochs:
        return flags

    acc_gaps: List[float] = []
    val_losses: List[float] = []
    loss_window: List[float] = []

    for row in recent_epochs:
        train_loss = row.get("train_loss")
        val_loss = row.get("val_loss")
        train_acc = row.get("train_acc")
        val_acc = row.get("val_acc")
        # Also support mAP for OD
        train_map = row.get("train_map")
        val_map = row.get("val_map")

        if val_loss is not None:
            val_losses.append(float(val_loss))
        if train_loss is not None and val_loss is not None:
            loss_window.append(float(val_loss))

        if train_acc is not None and val_acc is not None:
            acc_gaps.append(float(train_acc) - float(val_acc))
        elif train_map is not None and val_map is not None:
            acc_gaps.append(float(train_map) - float(val_map))

    # Overfitting detection (3+ consecutive epochs above gap threshold)
    if len(acc_gaps) >= 3 and all(g > 0.25 for g in acc_gaps[-3:]):
        flags.append(_flag(
            "SEVERE_OVERFITTING", "CRITICAL",
            "Train-val gap exceeds 25% for the last 3+ epochs.",
            "Model capacity is likely too high. Reduce layers/channels, add dropout or weight decay.",
            "training",
        ))
    elif len(acc_gaps) >= 3 and all(g > 0.15 for g in acc_gaps[-3:]):
        flags.append(_flag(
            "OVERFITTING", "WARNING",
            "Train-val gap exceeds 15% for the last 3+ epochs.",
            "Add dropout, weight decay, or stronger data augmentation.",
            "training",
        ))

    # Validation loss plateau (last 5 epochs, delta < 1e-4)
    if len(val_losses) >= 5:
        recent_5 = val_losses[-5:]
        delta = max(recent_5) - min(recent_5)

        if delta < 1e-4:
            flags.append(_flag(
                "VAL_LOSS_PLATEAU", "INFO",
                "Validation loss has not improved in 5 epochs (delta < 0.0001).",
                "Consider reducing the learning rate or enabling early stopping.",
                "training",
            ))
        elif all(recent_5[i] <= recent_5[i + 1] for i in range(len(recent_5) - 1)):
            flags.append(_flag(
                "VAL_LOSS_INCREASING", "WARNING",
                "Validation loss has been rising for 5 consecutive epochs.",
                "Possible overfitting - consider early stopping or stronger regularization.",
                "training",
            ))

        # Learning rate oscillation detection
        mean = sum(recent_5) / float(len(recent_5))
        variance = sum((v - mean) ** 2 for v in recent_5) / float(len(recent_5))
        std = variance ** 0.5
        if std > 0.3:
            flags.append(_flag(
                "LR_TOO_HIGH", "WARNING",
                "Loss standard deviation over last 5 epochs is %.3f (> 0.3) - oscillating." % std,
                "Reduce the learning rate to stabilize training.",
                "training",
            ))

    return flags


def evaluate_results_flags(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []
    task = metrics.get("task")
    per_class = metrics.get("per_class", [])

    if task == "classification":
        items: List[Tuple[str, Dict[str, Any]]] = []
        if isinstance(per_class, dict):
            items = list(per_class.items())
        else:
            items = [(item.get("class_name", "unknown"), item) for item in per_class]

        for class_name, item in items:
            f1 = float(item.get("f1", 0.0))
            recall = float(item.get("recall", 0.0))
            precision = float(item.get("precision", 0.0))

            if f1 < 0.5:
                flags.append(_flag(
                    "LOW_CLASS_F1", "WARNING",
                    "Class '%s' F1 is %.3f - below the 0.5 threshold." % (class_name, f1),
                    "Augment or collect more samples for class '%s'." % class_name,
                    "results",
                ))
            if recall < 0.4:
                flags.append(_flag(
                    "CRITICAL_CLASS_RECALL", "CRITICAL",
                    "Class '%s' recall is %.3f - model misses most of its instances." % (class_name, recall),
                    "Collect significantly more positive examples for '%s'." % class_name,
                    "results",
                ))
            if precision > 0.8 and recall < 0.5:
                flags.append(_flag(
                    "HIGH_PRECISION_LOW_RECALL", "WARNING",
                    "Class '%s' precision %.3f but recall %.3f - many missed positives." % (class_name, precision, recall),
                    "Lower the confidence threshold for class '%s' predictions." % class_name,
                    "results",
                ))

        # Find worst confusion pair
        confusion = metrics.get("confusion_matrix_normalized", [])
        class_names = metrics.get("class_names", [])
        if confusion and class_names:
            worst = 0.0
            worst_pair = None
            for i, row in enumerate(confusion):
                for j, val in enumerate(row):
                    if i == j:
                        continue
                    if val > worst:
                        worst = val
                        worst_pair = (class_names[i], class_names[j])
            if worst_pair is not None and worst >= 0.3:
                flags.append(_flag(
                    "CLASS_CONFUSION", "WARNING",
                    "%.0f%% of '%s' samples are predicted as '%s'." % (worst * 100, worst_pair[0], worst_pair[1]),
                    "Inspect hard examples and labels for the '%s' / '%s' pair." % worst_pair,
                    "results",
                ))

    elif task == "od":
        map50 = float(metrics.get("map50", metrics.get("overall", {}).get("map50", 0.0)))
        if map50 < 0.4:
            flags.append(_flag(
                "LOW_MAP", "WARNING",
                "mAP@50 is %.3f - below 0.4 for object detection." % map50,
                "Train for more epochs, tune anchor sizes, or collect more data.",
                "results",
            ))
        for item in per_class:
            cname = item.get("class_name", "unknown")
            recall = float(item.get("recall", 0.0))
            if recall < 0.4:
                flags.append(_flag(
                    "CRITICAL_CLASS_RECALL", "CRITICAL",
                    "OD class '%s' recall is %.3f." % (cname, recall),
                    "Add more annotated examples for '%s' or adjust IoU threshold." % cname,
                    "results",
                ))

    return flags


def merge_flags(*flag_lists: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge multiple flag lists, deduplicating by (flag_id, stage, message).
    Sorted CRITICAL -> WARNING -> INFO, then alphabetically by flag_id.
    """
    merged: List[Dict[str, str]] = []
    seen: set = set()

    for flag_list in flag_lists:
        for flag in flag_list:
            key = (flag.get("flag_id"), flag.get("stage"), flag.get("message"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(flag)

    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    merged.sort(key=lambda f: (severity_order.get(f.get("severity", "INFO"), 3), f.get("flag_id", "")))
    return merged
