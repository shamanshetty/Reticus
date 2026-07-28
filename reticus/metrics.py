from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .od_utils import box_iou

# np.trapz was removed in numpy 2.0 in favor of np.trapezoid; np.trapezoid
# doesn't exist before numpy 2.0. Support both without pinning an upper bound.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _interpolated_ap(recalls: List[float], precisions: List[float]) -> float:
    """
    11-point interpolated AP (PASCAL VOC method).
    At each recall level t in {0.0, 0.1, ..., 1.0}, take the max precision
    for all recall values >= t. Average the 11 results.
    """
    if not recalls:
        return 0.0
    ap = 0.0
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        max_p = 0.0
        for r, p in zip(recalls, precisions):
            if r >= t and p > max_p:
                max_p = p
        ap += max_p
    return ap / 11.0


def _compute_class_ap(
    detections: List[Tuple[float, bool]],
    gt_count: int,
    iou_threshold: float = 0.5,
) -> Tuple[float, List[float], List[float]]:
    """
    Compute AP for one class at one IoU threshold.
    `detections`: list of (confidence, is_tp) sorted by confidence desc.
    Returns (ap, recall_points, precision_points).
    """
    if gt_count == 0 or not detections:
        return 0.0, [], []

    cumulative_tp = 0
    cumulative_fp = 0
    recall_pts: List[float] = []
    precision_pts: List[float] = []

    for _conf, is_tp in detections:
        if is_tp:
            cumulative_tp += 1
        else:
            cumulative_fp += 1
        precision = _safe_div(cumulative_tp, cumulative_tp + cumulative_fp)
        recall = _safe_div(cumulative_tp, gt_count)
        recall_pts.append(recall)
        precision_pts.append(precision)

    ap = _interpolated_ap(recall_pts, precision_pts)
    return ap, recall_pts, precision_pts


def _match_detections(
    pred_boxes_all: List[List[List[float]]],
    true_boxes_all: List[List[List[float]]],
    class_count: int,
    iou_threshold: float,
) -> Tuple[Dict[int, List[Tuple[float, bool]]], Dict[int, int]]:
    """
    Pass 1: for each image, greedily match predictions to GT at the given
    IoU threshold (confidence-descending order). Returns:
      per_class_dets: class_id -> [(confidence, is_tp), ...]
      gt_counts:      class_id -> total GT box count
    """
    per_class_dets: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
    gt_counts: Dict[int, int] = defaultdict(int)

    for image_preds, image_truths in zip(pred_boxes_all, true_boxes_all):
        # Count GT boxes per class for this image
        truths_by_class: Dict[int, List[List[float]]] = defaultdict(list)
        for truth in image_truths:
            if len(truth) < 5:
                continue
            cid = int(truth[4])
            truths_by_class[cid].append(truth)
            gt_counts[cid] += 1

        # Sort preds by confidence desc
        sorted_preds = sorted(
            image_preds,
            key=lambda b: float(b[5]) if len(b) > 5 else 1.0,
            reverse=True,
        )

        matched: set = set()  # (class_id, truth_index)
        for pred in sorted_preds:
            if len(pred) < 5:
                continue
            cid = int(pred[4])
            conf = float(pred[5]) if len(pred) > 5 else 1.0
            class_truths = truths_by_class.get(cid, [])

            best_iou = 0.0
            best_idx = -1
            for tidx, truth in enumerate(class_truths):
                if (cid, tidx) in matched:
                    continue
                iou = box_iou(pred, truth)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = tidx

            is_tp = best_iou >= iou_threshold and best_idx >= 0
            if is_tp:
                matched.add((cid, best_idx))
            per_class_dets[cid].append((conf, is_tp))

    return per_class_dets, gt_counts


def classification_metrics(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    class_names: Sequence[str],
    confidences: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics including confusion matrix, per-class
    precision/recall/F1, overall accuracy, and PR curves.

    If `confidences` (softmax scores, shape [N, num_classes]) are provided,
    PR curves are computed by sweeping confidence thresholds - otherwise a
    single operating-point curve is returned.
    """
    true_arr = np.asarray(list(true_labels), dtype=int)
    pred_arr = np.asarray(list(pred_labels), dtype=int)
    num_classes = len(class_names)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_arr.tolist(), pred_arr.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    # Row-normalise
    normalized = confusion.astype(float)
    for i in range(num_classes):
        row_sum = normalized[i].sum()
        if row_sum > 0:
            normalized[i] = normalized[i] / row_sum

    per_class = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []
    supports: List[int] = []

    for i, class_name in enumerate(class_names):
        tp = float(confusion[i, i])
        fp = float(confusion[:, i].sum() - tp)
        fn = float(confusion[i, :].sum() - tp)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * prec * rec, prec + rec)
        support = int(confusion[i, :].sum())
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        supports.append(support)
        per_class.append({
            "class_id": i,
            "class_name": class_name,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "support": support,
        })

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = _safe_div(correct, total)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    weighted_f1 = float(np.average(f1_scores, weights=supports)) if sum(supports) > 0 else 0.0
    micro_f1 = accuracy  # micro F1 = accuracy for single-label classification
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    # Balanced accuracy = mean per-class recall (robust to class imbalance)
    balanced_accuracy = macro_recall

    # Cohen's kappa and Matthews correlation coefficient (multiclass, from the CM)
    cm = confusion.astype(float)
    n = cm.sum()
    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    if n > 0:
        expected_agree = float((row_marg * col_marg).sum()) / (n * n)
        observed_agree = float(np.trace(cm)) / n
        cohen_kappa = _safe_div(observed_agree - expected_agree, 1.0 - expected_agree)
        cov = float(np.trace(cm)) * n - float((row_marg * col_marg).sum())
        denom = ((n * n - float((col_marg * col_marg).sum())) *
                 (n * n - float((row_marg * row_marg).sum()))) ** 0.5
        mcc = _safe_div(cov, denom)
    else:
        cohen_kappa = 0.0
        mcc = 0.0

    # PR curves - sweep thresholds if confidences provided
    pr_curves: List[Dict[str, Any]] = []
    if confidences is not None:
        conf_arr = np.asarray(list(confidences), dtype=float)
        for i, class_name in enumerate(class_names):
            binary_true = (true_arr == i).astype(int)
            scores = conf_arr[:, i] if conf_arr.ndim == 2 and conf_arr.shape[1] > i else None
            if scores is not None and len(scores) > 0:
                thresholds = np.linspace(0.0, 1.0, 50).tolist()
                pr_recall: List[float] = []
                pr_precision: List[float] = []
                for thr in thresholds:
                    predicted = (scores >= thr).astype(int)
                    tp = float(np.sum((predicted == 1) & (binary_true == 1)))
                    fp = float(np.sum((predicted == 1) & (binary_true == 0)))
                    fn = float(np.sum((predicted == 0) & (binary_true == 1)))
                    pr_precision.append(_safe_div(tp, tp + fp))
                    pr_recall.append(_safe_div(tp, tp + fn))
                pr_curves.append({
                    "class_name": class_name,
                    "precision": pr_precision,
                    "recall": pr_recall,
                })
            else:
                pr_curves.append({
                    "class_name": class_name,
                    "precision": [1.0, precisions[i], 0.0],
                    "recall": [0.0, recalls[i], recalls[i]],
                })
    else:
        for i, item in enumerate(per_class):
            pr_curves.append({
                "class_name": item["class_name"],
                "precision": [1.0, item["precision"], 0.0],
                "recall": [0.0, item["recall"], item["recall"]],
            })

    # ROC curves, calibration, top-k, confidence histograms (need probs)
    roc_curves: List[Dict[str, Any]] = []
    calibration: Dict[str, Any] = {}
    confidence_hist: Dict[str, Any] = {}
    top5_accuracy: Optional[float] = None
    if confidences is not None:
        conf_arr = np.asarray(list(confidences), dtype=float)
        if conf_arr.ndim == 2 and conf_arr.shape[0] == len(true_arr) and conf_arr.shape[1] >= 1:
            thr_grid = np.linspace(0.0, 1.0, 51)

            # Per-class one-vs-rest ROC
            for i, class_name in enumerate(class_names):
                if i >= conf_arr.shape[1]:
                    break
                binary_true = (true_arr == i).astype(int)
                scores = conf_arr[:, i]
                P = float(binary_true.sum())
                N = float((1 - binary_true).sum())
                tpr: List[float] = []
                fpr: List[float] = []
                for t in thr_grid:
                    pred_pos = (scores >= t).astype(int)
                    tp = float(np.sum((pred_pos == 1) & (binary_true == 1)))
                    fp = float(np.sum((pred_pos == 1) & (binary_true == 0)))
                    tpr.append(_safe_div(tp, P))
                    fpr.append(_safe_div(fp, N))
                # AUC via trapezoid over fpr-sorted points
                order = np.argsort(fpr)
                fpr_s = np.array(fpr)[order]
                tpr_s = np.array(tpr)[order]
                auc = float(_trapezoid(tpr_s, fpr_s))
                roc_curves.append({
                    "class_name": class_name,
                    "fpr": [round(v, 4) for v in fpr],
                    "tpr": [round(v, 4) for v in tpr],
                    "auc": round(auc, 4),
                })

            # Calibration / reliability: bin by predicted confidence (max prob)
            max_conf = conf_arr.max(axis=1)
            pred_top = conf_arr.argmax(axis=1)
            correct_top = (pred_top == true_arr).astype(int)
            n_bins = 10
            bin_conf: List[float] = []
            bin_acc: List[float] = []
            bin_count: List[int] = []
            ece = 0.0
            total_n = len(max_conf)
            for b in range(n_bins):
                lo = b / n_bins
                hi = (b + 1) / n_bins
                mask = (max_conf >= lo) & (max_conf < hi if b < n_bins - 1 else max_conf <= hi)
                cnt = int(mask.sum())
                if cnt > 0:
                    avg_conf = float(max_conf[mask].mean())
                    acc = float(correct_top[mask].mean())
                    ece += (cnt / total_n) * abs(acc - avg_conf)
                else:
                    avg_conf = (lo + hi) / 2
                    acc = 0.0
                bin_conf.append(round(avg_conf, 4))
                bin_acc.append(round(acc, 4))
                bin_count.append(cnt)
            calibration = {
                "bin_confidence": bin_conf,
                "bin_accuracy": bin_acc,
                "bin_count": bin_count,
                "ece": round(ece, 4),
            }

            # Confidence histogram split by correctness
            edges = np.linspace(0.0, 1.0, 21)
            hc, _ = np.histogram(max_conf[correct_top == 1], bins=edges)
            hw, _ = np.histogram(max_conf[correct_top == 0], bins=edges)
            confidence_hist = {
                "edges": [round(float(e), 3) for e in edges.tolist()],
                "correct": hc.tolist(),
                "wrong": hw.tolist(),
            }

            # Top-5 accuracy
            k = min(5, conf_arr.shape[1])
            topk_idx = np.argsort(-conf_arr, axis=1)[:, :k]
            hits = sum(1 for r in range(len(true_arr)) if true_arr[r] in topk_idx[r])
            top5_accuracy = round(hits / len(true_arr), 6)

    overall_extra: Dict[str, Any] = {}
    if top5_accuracy is not None:
        overall_extra["top5_accuracy"] = top5_accuracy
    if calibration:
        overall_extra["ece"] = calibration["ece"]

    return {
        "task": "classification",
        "class_names": list(class_names),
        "confusion_matrix_raw": confusion.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
        "per_class": per_class,
        "pr_curves": pr_curves,
        "roc_curves": roc_curves,
        "calibration": calibration,
        "confidence_hist": confidence_hist,
        "overall": {
            "accuracy": round(accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "micro_f1": round(micro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "balanced_accuracy": round(balanced_accuracy, 6),
            "cohen_kappa": round(cohen_kappa, 6),
            "mcc": round(mcc, 6),
            "n_samples": total,
            **overall_extra,
        },
    }


def detection_metrics(
    pred_boxes: Any,
    true_boxes: Any,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Real mAP implementation using 11-point interpolated AP.

    mAP@50: AP at IoU threshold 0.50
    mAP@50:95: mean AP across thresholds [0.50, 0.55, ..., 0.95]

    Algorithm:
      Pass 1 - per image, per class: sort preds by confidence desc, greedy
               match to GT (by IoU), record (confidence, is_tp) per detection.
      Pass 2 - per class: sort all detections by confidence desc, compute
               cumulative P/R, run 11-point interpolated AP.
      Repeat Pass 1+2 at each IoU threshold for mAP@50:95.
    """
    # Normalise inputs
    if hasattr(pred_boxes, "tolist"):
        pred_boxes = pred_boxes.tolist()
    if hasattr(true_boxes, "tolist"):
        true_boxes = true_boxes.tolist()
    pred_boxes = list(pred_boxes or [])
    true_boxes = list(true_boxes or [])
    class_names = list(class_names)
    class_count = len(class_names)

    # Ensure lengths match
    while len(pred_boxes) < len(true_boxes):
        pred_boxes.append([])
    while len(true_boxes) < len(pred_boxes):
        true_boxes.append([])

    # Normalise nested lists
    pred_boxes_norm: List[List[List[float]]] = []
    true_boxes_norm: List[List[List[float]]] = []
    for img_preds in pred_boxes:
        if hasattr(img_preds, "tolist"):
            img_preds = img_preds.tolist()
        pred_boxes_norm.append([[float(v) for v in b] for b in img_preds])
    for img_truths in true_boxes:
        if hasattr(img_truths, "tolist"):
            img_truths = img_truths.tolist()
        true_boxes_norm.append([[float(v) for v in b] for b in img_truths])

    # Compute at IoU=0.50 for per-class AP and PR curves
    dets_50, gt_counts = _match_detections(
        pred_boxes_norm, true_boxes_norm, class_count, iou_threshold=0.5
    )

    # Compute at 10 thresholds for mAP@50:95
    thresholds = [round(0.50 + i * 0.05, 2) for i in range(10)]
    per_class_ap95: Dict[int, List[float]] = defaultdict(list)
    for thr in thresholds:
        dets_thr, gt_thr = _match_detections(
            pred_boxes_norm, true_boxes_norm, class_count, iou_threshold=thr
        )
        for cid in range(class_count):
            dets_sorted = sorted(dets_thr.get(cid, []), key=lambda x: x[0], reverse=True)
            ap_thr, _, _ = _compute_class_ap(dets_sorted, gt_thr.get(cid, 0), thr)
            per_class_ap95[cid].append(ap_thr)

    per_class: List[Dict[str, Any]] = []
    ap50_values: List[float] = []
    ap95_values: List[float] = []
    pr_curves: List[Dict[str, Any]] = []

    conf_curves: List[Dict[str, Any]] = []
    tot_tp = tot_fp = tot_fn = 0

    for cid, class_name in enumerate(class_names):
        dets_sorted = sorted(dets_50.get(cid, []), key=lambda x: x[0], reverse=True)
        gt_total = gt_counts.get(cid, 0)
        ap50, recall_pts, precision_pts = _compute_class_ap(dets_sorted, gt_total, 0.5)

        # Per-class stats from operating point (highest-conf prediction as threshold)
        cumtp, cumfp = 0, 0
        for _conf, is_tp in dets_sorted:
            if is_tp:
                cumtp += 1
            else:
                cumfp += 1
        precision = _safe_div(cumtp, cumtp + cumfp)
        recall = _safe_div(cumtp, gt_total)
        total_fn = max(0, gt_total - cumtp)
        tot_tp += cumtp
        tot_fp += cumfp
        tot_fn += total_fn

        # F1-vs-confidence sweep -> optimal operating threshold
        thr_grid = [round(i * 0.05, 2) for i in range(21)]
        f1_curve: List[float] = []
        p_curve: List[float] = []
        r_curve: List[float] = []
        best_f1 = 0.0
        best_thr = 0.0
        for t in thr_grid:
            tp_t = sum(1 for c, is_tp in dets_sorted if c >= t and is_tp)
            fp_t = sum(1 for c, is_tp in dets_sorted if c >= t and not is_tp)
            p_t = _safe_div(tp_t, tp_t + fp_t)
            r_t = _safe_div(tp_t, gt_total)
            f1_t = _safe_div(2.0 * p_t * r_t, p_t + r_t)
            p_curve.append(round(p_t, 4))
            r_curve.append(round(r_t, 4))
            f1_curve.append(round(f1_t, 4))
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thr = t

        ap95 = float(np.mean(per_class_ap95[cid])) if per_class_ap95[cid] else 0.0
        ap50_values.append(ap50)
        ap95_values.append(ap95)

        per_class.append({
            "class_id": cid,
            "class_name": class_name,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "ap50": round(ap50, 6),
            "ap50_95": round(ap95, 6),
            "support": gt_total,
            "tp": cumtp,
            "fp": cumfp,
            "fn": total_fn,
            "best_f1": round(best_f1, 6),
            "best_conf_threshold": best_thr,
        })
        pr_curves.append({
            "class_name": class_name,
            "recall": recall_pts,
            "precision": precision_pts,
        })
        conf_curves.append({
            "class_name": class_name,
            "thresholds": thr_grid,
            "f1": f1_curve,
            "precision": p_curve,
            "recall": r_curve,
        })

    # Collect IoU values for histogram (best IoU per detection)
    iou_histogram: List[float] = []
    for img_preds, img_truths in zip(pred_boxes_norm, true_boxes_norm):
        truths_by_class: Dict[int, List[List[float]]] = defaultdict(list)
        for truth in img_truths:
            if len(truth) >= 5:
                truths_by_class[int(truth[4])].append(truth)
        for pred in img_preds:
            if len(pred) < 5:
                continue
            cid = int(pred[4])
            best = 0.0
            for truth in truths_by_class.get(cid, []):
                v = box_iou(pred, truth)
                if v > best:
                    best = v
            iou_histogram.append(round(best, 4))

    map50 = float(np.mean(ap50_values)) if ap50_values else 0.0
    map50_95 = float(np.mean(ap95_values)) if ap95_values else 0.0

    mean_iou = float(np.mean(iou_histogram)) if iou_histogram else 0.0
    overall_precision = _safe_div(tot_tp, tot_tp + tot_fp)
    overall_recall = _safe_div(tot_tp, tot_tp + tot_fn)

    return {
        "task": "od",
        "class_names": class_names,
        "per_class": per_class,
        "pr_curves": pr_curves,
        "conf_curves": conf_curves,
        "map50": round(map50, 6),
        "map50_95": round(map50_95, 6),
        "iou_histogram": iou_histogram,
        "overall": {
            "map50": round(map50, 6),
            "map50_95": round(map50_95, 6),
            "precision": round(overall_precision, 6),
            "recall": round(overall_recall, 6),
            "mean_iou": round(mean_iou, 6),
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
        },
    }
