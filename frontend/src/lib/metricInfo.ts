/** Plain-language explanations for metric labels, surfaced via hover tooltips. */
export const METRIC_INFO: Record<string, string> = {
  accuracy: 'Fraction of predictions that exactly match the true label.',
  macro_f1: 'Unweighted mean of each class’s F1 score — treats every class equally regardless of size.',
  micro_f1: 'F1 computed over all samples pooled together. For single-label classification this equals accuracy.',
  weighted_f1: 'Mean of per-class F1 weighted by class support (sample count).',
  precision: 'Of all predicted positives, the fraction that were correct. TP / (TP + FP).',
  recall: 'Of all actual positives, the fraction that were found. TP / (TP + FN).',
  f1: 'Harmonic mean of precision and recall.',
  ap50: 'Average Precision at IoU ≥ 0.50 (PASCAL VOC 11-point interpolation).',
  ap50_95: 'Mean AP averaged over IoU thresholds 0.50–0.95 in 0.05 steps (COCO-style).',
  map50: 'Mean AP@50 averaged across all classes.',
  map50_95: 'Mean AP@50:95 averaged across all classes — the strict COCO primary metric.',
  support: 'Number of ground-truth samples (or boxes) for this class.',
  lr: 'Current optimizer learning rate.',
  eta: 'Estimated time remaining, from a rolling 5-epoch average.',
  gpu_mem_gb: 'GPU memory currently allocated by the framework (GB).',
  ram_gb: 'System RAM in use (GB).',
  cpu_pct: 'Current CPU utilization (%).',
  duplicate_count: 'Exact-duplicate images detected by content hash.',
  corrupted_count: 'Images that were unreadable or smaller than 32×32 px.',
  balanced_accuracy: 'Mean of per-class recall — accuracy that is not dominated by majority classes.',
  macro_precision: 'Unweighted mean of per-class precision.',
  macro_recall: 'Unweighted mean of per-class recall.',
  cohen_kappa: 'Agreement vs chance (−1…1). >0.8 excellent, <0.4 weak. Robust to imbalance.',
  mcc: "Matthews correlation coefficient (−1…1). A balanced single-number quality score even for skewed classes.",
  mean_iou: 'Average best IoU across all predicted boxes.',
  tp: 'True positives — correct detections (IoU ≥ 0.5, right class).',
  fp: 'False positives — predicted boxes with no matching ground truth.',
  fn: 'False negatives — ground-truth boxes that were missed.',
  best_conf_threshold: 'Confidence cut-off that maximizes F1 for this class.',
  time_per_epoch: 'Median wall-clock seconds per epoch.',
  overfit_gap: 'Train metric minus validation metric — a widening gap signals overfitting.',
};

export function metricInfo(key: string): string | undefined {
  return METRIC_INFO[key];
}
