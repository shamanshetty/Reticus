# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - Initial release

- Local, self-hosted dashboard for computer vision training runs - classification and
  object detection.
- Dataset analysis: class distribution, image size stats, duplicate detection,
  corrupted-image detection, per-class sample grid, OD box counts.
- Live training monitor: loss/accuracy/mAP curves, learning rate, epoch progress + ETA,
  GPU/CPU/RAM usage, pushed over WebSocket with automatic polling fallback.
- Results: confusion matrix, per-class precision/recall/F1/support, macro/micro/weighted
  F1, PR curves, mAP@50/mAP@50-95 for detection.
- Prediction viewer with filtering (class, correct/wrong, confidence) and pagination.
- Grad-CAM, Grad-CAM++, and EigenCAM support for PyTorch classifiers.
- Run comparison: config diff, overlaid curves, per-class metric deltas, auto-summary.
- Rule-based quality flags for both dataset issues and training/results problems.
- Self-contained offline HTML report generation (no external JS, no network calls).
- Inference tab for trying an uploaded image against the currently registered model.
- Run history with renaming.
- CLI (`reticus <project>`) to re-serve an existing project's data without a script.
- Zero network egress, no telemetry, no accounts - all data stays in `.reticus/`
  under the working directory.
