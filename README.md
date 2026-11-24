# Reticus

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Network](https://img.shields.io/badge/network-100%25%20local-teal)
![Status](https://img.shields.io/badge/status-v0.1.0-orange)

Local CV pipeline visualization for engineers. Pip-installable. Zero external HTTP calls. No account required.

<!--
  Add a real screenshot or short GIF of the dashboard here before publishing -
  e.g. the Training or Results page with real charts. Something like:
  ![Reticus dashboard](docs/screenshot.png)
  This is the single highest-leverage thing for a GitHub visitor deciding
  whether to try the tool - do this before anything else in this file.
-->

Spins up a local web dashboard for:
- Dataset analysis (class distribution, duplicates, corrupted images)
- Preprocessing inspection (transform pipeline list, before/after sample gallery)
- Training monitoring (loss/accuracy curves, ETA, system metrics, Grad-CAM)
- Results inspection (per-class metrics, PR curves, prediction browser with pagination)
- Live inference (upload an image, get a prediction from your running model)
- Run comparison (overlaid epoch curves, config diff, class delta table)

## Supported tasks

| Task | Status |
|---|---|
| Image classification | ✅ |
| Object detection (YOLO / COCO / VOC / xyxy boxes) | ✅ |

## Reticus vs. TensorBoard / Weights & Biases / MLflow

Reticus isn't trying to replace these - it's a different tradeoff, built specifically
for CV work and for people who don't want their training data touching the network.

| | Reticus | TensorBoard | wandb | MLflow |
|---|---|---|---|---|
| Runs 100% locally, no account | ✅ | ✅ | ❌ (cloud by default) | ✅ (self-hostable) |
| Zero network calls, ever | ✅ | ✅ | ❌ | depends on setup |
| CV-specific (Grad-CAM, box overlays, dataset health checks) | ✅ | ❌ | partial (plugins) | ❌ |
| Rule-based training/data quality flags out of the box | ✅ | ❌ | ❌ | ❌ |
| Self-contained offline HTML report | ✅ | ❌ | ❌ | ❌ |
| Model registry / artifact versioning | ❌ (v1) | ❌ | ✅ | ✅ |
| Hyperparameter sweeps | ❌ (v1) | ❌ | ✅ | partial |
| Team / multi-user collaboration | ❌ (by design) | ❌ | ✅ | ✅ |

If you need a model registry, sweeps, or a shared team dashboard, those tools do that
well. If you want a fast, private, CV-aware dashboard that doesn't need an account or
a network connection, that's what Reticus is for.

## Install

A PyPI release is coming. Until then, clone the repo and install from source:

```bash
git clone https://github.com/shamanshetty/Reticus.git && cd Reticus
```

Core (logging + metrics only - tiny, just numpy + Pillow):
```bash
pip install .
```

With the local dashboard (FastAPI server):
```bash
pip install ".[dashboard]"
```

Everything (dashboard + torch Grad-CAM):
```bash
pip install ".[all]"
```

For development, add `-e` (for example `pip install -e ".[all]"`).

| Extra | Adds | Use when |
|---|---|---|
| *(none)* | numpy, Pillow | Headless training - log metrics, no dashboard |
| `dashboard` | fastapi, uvicorn, psutil | You want the local web dashboard |
| `torch` | torch, grad-cam | Grad-CAM heatmaps |
| `all` | all of the above | One-shot full install |

### CLI

After any run, serve the dashboard on existing data without writing a script:
```bash
reticus                  # auto-pick the only project
reticus my_project       # named project
reticus my_project --port 9000 --no-browser
```

`Reticus(project)` with no dashboard extra installed gives a clear install hint.
Use `Reticus(project, local=False)` to log without starting a server at all.

## Quickstart - Classification

```python
from reticus import Reticus

r = Reticus("my_project")  # opens http://127.0.0.1:8765 in browser

r.start_run(config={
    "task": "classification",
    "model": "resnet18",
    "optimizer": "adam",
    "lr": 1e-3,
    "class_names": ["cat", "dog", "bird"],
})

for epoch in range(1, 11):
    r.log({
        "epoch": epoch,
        "train_loss": 0.8 - epoch * 0.07,
        "val_loss": 0.85 - epoch * 0.065,
        "train_acc": 0.5 + epoch * 0.04,
        "val_acc": 0.48 + epoch * 0.038,
        "lr": 1e-3,
    }, total_epochs=10)

# Log predictions (pass images as PIL, numpy array, or torch.Tensor)
r.log_predictions(
    images=val_images,
    class_names=["cat", "dog", "bird"],
    task="classification",
    preds=pred_labels,
    targets=true_labels,
)

r.end_run()
```

## Quickstart - Object Detection

```python
r.start_run(config={
    "task": "od",
    "class_names": CLASSES,  # e.g. ["person", "car", "bike"]
})

r.log({"epoch": 1, "train_loss": 1.8, "val_loss": 1.9, "map50": 0.32})

# pred_boxes / true_boxes: list-of-images, each a list of boxes
# Classification box: [x1, y1, x2, y2, class_id, confidence]
# GT box:             [x1, y1, x2, y2, class_id]
r.log_predictions(
    images=val_images,
    class_names=CLASSES,
    task="od",
    pred_boxes=preds,
    true_boxes=trues,
    box_format="xyxy",  # or "yolo", "coco", "voc"
)

r.end_run()
```

## Data notes

- **Log plain Python numbers, not tensors.** Call `loss.item()` / `float(x)` before
  passing values to `r.log(...)` - Reticus stores JSON and won't serialize a raw tensor.
- **Images** passed to `log_predictions` / `log_gradcam` may be `PIL.Image`, a numpy
  array (HWC or CHW, 0-1 floats or 0-255 ints), a `torch.Tensor`, or `None` (a neutral
  placeholder is generated if you only care about labels, not pixels).
- **`class_names`** is an ordered list; a class id is its index in that list. Keep it
  consistent across the whole run.
- **Dataset layout** for `analyze_dataset`: classification expects a subdirectory per
  class (`train/cat/...`, `train/dog/...`); detection expects images plus YOLO-format `.txt`
  labels (flat, or `images/` + `labels/` side by side).

## Dataset Analysis

```python
r.analyze_dataset(
    data_dir="./data/train",
    class_names=["cat", "dog", "bird"],
    split_dirs={"train": "./data/train", "val": "./data/val"},
    task="classification",  # or "od"
)
```

## Grad-CAM

```python
r.log_gradcam(
    images=val_images,
    targets=true_labels,
    class_names=CLASSES,
    model=model,
    target_layer=model.layer4[-1],
)
```

## Inference Tab - upload an image, get a live prediction

Register any callable as a predictor and the dashboard's **Inference** tab lets
you drag-and-drop an image and see the model's prediction immediately - no
extra server, no separate script.

```python
r.register_predictor(my_predict_fn, class_names=["cat", "dog", "bird"])

# or, for a plain torch.nn.Module:
r.register_torch_classifier(model, class_names=CLASSES, device="cpu")
```

`my_predict_fn(pil_image)` returns either a 1-D sequence of class
probabilities (aligned to `class_names`) or a `{class_name: prob}` dict.
Works with any framework - PyTorch, TensorFlow, ONNX, scikit-learn - since
Reticus only calls the function you give it.

## HTML Report

```python
html_path = r.generate_report(run_id="run_1")
# or compare two runs:
html_path = r.generate_report(run_id="run_2", compare_with="run_1")
```

## API Reference

### `Reticus(project, local=True, port=8765, open_browser=True)`

| Param | Default | Description |
|---|---|---|
| `project` | required | Project name - data stored in `.reticus/projects/{name}/` |
| `local` | `True` | Bind server to `127.0.0.1` (never `0.0.0.0`) |
| `port` | `8765` | Dashboard port |
| `open_browser` | `True` | Auto-open browser on first start |

### Methods

| Method | Description |
|---|---|
| `start_run(config)` | Begin a new run, write `config.json` |
| `log(payload, total_epochs=None)` | Append epoch record, evaluate training flags |
| `log_predictions(images, class_names, task, ...)` | Save prediction thumbnails + JSONL |
| `log_gradcam(images, targets, class_names, model, target_layer)` | Compute + save Grad-CAM overlays |
| `end_run()` | Compute final metrics, merge flags, build prediction index |
| `analyze_dataset(data_dir, class_names, ...)` | Scan dataset, detect issues, store analysis |
| `log_preprocessing(steps, samples=None, processed_samples=None)` | Record transform pipeline + optional before/after gallery |
| `register_predictor(predict_fn, class_names=None)` | Register a callable for the live Inference tab |
| `register_torch_classifier(model, class_names, device="cpu")` | Convenience wrapper for a `torch.nn.Module` predictor |
| `save_model(model, class_names=None, scripted=True)` | Save TorchScript (or pickle fallback) model artifact |
| `generate_report(run_id, compare_with=None)` | Render standalone HTML report |

## Architecture

- **Storage**: all writes are atomic (temp-file rename). JSONL pagination uses a byte-offset index for O(chunk_size) seeks instead of O(N) line skipping.
- **Metrics**: real 11-point interpolated AP (PASCAL VOC method) at 10 IoU thresholds. Not `p x r`.
- **Server**: FastAPI on a background thread. Binds `127.0.0.1` only. Never blocks training.
  The thread is intentionally non-daemon, so the dashboard keeps running (and the
  Inference tab stays live) even after your training script's main work finishes -
  it exits when your script's process exits.
- **Frontend**: React 18 + Vite + shadcn/ui (Radix UI primitives). Pre-built into `reticus/dashboard/dist/` and served as static files.
- **No network egress**: zero external HTTP calls anywhere in the codebase.

## Privacy

Data never leaves the machine. The dashboard server binds to `127.0.0.1` only. No telemetry, no accounts, no external services.

## What Reticus deliberately doesn't do (v1)

These are scope decisions, not bugs or oversights:

- **No hyperparameter sweep / search.** Reticus visualizes runs you've already
  produced; it doesn't launch or schedule them. May reconsider for a future version.
- **No model registry or artifact versioning.** `save_model()` writes one artifact per
  run - there's no registry, no promotion workflow, no version graph.
- **No dataset artifact versioning.** `analyze_dataset()` reads whatever's on disk at
  the time; it doesn't snapshot or diff dataset versions over time.
- **No LLM-based suggestions.** The rule-based flags are deterministic threshold
  checks, not model-generated advice - may be worth exploring later, not in v1.
- **No multi-user / team collaboration - by design.** Reticus is a local, single-user
  tool. There's no shared server, no permissions model, no concept of "someone else's
  run."
- **No cloud or remote tracking - by design.** This is the core of Reticus's privacy
  positioning, not a missing feature. If you need a shared, remote dashboard, use
  wandb or MLflow's tracking server instead.

## Class names

Reticus does not auto-detect class names from folder structure, `classes.txt`,
`data.yaml`, or a dataloader's `.classes` attribute - you always pass `class_names`
explicitly to `start_run`, `log_predictions`, etc. This is deliberate: it keeps Reticus
correct regardless of which of the many dataset conventions your pipeline actually
uses, instead of guessing wrong on an untested one.

## License

MIT
