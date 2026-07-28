from __future__ import annotations

import shutil
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageDraw

from .data_analyzer import DataAnalyzer
from .gradcam import generate_gradcam
from .hooks import append_epoch
from .metrics import classification_metrics, detection_metrics
from .od_utils import normalize_boxes
from .reporter import generate_report as _gen_report
from .rules import evaluate_data_flags, evaluate_results_flags, evaluate_training_flags, merge_flags
from .storage import (
    build_index,
    count_jsonl_lines,
    ensure_dir,
    ensure_run_dir,
    project_dir,
    read_json,
    read_jsonl_lines,
    read_jsonl_page,
    run_dir,
    stream_jsonl,
    write_json,
    write_jsonl_line,
)

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Reticus:
    """
    Main entry point for the Reticus local CV visualization library.

    Usage::

        cv = Reticus(project="my-project")
        cv.analyze_dataset(...)
        cv.start_run(config={...})
        for epoch in range(epochs):
            cv.log({...}, total_epochs=epochs)
        cv.log_predictions(...)
        cv.log_gradcam(...)
        cv.end_run()
        cv.generate_report()
    """

    def __init__(
        self,
        project: str,
        local: bool = True,
        port: int = 8765,
        open_browser: bool = True,
    ) -> None:
        self.project = project
        self.port = port
        self.open_browser = open_browser
        self.root_dir = Path.cwd()
        self._proj_dir = project_dir(project, self.root_dir)
        self.current_run_id: Optional[str] = None
        self.current_task: str = "classification"
        self.current_class_names: List[str] = []
        self._server_thread: Optional[threading.Thread] = None
        self._app: Any = None

        if local:
            try:
                from .server import create_app  # lazy - web stack only loaded when serving
            except ImportError as exc:
                raise ImportError(
                    "Reticus dashboard requires the web stack (fastapi, uvicorn). "
                    "Install it with:  pip install \"reticus[dashboard]\"\n"
                    "Or run logging-only without the dashboard:  Reticus(project, local=False)"
                ) from exc

            self._app = create_app(self.root_dir, {"project": project})
            self._start_server()

    # Server

    def _start_server(self) -> None:
        from .server_manager import check_started, get_or_start_server

        get_or_start_server(self.project, self.port, self.root_dir, self._app)
        error = check_started(self.project, self.port)
        if error is not None:
            print(
                "\n[reticus] WARNING: dashboard failed to start on "
                "http://127.0.0.1:%d - %s\n"
                "  The port is likely already in use (e.g. another Reticus run, "
                "or a previous run that didn't shut down). Logging will still "
                "work, but there is no dashboard to view.\n"
                "  Fix: Reticus(%r, port=<a-free-port>)\n" % (self.port, error, self.project)
            )
            return

        if self.open_browser:
            try:
                webbrowser.open("http://127.0.0.1:%d" % self.port)
            except Exception:
                pass

    # Inference / model registration

    def register_predictor(
        self,
        predict_fn: Any,
        class_names: Optional[Sequence[str]] = None,
        _used_default_preprocess: bool = False,
        _model_name: Optional[str] = None,
        _source: str = "register_predictor",
    ) -> None:
        """
        Register a callable used by the dashboard's Inference tab.

        `predict_fn(pil_image)` must return either a 1-D sequence of class
        probabilities (aligned to class_names) or a dict {class_name: prob}.
        Works while this process is alive.

        Whatever `predict_fn` does - including any preprocessing it applies -
        is exactly what runs on an uploaded image. To match production
        behaviour, `predict_fn` must apply the same preprocessing your
        serving code will apply (resize, normalize, crop, etc.).
        """
        names = list(class_names) if class_names is not None else self.current_class_names
        if self._app is not None:
            self._app.state.reticus["predictor"] = predict_fn
            self._app.state.reticus["predictor_classes"] = names
        write_json(self._proj_dir / "model_info.json", {
            **read_json(self._proj_dir / "model_info.json", {}),
            "predictor": "live",
            "class_names": names,
            "used_default_preprocess": _used_default_preprocess,
            # plain functions/lambdas have __name__ (e.g. "demo_predictor");
            # only fall back to the class name for callable objects that don't.
            "model_name": _model_name or getattr(predict_fn, "__name__", None) or predict_fn.__class__.__name__,
            "source": _source,
            "registered_at": _now_iso(),
            "run_id": self.current_run_id,
        })

    def register_torch_classifier(
        self,
        model: Any,
        class_names: Sequence[str],
        preprocess: Optional[Any] = None,
        device: str = "cpu",
        input_size: int = 224,
    ) -> None:
        """
        Convenience wrapper: register a torch.nn.Module as a softmax classifier predictor.

        If `preprocess` is omitted, a generic resize->tensor->ImageNet-normalize
        pipeline is used as a fallback. That fallback will NOT match your
        production preprocessing unless your training pipeline also used
        plain ImageNet normalization at this input_size - pass your own
        `preprocess(pil_image) -> tensor` to guarantee dashboard predictions
        match what a deployed copy of this model would produce.
        """
        from .inference import build_torch_classifier_predictor

        fn = build_torch_classifier_predictor(model, preprocess=preprocess, device=device, input_size=input_size)
        self.register_predictor(
            fn, class_names=class_names, _used_default_preprocess=preprocess is None,
            _model_name=type(model).__name__, _source="register_torch_classifier",
        )

    def save_model(
        self,
        model: Any,
        class_names: Optional[Sequence[str]] = None,
        scripted: bool = True,
        example_input: Any = None,
    ) -> Path:
        """
        Save a model artifact for download / standalone inference.

        If `scripted` and torch is available, attempts TorchScript (self-contained,
        reloadable without the model class). Falls back to a full pickle save.
        Returns the saved path.
        """
        artifacts = ensure_dir(self._proj_dir / "model")
        names = list(class_names) if class_names is not None else self.current_class_names
        out = artifacts / "model.pt"
        fmt = "pickle"
        try:
            import torch

            scripted_ok = False
            if scripted:
                try:
                    if example_input is not None:
                        sm = torch.jit.trace(model.eval(), example_input)
                    else:
                        sm = torch.jit.script(model.eval())
                    sm.save(str(out))
                    fmt = "torchscript"
                    scripted_ok = True
                except Exception:
                    scripted_ok = False
            if not scripted_ok:
                torch.save(model, str(out))
                fmt = "pickle"
        except ImportError:
            raise ImportError("save_model requires torch. Install with: pip install \"reticus[torch]\"")

        write_json(self._proj_dir / "model_info.json", {
            **read_json(self._proj_dir / "model_info.json", {}),
            "saved": True,
            "format": fmt,
            "filename": out.name,
            "class_names": names,
        })
        # make the artifact reachable from the static mount
        return out

    # Internal helpers

    def _active_run_dir(self) -> Path:
        if self.current_run_id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        return run_dir(self.project, self.current_run_id, self.root_dir)

    def _image_to_pil(self, image: Any) -> Image.Image:
        if image is None:
            return Image.new("RGB", (64, 64), (30, 30, 30))
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        # torch.Tensor
        if hasattr(image, "detach"):
            image = image.detach().cpu()
        if hasattr(image, "numpy"):
            image = image.numpy()
        if _NP:
            arr = np.asarray(image, dtype="float32")
        else:
            arr = image
        if _NP and arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)
        if _NP and arr.ndim == 2:
            arr = arr[:, :, None]
            arr = arr.repeat(3, axis=2)
        if _NP:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.clip(0, 255).astype("uint8")
            return Image.fromarray(arr).convert("RGB")
        return Image.new("RGB", (64, 64), (30, 30, 30))

    def _save_classification_prediction(
        self,
        run_path: Path,
        idx: int,
        image: Any,
        true_label: int,
        pred_label: int,
        class_names: Sequence[str],
        conf: Optional[float] = None,
        probs: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        pil = self._image_to_pil(image)
        draw = ImageDraw.Draw(pil)
        true_name = class_names[true_label] if 0 <= true_label < len(class_names) else str(true_label)
        pred_name = class_names[pred_label] if 0 <= pred_label < len(class_names) else str(pred_label)
        label = "T: %s  |  P: %s" % (true_name, pred_name)
        bar_w = min(pil.width, 440)
        draw.rectangle([0, 0, bar_w, 28], fill=(0, 0, 0, 200))
        draw.text((8, 6), label, fill=(255, 255, 255))
        out = run_path / "predictions" / ("%d_%d_%d.png" % (idx, true_label, pred_label))
        pil.save(out)
        row: Dict[str, Any] = {
            "idx": idx,
            "img": str(Path("predictions") / out.name),
            "true": true_label,
            "pred": pred_label,
            "conf": round(conf if conf is not None else 1.0, 4),
            "correct": true_label == pred_label,
            "class_name": pred_name,
        }
        if probs is not None:
            row["probs"] = [round(float(p), 5) for p in probs]
        return row

    def _save_od_prediction(
        self,
        run_path: Path,
        idx: int,
        image: Any,
        true_boxes: Sequence[Sequence[float]],
        pred_boxes: Sequence[Sequence[float]],
        class_names: Sequence[str],
        box_format: str,
    ) -> Dict[str, Any]:
        from .od_utils import box_iou as _iou
        pil = self._image_to_pil(image)
        w, h = pil.size
        true_norm = normalize_boxes(true_boxes, box_format, w, h)
        pred_norm = normalize_boxes(pred_boxes, box_format, w, h)
        draw = ImageDraw.Draw(pil)

        def _rect(b: Sequence[float]) -> List[float]:
            # PIL requires x1>=x0, y1>=y0 - guard against inverted boxes
            x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

        for box in true_norm:
            cid = int(box[4]) if len(box) > 4 else 0
            draw.rectangle(_rect(box), outline=(0, 200, 0), width=2)
            if 0 <= cid < len(class_names):
                draw.text((box[0], max(0, box[1] - 14)), class_names[cid], fill=(0, 200, 0))

        for box in pred_norm:
            cid = int(box[4]) if len(box) > 4 else 0
            conf = float(box[5]) if len(box) > 5 else 1.0
            color = (255, 140, 0) if conf < 0.5 else (255, 50, 50)
            draw.rectangle(_rect(box), outline=color, width=2)
            if 0 <= cid < len(class_names):
                draw.text((box[0], box[1] + 2), "%s %.2f" % (class_names[cid], conf), fill=color)

        out = run_path / "predictions" / ("%d.png" % idx)
        pil.save(out)

        best_iou = 0.0
        if true_norm and pred_norm:
            best_iou = max(_iou(pred_norm[0], true_norm[0]), 0.0)

        return {
            "idx": idx,
            "img": str(Path("predictions") / out.name),
            "true_boxes": true_norm,
            "pred_boxes": pred_norm,
            "iou": round(best_iou, 4),
        }

    # Public API

    def log_preprocessing(
        self,
        steps: Sequence[Dict[str, Any]],
        samples: Optional[Sequence[Any]] = None,
        processed_samples: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record the preprocessing/augmentation pipeline for the Preprocessing tab.

        `steps` - ordered list of dicts describing each transform, e.g.::

            [
                {"name": "Resize", "params": {"size": 224}},
                {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
                {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406]}},
            ]

        `samples` / `processed_samples` - optional, equal-length lists of raw
        and corresponding post-transform images (any format accepted by
        `log_predictions`: PIL, numpy, torch.Tensor) to show as a before/after
        gallery. Pass only `samples` if you just want originals shown.
        """
        steps_list = [{"name": str(s.get("name", "")), "params": s.get("params", {})} for s in steps]

        gallery: List[Dict[str, Any]] = []
        if samples:
            samples_dir = ensure_dir(self._proj_dir / "preprocessing_samples")
            n = len(samples)
            for idx in range(n):
                raw_pil = self._image_to_pil(samples[idx])
                raw_path = samples_dir / ("%d_raw.png" % idx)
                raw_pil.save(raw_path)
                entry: Dict[str, Any] = {"idx": idx, "raw": str(Path("preprocessing_samples") / raw_path.name)}

                if processed_samples and idx < len(processed_samples):
                    proc_pil = self._image_to_pil(processed_samples[idx])
                    proc_path = samples_dir / ("%d_processed.png" % idx)
                    proc_pil.save(proc_path)
                    entry["processed"] = str(Path("preprocessing_samples") / proc_path.name)

                gallery.append(entry)

        payload = {"steps": steps_list, "samples": gallery}
        write_json(self._proj_dir / "preprocessing_pipeline.json", payload)
        return payload

    def analyze_dataset(
        self,
        data_dir: str,
        class_names: Sequence[str],
        split_dirs: Optional[Dict[str, str]] = None,
        task: str = "classification",
    ) -> Dict[str, Any]:
        from .hooks import collect_system_metrics

        # Write initial preprocessing status
        write_json(self._proj_dir / "preprocessing_status.json", {
            "status": "running",
            "phase": "preprocessing",
            "phase_progress": {"current": 0, "total": 0, "message": "Starting analysis..."},
        })

        # Progress callback for DataAnalyzer
        progress_path = self._proj_dir / "preprocessing_progress.jsonl"

        def _progress_cb(payload: Dict[str, Any]) -> None:
            write_jsonl_line(progress_path, {
                **payload,
                **collect_system_metrics(),
            })

        analyzer = DataAnalyzer()
        analysis = analyzer.analyze(
            data_dir=data_dir,
            class_names=class_names,
            split_dirs=split_dirs,
            task=task,
            progress_callback=_progress_cb,
        )

        # Copy sample images into .reticus/ so the static server can serve them
        sample_grid = analysis.get("sample_grid", {})
        if isinstance(sample_grid, dict):
            cached_grid: Dict[str, List[str]] = {}
            samples_dir = self._proj_dir / "samples"
            for class_name, paths in sample_grid.items():
                cached: List[str] = []
                for p in paths:
                    src = Path(p)
                    if not src.exists():
                        continue
                    dst = samples_dir / class_name / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    # Store path relative to .reticus/ root for /static/ serving
                    try:
                        rel = str(dst.relative_to(self.root_dir / ".reticus"))
                    except ValueError:
                        rel = str(dst)
                    cached.append(rel.replace("\\", "/"))
                cached_grid[class_name] = cached
            analysis["sample_grid"] = cached_grid

        write_json(self._proj_dir / "dataset_analysis.json", analysis)
        flags = evaluate_data_flags(analysis)
        write_json(self._proj_dir / "dataset_flags.json", flags)

        # Mark preprocessing complete
        write_json(self._proj_dir / "preprocessing_status.json", {
            "status": "completed",
            "phase": "preprocessing",
            "phase_progress": {"current": 1, "total": 1, "message": "Complete"},
        })

        return analysis

    def start_run(self, config: Dict[str, Any]) -> str:
        run_id, rpath = ensure_run_dir(self.project, self.root_dir)
        self.current_run_id = run_id
        self.current_task = config.get("task", "classification")
        self.current_class_names = list(config.get("class_names", []))

        write_json(rpath / "config.json", config)
        write_json(rpath / "status.json", {
            "status": "running",
            "phase": "training",
            "phase_progress": None,
            "display_name": run_id,
            "started_at": _now_iso(),
            "ended_at": None,
        })
        write_json(rpath / "metrics.json", {})
        write_json(rpath / "flags.json", [])
        write_json(rpath / "predictions_index.json", {"total": 0, "chunk_size": 500, "offsets": []})

        # Keep server aware of current project (only if a server is running)
        if self._app is not None:
            self._app.state.reticus["project"] = self.project

        return run_id

    def log(
        self,
        payload: Dict[str, Any],
        total_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Log one epoch. Appends to epochs.jsonl and re-evaluates training flags
        using only the last 10 rows (never reads the full file).
        """
        rpath = self._active_run_dir()
        epochs_path = rpath / "epochs.jsonl"

        record = append_epoch(epochs_path, payload, total_epochs=total_epochs)

        # Read only last 10 rows for flag evaluation
        total_lines = count_jsonl_lines(epochs_path)
        window_start = max(0, total_lines - 10)
        window = read_jsonl_page(epochs_path, window_start, 10)
        training_flags = evaluate_training_flags(window)

        existing_flags = read_json(rpath / "flags.json", [])
        write_json(rpath / "flags.json", merge_flags(existing_flags, training_flags))

        return record

    def log_predictions(
        self,
        images: Any,
        class_names: Sequence[str],
        task: str,
        preds: Optional[Sequence[int]] = None,
        targets: Optional[Sequence[int]] = None,
        probs: Optional[Sequence[Sequence[float]]] = None,
        pred_boxes: Optional[Sequence[Any]] = None,
        true_boxes: Optional[Sequence[Any]] = None,
        box_format: str = "xyxy",
    ) -> List[Dict[str, Any]]:
        from .hooks import collect_system_metrics

        rpath = self._active_run_dir()
        class_names = list(class_names)
        rows: List[Dict[str, Any]] = []

        # Set phase to evaluation on first call
        status = read_json(rpath / "status.json", {})
        if status.get("phase") != "evaluation":
            write_json(rpath / "status.json", {
                **status,
                "phase": "evaluation",
                "phase_progress": {"current": 0, "total": len(images), "message": "Logging predictions"},
            })

        eval_progress_path = rpath / "evaluation_progress.jsonl"
        total = len(images)

        if task == "classification":
            preds_list = list(preds or [])
            targets_list = list(targets or [])
            probs_list = list(probs) if probs is not None else None
            for idx, image in enumerate(images):
                row_probs = None
                if probs_list is not None and idx < len(probs_list):
                    row_probs = list(probs_list[idx])
                conf_val = max(row_probs) if row_probs else None
                row = self._save_classification_prediction(
                    rpath, idx, image,
                    int(targets_list[idx]), int(preds_list[idx]),
                    class_names, conf=conf_val, probs=row_probs,
                )
                write_jsonl_line(rpath / "predictions.jsonl", row)
                rows.append(row)

                # Progress every 100 predictions
                if idx % 100 == 0:
                    write_jsonl_line(eval_progress_path, {
                        "ts": int(time.time()),
                        "phase": "evaluation",
                        "stage": "predictions",
                        "predictions_logged": idx + 1,
                        "total_predictions": total,
                        **collect_system_metrics(),
                    })
        else:
            pb = list(pred_boxes or [])
            tb = list(true_boxes or [])
            for idx, image in enumerate(images):
                row = self._save_od_prediction(
                    rpath, idx, image,
                    tb[idx] if idx < len(tb) else [],
                    pb[idx] if idx < len(pb) else [],
                    class_names, box_format,
                )
                write_jsonl_line(rpath / "predictions.jsonl", row)
                rows.append(row)

                # Progress every 100 predictions
                if idx % 100 == 0:
                    write_jsonl_line(eval_progress_path, {
                        "ts": int(time.time()),
                        "phase": "evaluation",
                        "stage": "predictions",
                        "predictions_logged": idx + 1,
                        "total_predictions": total,
                        **collect_system_metrics(),
                    })

        index = build_index(rpath / "predictions.jsonl")
        write_json(rpath / "predictions_index.json", index)

        # Final progress update
        write_jsonl_line(eval_progress_path, {
            "ts": int(time.time()),
            "phase": "evaluation",
            "stage": "predictions",
            "predictions_logged": total,
            "total_predictions": total,
            **collect_system_metrics(),
        })

        return rows

    def log_gradcam(
        self,
        model: Any,
        images: Any,
        targets: Sequence[int],
        class_names: Sequence[str],
        target_layer: Any,
        method: str = "gradcam",
    ) -> List[Dict[str, Any]]:
        from .hooks import collect_system_metrics

        rpath = self._active_run_dir()
        eval_progress_path = rpath / "evaluation_progress.jsonl"
        total = len(targets)

        # Initial progress
        write_jsonl_line(eval_progress_path, {
            "ts": int(time.time()),
            "phase": "evaluation",
            "stage": "gradcam",
            "gradcam_generated": 0,
            "total_gradcam": total,
            **collect_system_metrics(),
        })

        rows = generate_gradcam(
            model=model,
            images=images,
            targets=targets,
            target_layer=target_layer,
            method=method,
            class_names=list(class_names),
            output_dir=rpath / "gradcam",
        )
        write_json(rpath / "gradcam_meta.json", rows)

        # Final progress
        write_jsonl_line(eval_progress_path, {
            "ts": int(time.time()),
            "phase": "evaluation",
            "stage": "gradcam",
            "gradcam_generated": len(rows),
            "total_gradcam": total,
            **collect_system_metrics(),
        })

        return rows

    def end_run(self) -> Dict[str, Any]:
        """
        Finalise the run: compute metrics, merge all flags, build prediction index.
        Uses stream_jsonl for predictions - never loads 1M rows into memory.
        """
        from .hooks import collect_system_metrics

        rpath = self._active_run_dir()
        eval_progress_path = rpath / "evaluation_progress.jsonl"

        # Log metrics computation stage
        write_jsonl_line(eval_progress_path, {
            "ts": int(time.time()),
            "phase": "evaluation",
            "stage": "metrics",
            **collect_system_metrics(),
        })

        config = read_json(rpath / "config.json", {})
        task = config.get("task", self.current_task)
        class_names = config.get("class_names", self.current_class_names) or self.current_class_names
        epochs = read_jsonl_lines(rpath / "epochs.jsonl")

        metrics: Dict[str, Any] = {"task": task, "class_names": class_names}

        if task == "classification":
            true_labels: List[int] = []
            pred_labels: List[int] = []
            prob_vectors: List[List[float]] = []
            have_probs = True
            for row in stream_jsonl(rpath / "predictions.jsonl"):
                true_labels.append(int(row.get("true", 0)))
                pred_labels.append(int(row.get("pred", 0)))
                pv = row.get("probs")
                if pv is None:
                    have_probs = False
                else:
                    prob_vectors.append([float(x) for x in pv])
            if true_labels:
                confidences = prob_vectors if (have_probs and prob_vectors) else None
                metrics.update(classification_metrics(
                    true_labels, pred_labels, class_names, confidences=confidences,
                ))
        else:
            pred_boxes_all: List[Any] = []
            true_boxes_all: List[Any] = []
            for row in stream_jsonl(rpath / "predictions.jsonl"):
                pred_boxes_all.append(row.get("pred_boxes", []))
                true_boxes_all.append(row.get("true_boxes", []))
            if pred_boxes_all:
                metrics.update(detection_metrics(pred_boxes_all, true_boxes_all, class_names))

        write_json(rpath / "metrics.json", metrics)
        build_index(rpath / "predictions.jsonl")

        result_flags = evaluate_results_flags(metrics)
        training_flags = evaluate_training_flags(epochs[-10:])
        data_flags = read_json(self._proj_dir / "dataset_flags.json", [])
        all_flags = merge_flags(data_flags, training_flags, result_flags)
        write_json(rpath / "flags.json", all_flags)
        prev_status = read_json(rpath / "status.json", {})
        write_json(rpath / "status.json", {
            **prev_status,
            "status": "completed",
            "phase": "completed",
            "phase_progress": None,
            "ended_at": _now_iso(),
        })

        return metrics

    def rename_run(self, run_id: str, display_name: str) -> None:
        """Set a friendly display name for a run, shown in the History tab."""
        rpath = run_dir(self.project, run_id, self.root_dir)
        status = read_json(rpath / "status.json", {})
        status["display_name"] = display_name
        write_json(rpath / "status.json", status)

    def generate_report(
        self,
        run_id: str = "latest",
        compare_with: Optional[str] = None,
    ) -> str:
        if run_id == "latest":
            runs_dir = self._proj_dir / "runs"
            latest: Optional[str] = None
            if runs_dir.exists():
                for child in sorted(runs_dir.iterdir()):
                    if child.is_dir() and child.name.startswith("run_"):
                        latest = child.name
            if latest is None:
                raise RuntimeError("No runs found for project '%s'." % self.project)
            run_id = latest

        path = _gen_report(
            self.project, run_id, self.root_dir,
            compare_with=compare_with,
            open_browser=True,
        )
        return str(path)
