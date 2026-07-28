from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncio

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .reporter import generate_report
from .storage import (
    count_jsonl_lines,
    ensure_dir,
    project_dir,
    read_json,
    read_jsonl_lines,
    read_jsonl_page,
    run_dir,
    stream_jsonl,
    write_json,
)


MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB - generous for a single inference image


def _project_name(state: Dict[str, Any]) -> str:
    return str(state.get("project", "reticus"))


def _load_runs(proj_root: Path) -> List[Dict[str, Any]]:
    runs_dir = proj_root / "runs"
    if not runs_dir.exists():
        return []
    runs: List[Dict[str, Any]] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("run_"):
            continue
        config = read_json(child / "config.json", {})
        metrics = read_json(child / "metrics.json", {})
        status_data = read_json(child / "status.json", {"status": "running"})
        runs.append({
            "run_id": child.name,
            "display_name": status_data.get("display_name", child.name),
            "status": status_data.get("status", "running"),
            "started_at": status_data.get("started_at"),
            "ended_at": status_data.get("ended_at"),
            "config": config,
            "metrics": metrics,
        })
    return runs


def _prefix_paths(rows: List[Dict[str, Any]], prefix: str, fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """
    Rewrite image-path fields from run/project-relative (e.g. "gradcam\\0.png",
    stored with OS-native separators) into paths relative to the /static mount
    root (e.g. "projects/p/runs/run_1/gradcam/0.png"), with forward slashes so
    they're valid URL paths on every OS.
    """
    out = []
    for row in rows:
        row = dict(row)
        for f in fields:
            v = row.get(f)
            if v:
                row[f] = "%s/%s" % (prefix, str(v).replace("\\", "/"))
        out.append(row)
    return out


def _matches_filter(
    row: Dict[str, Any],
    filter_class: Optional[int],
    filter_correct: Optional[bool],
    conf_min: float,
    conf_max: float,
) -> bool:
    is_od = "true_boxes" in row or "pred_boxes" in row

    if filter_class is not None:
        if is_od:
            classes_in_row = {int(b[4]) for b in row.get("true_boxes", []) if len(b) > 4} | {
                int(b[4]) for b in row.get("pred_boxes", []) if len(b) > 4
            }
            if filter_class not in classes_in_row:
                return False
        elif row.get("true") != filter_class and row.get("pred") != filter_class:
            return False

    # OD rows have no correct/conf fields, so use IoU: >= 0.5 counts as
    # "correct", and the conf range filter applies to IoU.
    if is_od:
        iou = float(row.get("iou", 0.0))
        if filter_correct is not None and (iou >= 0.5) != filter_correct:
            return False
        if iou < conf_min or iou > conf_max:
            return False
        return True

    if filter_correct is not None:
        if bool(row.get("correct", False)) != filter_correct:
            return False
    conf = float(row.get("conf", 1.0))
    if conf < conf_min or conf > conf_max:
        return False
    return True


def _compare_runs(proj_root: Path, run_a: str, run_b: str) -> Dict[str, Any]:
    dir_a = proj_root / "runs" / run_a
    dir_b = proj_root / "runs" / run_b
    for rdir, rid in ((dir_a, run_a), (dir_b, run_b)):
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % rid)

    config_a = read_json(dir_a / "config.json", {})
    config_b = read_json(dir_b / "config.json", {})
    metrics_a = read_json(dir_a / "metrics.json", {})
    metrics_b = read_json(dir_b / "metrics.json", {})
    epochs_a = read_jsonl_lines(dir_a / "epochs.jsonl")
    epochs_b = read_jsonl_lines(dir_b / "epochs.jsonl")

    # Config diff (only changed keys)
    all_keys = sorted(set(list(config_a.keys()) + list(config_b.keys())))
    config_diff = [
        {"key": k, "run_a": config_a.get(k), "run_b": config_b.get(k)}
        for k in all_keys
        if config_a.get(k) != config_b.get(k)
    ]

    # Per-class F1 delta
    pc_a = {item.get("class_name"): item for item in metrics_a.get("per_class", [])}
    pc_b = {item.get("class_name"): item for item in metrics_b.get("per_class", [])}
    class_names = sorted(set(list(pc_a.keys()) + list(pc_b.keys())))
    class_deltas = []
    for cname in class_names:
        f1_a = float(pc_a.get(cname, {}).get("f1", pc_a.get(cname, {}).get("ap50", 0.0)))
        f1_b = float(pc_b.get(cname, {}).get("f1", pc_b.get(cname, {}).get("ap50", 0.0)))
        class_deltas.append({
            "class_name": cname,
            "run_a": round(f1_a, 4),
            "run_b": round(f1_b, 4),
            "delta": round(f1_b - f1_a, 4),
        })

    overall_a = float(metrics_a.get("overall", {}).get("macro_f1", metrics_a.get("overall", {}).get("map50", 0.0)))
    overall_b = float(metrics_b.get("overall", {}).get("macro_f1", metrics_b.get("overall", {}).get("map50", 0.0)))
    delta = overall_b - overall_a
    summary = "Run %s vs %s: overall metric %+.3f." % (run_b, run_a, delta)
    if class_deltas:
        best = max(class_deltas, key=lambda x: abs(x["delta"]))
        summary += " %s: %.3f -> %.3f (%+.3f)." % (
            best["class_name"], best["run_a"], best["run_b"], best["delta"]
        )

    return {
        "run_a": run_a,
        "run_b": run_b,
        "config_diff": config_diff,
        "class_deltas": class_deltas,
        "summary": summary,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "epochs_a": epochs_a,
        "epochs_b": epochs_b,
    }


def create_app(root_dir: Path, state: Optional[Dict[str, Any]] = None) -> FastAPI:
    app = FastAPI(title="Reticus", docs_url=None, redoc_url=None)
    app.state.reticus = state or {}
    app.state.root_dir = Path(root_dir)

    # Static files from .reticus/ (images, gradcam, etc.)
    static_root = ensure_dir(Path(root_dir) / ".reticus")
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")

    # Frontend dist - try multiple candidate paths
    frontend_dist: Optional[Path] = None
    _pkg_dist = Path(__file__).parent / "dashboard" / "dist"
    for candidate in [
        _pkg_dist,
        Path(root_dir) / "reticus" / "dashboard" / "dist",
        Path(root_dir) / "frontend" / "dist",
    ]:
        if candidate.exists() and (candidate / "index.html").exists():
            frontend_dist = candidate
            break

    if frontend_dist is not None and (frontend_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    # Root

    @app.get("/")
    def index() -> HTMLResponse:
        if frontend_dist is not None:
            return FileResponse(str(frontend_dist / "index.html"))
        html = _shell_html()
        return HTMLResponse(html)

    # API routes

    @app.get("/api/project")
    def api_project() -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        proj_root = project_dir(name, app.state.root_dir)
        runs = _load_runs(proj_root)
        latest = runs[-1]["run_id"] if runs else None
        dataset = read_json(proj_root / "dataset_analysis.json", {})
        dataset_flags = read_json(proj_root / "dataset_flags.json", [])
        return {
            "project": name,
            "runs": runs,
            "latest_run": latest,
            "dataset": dataset,
            "dataset_flags": dataset_flags,
        }

    @app.get("/api/dataset")
    def api_dataset() -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        proj_root = project_dir(name, app.state.root_dir)
        return read_json(proj_root / "dataset_analysis.json", {})

    @app.get("/api/runs")
    def api_runs() -> List[Dict[str, Any]]:
        name = _project_name(app.state.reticus)
        proj_root = project_dir(name, app.state.root_dir)
        return _load_runs(proj_root)

    @app.get("/api/runs/{run_id}")
    def api_run_detail(run_id: str) -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        return {
            "run_id": run_id,
            "config": read_json(rdir / "config.json", {}),
            "status": read_json(rdir / "status.json", {"status": "running"}),
            "metrics": read_json(rdir / "metrics.json", {}),
            "flags": read_json(rdir / "flags.json", []),
        }

    @app.patch("/api/runs/{run_id}/rename")
    def api_run_rename(run_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        display_name = str(body.get("display_name", "")).strip()
        if not display_name:
            raise HTTPException(status_code=400, detail="display_name must be a non-empty string")
        if len(display_name) > 200:
            raise HTTPException(status_code=400, detail="display_name too long (max 200 chars)")
        status = read_json(rdir / "status.json", {"status": "running"})
        status["display_name"] = display_name
        write_json(rdir / "status.json", status)
        return {"run_id": run_id, "display_name": display_name}

    @app.get("/api/runs/{run_id}/epochs")
    def api_epochs(run_id: str) -> List[Dict[str, Any]]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        return read_jsonl_lines(rdir / "epochs.jsonl")

    @app.get("/api/runs/{run_id}/metrics")
    def api_metrics(run_id: str) -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        m = read_json(rdir / "metrics.json", None)
        if m is None:
            raise HTTPException(status_code=404, detail="metrics not yet available for '%s'" % run_id)
        return m

    @app.get("/api/runs/{run_id}/flags")
    def api_flags(run_id: str) -> List[Dict[str, Any]]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        return read_json(rdir / "flags.json", [])

    @app.get("/api/runs/{run_id}/predictions")
    def api_predictions(
        run_id: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=200),
        filter_class: Optional[int] = None,
        filter_correct: Optional[bool] = None,
        conf_min: float = Query(0.0, ge=0.0, le=1.0),
        conf_max: float = Query(1.0, ge=0.0, le=1.0),
    ) -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)

        preds_path = rdir / "predictions.jsonl"
        index = read_json(rdir / "predictions_index.json", None)
        no_filter = (
            filter_class is None
            and filter_correct is None
            and conf_min == 0.0
            and conf_max == 1.0
        )

        if no_filter and index is not None:
            # Fast path: O(chunk_size) byte-offset seek
            total = index.get("total", 0)
            start = (page - 1) * page_size
            items = read_jsonl_page(preds_path, start, page_size, index=index)
        else:
            # Filtered path: stream entire file with generator
            items = [
                row for row in stream_jsonl(preds_path)
                if _matches_filter(row, filter_class, filter_correct, conf_min, conf_max)
            ]
            total = len(items)
            start = (page - 1) * page_size
            items = items[start: start + page_size]

        total_pages = max(1, (total + page_size - 1) // page_size)
        prefix = "projects/%s/runs/%s" % (name, run_id)
        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "items": _prefix_paths(items, prefix, ("img",)),
        }

    @app.get("/api/runs/{run_id}/gradcam")
    def api_gradcam(run_id: str) -> List[Dict[str, Any]]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        rows = read_json(rdir / "gradcam_meta.json", [])
        prefix = "projects/%s/runs/%s" % (name, run_id)
        return _prefix_paths(rows, prefix, ("img_overlay", "img_original"))

    @app.get("/api/runs/{run_id}/pr_curves")
    def api_pr_curves(run_id: str) -> List[Dict[str, Any]]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        metrics = read_json(rdir / "metrics.json", {})
        return metrics.get("pr_curves", [])

    @app.get("/api/compare")
    def api_compare(
        run_a: str = Query(...),
        run_b: str = Query(...),
    ) -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        proj_root = project_dir(name, app.state.root_dir)
        return _compare_runs(proj_root, run_a, run_b)

    @app.get("/api/project/preprocessing")
    def api_preprocessing() -> List[Dict[str, Any]]:
        name = _project_name(app.state.reticus)
        proj_root = project_dir(name, app.state.root_dir)
        return read_jsonl_lines(proj_root / "preprocessing_progress.jsonl")

    @app.get("/api/project/preprocessing_pipeline")
    def api_preprocessing_pipeline() -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        proj_root = project_dir(name, app.state.root_dir)
        pipeline = read_json(proj_root / "preprocessing_pipeline.json", {"steps": [], "samples": []})
        prefix = "projects/%s" % name
        pipeline["samples"] = _prefix_paths(pipeline.get("samples", []), prefix, ("raw", "processed"))
        return pipeline

    @app.get("/api/runs/{run_id}/evaluation")
    def api_evaluation(run_id: str) -> List[Dict[str, Any]]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        return read_jsonl_lines(rdir / "evaluation_progress.jsonl")

    @app.get("/api/runs/{run_id}/errors")
    def api_errors(run_id: str, limit: int = Query(60, ge=1, le=300)) -> Dict[str, Any]:
        """Worst predictions: confident mistakes first (classification) or low-IoU (detection)."""
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        metrics = read_json(rdir / "metrics.json", {})
        task = metrics.get("task", "classification")
        rows: List[Dict[str, Any]] = []
        for row in stream_jsonl(rdir / "predictions.jsonl"):
            if task == "classification":
                if row.get("correct") is False:
                    rows.append({**row, "_sort": float(row.get("conf", 0.0))})
            else:
                iou = float(row.get("iou", 0.0))
                if iou < 0.5:
                    rows.append({**row, "_sort": 1.0 - iou})
        rows.sort(key=lambda r: r.get("_sort", 0.0), reverse=True)
        rows = rows[:limit]
        for r in rows:
            r.pop("_sort", None)
        prefix = "projects/%s/runs/%s" % (name, run_id)
        return {"task": task, "total_errors": len(rows), "items": _prefix_paths(rows, prefix, ("img",))}

    @app.get("/api/runs/{run_id}/threshold")
    def api_threshold(run_id: str) -> Dict[str, Any]:
        """Confidence-threshold sweep for classification: coverage + accuracy-on-covered."""
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        confs: List[float] = []
        corrects: List[bool] = []
        for row in stream_jsonl(rdir / "predictions.jsonl"):
            if "correct" not in row:
                continue
            confs.append(float(row.get("conf", 1.0)))
            corrects.append(bool(row.get("correct")))
        total = len(confs)
        thresholds = [round(i * 0.05, 2) for i in range(21)]
        coverage: List[float] = []
        accuracy: List[float] = []
        for t in thresholds:
            kept = [c for c, cf in zip(corrects, confs) if cf >= t]
            coverage.append(round(len(kept) / total, 4) if total else 0.0)
            accuracy.append(round(sum(kept) / len(kept), 4) if kept else 0.0)
        return {"total": total, "thresholds": thresholds, "coverage": coverage, "accuracy": accuracy}

    # Inference / model

    @app.get("/api/model/info")
    def api_model_info() -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        proj_root = project_dir(name, app.state.root_dir)
        info = read_json(proj_root / "model_info.json", {})
        has_live = callable(app.state.reticus.get("predictor"))
        model_file = proj_root / "model" / info.get("filename", "model.pt")
        return {
            "live_predictor": has_live,
            "downloadable": bool(info.get("saved")) and model_file.exists(),
            "format": info.get("format"),
            "class_names": app.state.reticus.get("predictor_classes") or info.get("class_names", []),
            "ready": has_live,
            "used_default_preprocess": bool(info.get("used_default_preprocess", False)),
            "model_name": info.get("model_name"),
            "source": info.get("source"),
            "registered_at": info.get("registered_at"),
            "registered_run_id": info.get("run_id"),
        }

    @app.post("/api/inference")
    async def api_inference(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=50)) -> Dict[str, Any]:
        predictor = app.state.reticus.get("predictor")
        if not callable(predictor):
            raise HTTPException(
                status_code=409,
                detail="No predictor registered. Call r.register_predictor(...) in your script while it is running.",
            )
        class_names = app.state.reticus.get("predictor_classes") or []
        from .inference import decode_image, run_predictor

        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image too large (max %d MB)." % (MAX_UPLOAD_BYTES // (1024 * 1024)))
        try:
            image = decode_image(data)
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
        try:
            result = run_predictor(predictor, image, class_names, top_k=top_k)
        except Exception as exc:  # predictor raised
            raise HTTPException(status_code=500, detail="Predictor failed: %s" % exc)
        result["width"], result["height"] = image.size
        return result

    @app.get("/api/model/download")
    def api_model_download() -> FileResponse:
        name = _project_name(app.state.reticus)
        proj_root = project_dir(name, app.state.root_dir)
        info = read_json(proj_root / "model_info.json", {})
        model_file = proj_root / "model" / info.get("filename", "model.pt")
        if not model_file.exists():
            raise HTTPException(status_code=404, detail="No saved model. Call r.save_model(model) first.")
        return FileResponse(str(model_file), filename="reticus_%s_model.pt" % name, media_type="application/octet-stream")

    @app.post("/api/runs/{run_id}/report")
    def api_report(run_id: str) -> Dict[str, Any]:
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        if not rdir.exists():
            raise HTTPException(status_code=404, detail="run '%s' not found" % run_id)
        path = generate_report(name, run_id, app.state.root_dir)
        return {"path": str(path)}

    # Live stream (WebSocket)

    @app.websocket("/ws/runs/{run_id}")
    async def ws_run(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        name = _project_name(app.state.reticus)
        rdir = run_dir(name, run_id, app.state.root_dir)
        last_epoch_count = -1
        last_status: Optional[str] = None
        try:
            while True:
                if rdir.exists():
                    epochs = read_jsonl_lines(rdir / "epochs.jsonl")
                    status = read_json(rdir / "status.json", {"status": "running"})
                    status_key = "%s:%s" % (status.get("status"), status.get("phase"))
                    if len(epochs) != last_epoch_count or status_key != last_status:
                        last_epoch_count = len(epochs)
                        last_status = status_key
                        await websocket.send_json({
                            "type": "update",
                            "epochs": epochs,
                            "status": status,
                            "flags": read_json(rdir / "flags.json", []),
                        })
                    if status.get("status") == "completed":
                        await websocket.send_json({"type": "done"})
                        await websocket.close()
                        return
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return
        except Exception:
            return

    # SPA catch-all - MUST be registered last so it does not shadow /api/*
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> HTMLResponse:
        if (
            full_path.startswith("api/")
            or full_path.startswith("static/")
            or full_path.startswith("assets/")
        ):
            raise HTTPException(status_code=404)
        if frontend_dist is not None:
            return FileResponse(str(frontend_dist / "index.html"))
        return HTMLResponse(_shell_html())

    return app


def _shell_html() -> str:
    return """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reticus</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0a0a0a;color:#e5e7eb;font-family:Inter,system-ui,sans-serif}
.shell{display:flex;min-height:100vh}
.sidebar{width:260px;background:#111;border-right:1px solid #222;padding:20px}
.main{flex:1;padding:24px}
.logo{font-size:11px;text-transform:uppercase;letter-spacing:.3em;color:#6366f1;margin-bottom:8px}
.muted{color:#6b7280;font-size:13px}
.card{background:#111;border:1px solid #222;border-radius:12px;padding:16px;margin-bottom:12px}
h1{font-size:20px;font-weight:600;color:#fff;margin-bottom:8px}
</style>
</head>
<body>
<div class="shell">
<aside class="sidebar">
<div class="logo">Reticus</div>
<p class="muted">Local CV visibility</p>
<div class="card" style="margin-top:16px">
<p class="muted" style="font-size:12px">Backend running. Build the frontend to see the dashboard.</p>
</div>
</aside>
<main class="main">
<div class="card">
<h1>Reticus is running</h1>
<p class="muted">Backend API is live at <code>http://127.0.0.1</code>. Run <code>npm run build</code> in the <code>frontend/</code> directory to serve the full dashboard.</p>
</div>
</main>
</div>
</body>
</html>"""
