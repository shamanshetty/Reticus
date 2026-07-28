from __future__ import annotations

import base64
import html
import json
import webbrowser
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .storage import ensure_dir, project_dir, read_json, read_jsonl_lines, run_dir

_VENDOR_DIR = Path(__file__).parent / "vendor"


@lru_cache(maxsize=1)
def _plotly_js() -> str:
    """Inline the vendored Plotly bundle so reports never call out to a CDN."""
    return (_VENDOR_DIR / "plotly.min.js").read_text(encoding="utf-8")


def _b64_image(path: Path) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        ext = path.suffix.lower().lstrip(".")
        mime = "png" if ext == "png" else "jpeg"
        return "data:image/%s;base64,%s" % (mime, data)
    except OSError:
        return None


def _f1_color(f1: float) -> str:
    if f1 >= 0.75:
        return "#22c55e"
    if f1 >= 0.5:
        return "#f59e0b"
    return "#ef4444"


def _render_per_class_table(per_class: List[Dict[str, Any]]) -> str:
    rows_html = ""
    for item in per_class:
        f1 = float(item.get("f1", item.get("ap50", 0.0)))
        color = _f1_color(f1)
        rows_html += (
            "<tr>"
            "<td>%(class_name)s</td>"
            "<td>%(precision).3f</td>"
            "<td>%(recall).3f</td>"
            "<td style='color:%(color)s;font-weight:600'>%(f1).3f</td>"
            "<td>%(support)s</td>"
            "</tr>"
        ) % {
            "class_name": html.escape(str(item.get("class_name", ""))),
            "precision": float(item.get("precision", 0)),
            "recall": float(item.get("recall", 0)),
            "f1": f1,
            "color": color,
            "support": html.escape(str(item.get("support", item.get("ap50", "")))),
        }
    return """
<table class="metrics-table">
<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
<tbody>%s</tbody>
</table>""" % rows_html


def _render_flags(flags: List[Dict[str, Any]]) -> str:
    if not flags:
        return "<p style='color:#6b7280'>No flags raised.</p>"
    severity_colors = {"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "INFO": "#6366f1"}
    items = ""
    for flag in flags:
        color = severity_colors.get(flag.get("severity", "INFO"), "#6b7280")
        items += (
            "<div class='flag-item' style='border-left:3px solid %(color)s'>"
            "<span class='flag-severity' style='color:%(color)s'>%(severity)s</span> "
            "<span class='flag-msg'>%(message)s</span>"
            "<p class='flag-hint'>%(suggestion)s</p>"
            "</div>"
        ) % {
            "color": color,
            "severity": html.escape(str(flag.get("severity", ""))),
            "message": html.escape(str(flag.get("message", ""))),
            "suggestion": html.escape(str(flag.get("suggestion", ""))),
        }
    return items


def generate_report(
    project: str,
    run_id: str,
    root: Path,
    compare_with: Optional[str] = None,
    open_browser: bool = False,
) -> Path:
    """
    Generate a self-contained HTML report for a completed run.
    All images embedded as base64. Full epoch history (no cap).
    """
    proj_dir = project_dir(project, root)
    rdir = run_dir(project, run_id, root)
    reports_dir = ensure_dir(proj_dir / "reports")
    out_path = reports_dir / ("%s_report.html" % run_id)

    config = read_json(rdir / "config.json", {})
    metrics = read_json(rdir / "metrics.json", {})
    flags = read_json(rdir / "flags.json", [])
    epochs = read_jsonl_lines(rdir / "epochs.jsonl")
    dataset = read_json(proj_dir / "dataset_analysis.json", {})

    # Embed up to 20 gradcam overlays
    gradcam_meta = read_json(rdir / "gradcam_meta.json", [])
    gradcam_gallery = ""
    for item in gradcam_meta[:20]:
        overlay_path = rdir / item.get("img_overlay", "")
        b64 = _b64_image(overlay_path)
        if b64:
            label = html.escape(str(item.get("class_name", "")))
            gradcam_gallery += (
                "<div class='img-card'>"
                "<img src='%s' alt='gradcam'>"
                "<p>%s</p>"
                "</div>"
            ) % (b64, label)

    overall = metrics.get("overall", {})
    task = metrics.get("task", config.get("task", "classification"))

    # Config table
    config_rows = "".join(
        "<tr><td>%s</td><td>%s</td></tr>" % (html.escape(str(k)), html.escape(str(v)))
        for k, v in sorted(config.items())
    )

    # Epoch chart data as JSON - escape "</" so a class/config value containing
    # "</script>" cannot break out of the inline <script> block.
    epochs_json = json.dumps(epochs).replace("</", "<\\/")
    metrics_json = json.dumps(metrics).replace("</", "<\\/")

    # Per-class table
    per_class_html = _render_per_class_table(metrics.get("per_class", []))

    # Flags HTML
    flags_html = _render_flags(flags)

    # Compare section
    compare_html = ""
    if compare_with:
        compare_dir = run_dir(project, compare_with, root)
        compare_metrics = read_json(compare_dir / "metrics.json", {})
        safe_compare_with = html.escape(str(compare_with))
        safe_run_id = html.escape(str(run_id))
        compare_html = "<h2>Comparison with %s</h2>" % safe_compare_with
        compare_html += "<p>Run %s macro F1: %.3f -> Run %s macro F1: %.3f</p>" % (
            safe_compare_with,
            float(compare_metrics.get("overall", {}).get("macro_f1", 0)),
            safe_run_id,
            float(overall.get("macro_f1", 0)),
        )

    safe_project = html.escape(str(project))
    safe_run_id = html.escape(str(run_id))
    safe_task = html.escape(str(task))

    report_html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reticus - %(project)s / %(run_id)s</title>
<script>%(plotly_js)s</script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a0a;color:#e4e4e7;font-family:'Inter',system-ui,sans-serif;font-size:14px;padding:24px}
h1{font-size:24px;font-weight:700;color:#fff;margin-bottom:4px}
h2{font-size:18px;font-weight:600;color:#a1a1aa;margin:32px 0 12px}
.section{background:#111;border:1px solid #222;border-radius:12px;padding:20px;margin-bottom:20px}
.card-row{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px}
.card{background:#111;border:1px solid #222;border-radius:10px;padding:14px 20px;min-width:140px}
.card-label{font-size:11px;text-transform:uppercase;letter-spacing:.15em;color:#6b7280}
.card-value{font-size:22px;font-weight:700;color:#6366f1;margin-top:4px}
.metrics-table{width:100%%;border-collapse:collapse;margin-top:12px}
.metrics-table th{text-align:left;padding:8px 12px;border-bottom:1px solid #222;color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:.1em}
.metrics-table td{padding:8px 12px;border-bottom:1px solid #1a1a1a;color:#d4d4d8}
.flag-item{padding:10px 14px;margin-bottom:8px;background:#0f0f0f;border-radius:6px}
.flag-severity{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em}
.flag-msg{font-weight:500;color:#e4e4e7}
.flag-hint{font-size:12px;color:#6b7280;margin-top:4px}
.img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-top:12px}
.img-card{text-align:center}
.img-card img{width:100%%;border-radius:6px;border:1px solid #222}
.img-card p{font-size:11px;color:#6b7280;margin-top:4px}
.config-table{width:100%%;border-collapse:collapse}
.config-table td{padding:6px 10px;border-bottom:1px solid #1a1a1a;color:#d4d4d8}
.config-table td:first-child{color:#6b7280;width:180px;font-size:12px}
.badge{display:inline-block;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600}
</style>
</head>
<body>
<h1>Reticus - %(project)s</h1>
<p style="color:#6b7280;margin-bottom:24px">Run: %(run_id)s &nbsp;-&nbsp; Task: %(task)s</p>

<div class="card-row">
  <div class="card"><div class="card-label">Accuracy / mAP</div><div class="card-value">%(accuracy)s</div></div>
  <div class="card"><div class="card-label">Macro F1</div><div class="card-value">%(macro_f1)s</div></div>
  <div class="card"><div class="card-label">Weighted F1</div><div class="card-value">%(weighted_f1)s</div></div>
  <div class="card"><div class="card-label">Epochs logged</div><div class="card-value">%(epoch_count)s</div></div>
</div>

<h2>Configuration</h2>
<div class="section">
<table class="config-table"><tbody>%(config_rows)s</tbody></table>
</div>

<h2>Training Curves</h2>
<div class="section" id="training-chart" style="height:360px"></div>

<h2>Results - Per-class Metrics</h2>
<div class="section">%(per_class_html)s</div>

<h2>Flags &amp; Suggestions</h2>
<div class="section">%(flags_html)s</div>

%(gradcam_section)s

%(compare_html)s

<script>
var epochs = %(epochs_json)s;
var metrics = %(metrics_json)s;
if (epochs.length) {
  var x = epochs.map(function(r){ return r.epoch; });
  Plotly.newPlot('training-chart', [
    {x:x, y:epochs.map(function(r){return r.train_loss;}), name:'train_loss', mode:'lines+markers', line:{color:'#818cf8'}},
    {x:x, y:epochs.map(function(r){return r.val_loss;}),   name:'val_loss',   mode:'lines+markers', line:{color:'#fb7185'}},
    {x:x, y:epochs.map(function(r){return r.train_acc||r.train_map;}), name:'train metric', mode:'lines+markers', line:{color:'#34d399'}},
    {x:x, y:epochs.map(function(r){return r.val_acc||r.val_map;}),     name:'val metric',   mode:'lines+markers', line:{color:'#f59e0b'}}
  ], {
    paper_bgcolor:'#111', plot_bgcolor:'#111', font:{color:'#e4e4e7'},
    margin:{l:42,r:12,t:20,b:40}, height:340,
    xaxis:{gridcolor:'#222'}, yaxis:{gridcolor:'#222'},
    legend:{bgcolor:'#111',bordercolor:'#222'}
  }, {displayModeBar:false,responsive:true});
}
</script>
</body>
</html>""" % {
        "project": safe_project,
        "run_id": safe_run_id,
        "task": safe_task,
        "plotly_js": _plotly_js(),
        "accuracy": "%.3f" % float(overall.get("accuracy", overall.get("map50", 0))),
        "macro_f1": "%.3f" % float(overall.get("macro_f1", 0)),
        "weighted_f1": "%.3f" % float(overall.get("weighted_f1", 0)),
        "epoch_count": len(epochs),
        "config_rows": config_rows,
        "per_class_html": per_class_html,
        "flags_html": flags_html,
        "epochs_json": epochs_json,
        "metrics_json": metrics_json,
        "gradcam_section": (
            "<h2>Grad-CAM Gallery</h2><div class='section'><div class='img-grid'>%s</div></div>" % gradcam_gallery
            if gradcam_gallery else ""
        ),
        "compare_html": compare_html,
    }

    out_path.write_text(report_html, encoding="utf-8")

    if open_browser:
        try:
            webbrowser.open(out_path.as_uri())
        except Exception:
            pass

    return out_path
