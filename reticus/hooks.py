from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .storage import read_jsonl_lines, write_jsonl_line

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def collect_system_metrics() -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if _PSUTIL:
        try:
            metrics["cpu_pct"] = psutil.cpu_percent(interval=None)
        except Exception:
            metrics["cpu_pct"] = None
        try:
            vm = psutil.virtual_memory()
            metrics["ram_gb"] = round(vm.used / (1024 ** 3), 2)
        except Exception:
            metrics["ram_gb"] = None
    else:
        metrics["cpu_pct"] = None
        metrics["ram_gb"] = None

    metrics["gpu_util"] = None
    metrics["gpu_mem_gb"] = None
    metrics["gpu_mem_reserved_gb"] = None

    if _TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            metrics["gpu_util"] = None  # torch has no GPU util API; leave for nvidia-ml
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            metrics["gpu_mem_gb"] = round(allocated / (1024 ** 3), 2)
            metrics["gpu_mem_reserved_gb"] = round(reserved / (1024 ** 3), 2)
        except Exception:
            pass

    return metrics


def build_epoch_record(
    payload: Dict[str, Any],
    existing_epochs: Optional[List[Dict[str, Any]]] = None,
    total_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build the full record for one epoch, adding system metrics, timestamp,
    and an ETA estimate (seconds remaining) if total_epochs is provided.
    """
    record: Dict[str, Any] = {}
    record.update(payload)
    record.update(collect_system_metrics())
    record["ts"] = int(time.time())

    if total_epochs is not None and existing_epochs:
        current_epoch = int(payload.get("epoch", len(existing_epochs) + 1))
        epochs_left = total_epochs - current_epoch
        # Use rolling 5-epoch window to estimate seconds-per-epoch
        window = existing_epochs[-5:]
        if len(window) >= 2:
            ts_list = [int(r.get("ts", 0)) for r in window if r.get("ts")]
            if len(ts_list) >= 2:
                elapsed = ts_list[-1] - ts_list[0]
                per_epoch = elapsed / max(1, len(ts_list) - 1)
                record["eta_seconds"] = max(0, int(per_epoch * epochs_left))
        if "eta_seconds" not in record and len(existing_epochs) >= 1:
            # Fallback: use wall clock from first epoch
            first_ts = int(existing_epochs[0].get("ts", record["ts"]))
            elapsed_total = record["ts"] - first_ts
            epochs_done = len(existing_epochs)
            if epochs_done > 0:
                per_epoch = elapsed_total / epochs_done
                record["eta_seconds"] = max(0, int(per_epoch * epochs_left))

    return record


def append_epoch(
    path: Path,
    payload: Dict[str, Any],
    total_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build an epoch record, append it to the JSONL file, and return it.
    Reads existing epochs only to compute ETA (last 5 lines).
    """
    existing: List[Dict[str, Any]] = []
    if total_epochs is not None:
        existing = read_jsonl_lines(path)

    record = build_epoch_record(payload, existing_epochs=existing, total_epochs=total_epochs)
    write_jsonl_line(path, record)
    return record
