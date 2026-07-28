from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_dir(project: str, root: Path) -> Path:
    return ensure_dir(root / ".reticus" / "projects" / project)


def run_root(project: str, root: Path) -> Path:
    return ensure_dir(project_dir(project, root) / "runs")


def next_run_id(project: str, root: Path) -> str:
    runs_dir = run_root(project, root)
    existing: List[int] = []
    for child in runs_dir.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                existing.append(int(child.name[4:]))
            except ValueError:
                pass
    n = max(existing) + 1 if existing else 1
    return "run_%d" % n


def ensure_run_dir(
    project: str,
    root: Path,
    run_id: Optional[str] = None,
) -> Tuple[str, Path]:
    rid = run_id if run_id is not None else next_run_id(project, root)
    path = run_root(project, root) / rid
    ensure_dir(path)
    ensure_dir(path / "predictions")
    ensure_dir(path / "gradcam")
    ensure_dir(path / "artifacts")
    return rid, path


def run_dir(project: str, run_id: str, root: Path) -> Path:
    return root / ".reticus" / "projects" / project / "runs" / run_id


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        if default is None:
            return {}
        return default


def write_json(path: Path, payload: Any) -> None:
    """Atomic write via temp-file rename to prevent partial reads."""
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_jsonl_line(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def build_index(path: Path, chunk_size: int = 500) -> Dict[str, Any]:
    """
    Scan a JSONL file and record the byte offset at every chunk_size boundary.
    Returns {"total": int, "chunk_size": int, "offsets": [byte_offset_of_line_0, ...]}.
    offsets[k] is the byte offset of line (k * chunk_size).
    """
    offsets: List[int] = []
    total = 0
    try:
        with open(path, "rb") as fh:
            while True:
                if total % chunk_size == 0:
                    offsets.append(fh.tell())
                line = fh.readline()
                if not line:
                    break
                total += 1
        # Remove trailing sentinel if file ended exactly on boundary
        if offsets and total % chunk_size == 0 and len(offsets) > total // chunk_size:
            offsets = offsets[: total // chunk_size + 1]
    except FileNotFoundError:
        pass
    return {"total": total, "chunk_size": chunk_size, "offsets": offsets}


def read_jsonl_page(
    path: Path,
    start: int = 0,
    limit: int = 20,
    index: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Read `limit` lines starting at line `start` from a JSONL file.

    If `index` is provided with byte offsets, seeks to the nearest chunk
    boundary before `start` (O(chunk_size) forward scan instead of O(N)).
    """
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "rb") as fh:
            seek_offset = 0
            lines_to_skip = start

            if index is not None:
                offsets = index.get("offsets", [])
                chunk_size = index.get("chunk_size", 500)
                if offsets and chunk_size > 0 and start > 0:
                    chunk_idx = start // chunk_size
                    if chunk_idx < len(offsets):
                        seek_offset = offsets[chunk_idx]
                        lines_to_skip = start - chunk_idx * chunk_size

            fh.seek(seek_offset)
            for _ in range(lines_to_skip):
                line = fh.readline()
                if not line:
                    return rows

            for _ in range(limit):
                line = fh.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return rows


def stream_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Generator that yields one parsed dict per line. Never loads full file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def read_jsonl_lines(path: Path) -> List[Dict[str, Any]]:
    """Read entire JSONL file into a list. Use only for small files (epochs, flags)."""
    return list(stream_jsonl(path))


def count_jsonl_lines(path: Path) -> int:
    """Count lines without parsing JSON."""
    count = 0
    try:
        with open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    count += 1
    except FileNotFoundError:
        pass
    return count
