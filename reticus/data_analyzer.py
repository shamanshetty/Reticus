from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

IMAGE_SUFFIXES: FrozenSet[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"})
YOLO_SUFFIXES: FrozenSet[str] = frozenset({".txt"})
MIN_IMAGE_SIZE: int = 32
SAMPLE_PER_CLASS: int = 6
MAX_DUPLICATE_PAIRS: int = 50


class DataAnalyzer:
    def __init__(self, seed: Optional[int] = None) -> None:
        effective_seed = seed if seed is not None else int(time.time()) % (2 ** 32)
        self._rng = random.Random(effective_seed)

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_SUFFIXES

    def _iter_images(self, root: Path) -> Iterable[Path]:
        if not root.exists():
            return
        for path in root.rglob("*"):
            if path.is_file() and self._is_image(path):
                yield path

    def _scan_image(self, path: Path) -> Tuple[Optional[Tuple[int, int]], bool]:
        """
        Returns (size, is_valid).
        Uses Image.open() without .load() - header-only, 10-100x faster than
        decoding the full pixel buffer just to read dimensions.
        """
        try:
            with Image.open(path) as img:
                w, h = img.size
            if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                return (w, h), False
            return (w, h), True
        except Exception:
            return None, False

    def _md5(self, path: Path) -> Optional[str]:
        try:
            h = hashlib.md5()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return None

    def _class_images(
        self,
        root: Path,
        class_names: Sequence[str],
    ) -> Dict[str, List[Path]]:
        """Find images per class assuming subdirectory-per-class layout."""
        result: Dict[str, List[Path]] = {name: [] for name in class_names}
        if not root.exists():
            return result
        for path in root.iterdir():
            if path.is_dir() and path.name in result:
                for img in path.iterdir():
                    if img.is_file() and self._is_image(img):
                        result[path.name].append(img)
        return result

    def _find_image_for_label(self, txt_path: Path) -> Optional[Path]:
        """
        Resolve a YOLO .txt label file to its image, supporting the two
        dominant real-world layouts: image + .txt side by side in the same
        directory, or sibling images/ + labels/ directories.
        """
        stem = txt_path.stem
        candidates = [txt_path.with_suffix(ext) for ext in IMAGE_SUFFIXES]
        if txt_path.parent.name == "labels":
            img_dir = txt_path.parent.parent / "images"
            candidates += [img_dir / (stem + ext) for ext in IMAGE_SUFFIXES]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _od_scan(
        self,
        data_dir: Path,
        class_names: Sequence[str],
    ) -> Tuple[Dict[str, int], List[str], int, Dict[str, List[Path]]]:
        """
        Scan YOLO-format .txt label files under data_dir. Returns
        (box_counts, low_box_classes, degenerate_count, sample_grid).

        Real OD datasets are typically flat (images + .txt labels, or
        images/ + labels/ sibling dirs) - not subdirectory-per-class like
        classification - so this walks labels directly rather than assuming
        a class-named folder structure.
        """
        box_counts: Dict[str, int] = {name: 0 for name in class_names}
        sample_grid: Dict[str, List[Path]] = {name: [] for name in class_names}
        degenerate = 0
        class_count = len(class_names)

        for txt_path in data_dir.rglob("*.txt"):
            try:
                with open(txt_path, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
            except OSError:
                continue

            classes_in_image: set = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                if 0 <= cid < class_count:
                    name = class_names[cid]
                    box_counts[name] += 1
                    classes_in_image.add(name)
                # Check for degenerate boxes (width or height <= 0)
                try:
                    w, h = float(parts[3]), float(parts[4])
                    if w <= 0.0 or h <= 0.0:
                        degenerate += 1
                except ValueError:
                    pass

            if classes_in_image:
                img_path = self._find_image_for_label(txt_path)
                if img_path is not None:
                    for name in classes_in_image:
                        if len(sample_grid[name]) < SAMPLE_PER_CLASS:
                            sample_grid[name].append(img_path)

        low_box_classes = [
            name for name, cnt in box_counts.items() if cnt < 100
        ]
        return box_counts, low_box_classes, degenerate, sample_grid

    def analyze(
        self,
        data_dir: str,
        class_names: Sequence[str],
        split_dirs: Optional[Dict[str, str]] = None,
        task: str = "classification",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        root = Path(data_dir)
        class_names = list(class_names)

        # Class distribution. For OD this is box count per class (flat YOLO
        # layout); for classification it's image count per class subfolder.
        if task == "od":
            if split_dirs:
                class_distribution = {}
                for split_name, split_path in split_dirs.items():
                    box_counts, _, _, _ = self._od_scan(Path(split_path), class_names)
                    class_distribution[split_name] = box_counts
            else:
                box_counts, _, _, _ = self._od_scan(root, class_names)
                class_distribution = {"all": box_counts}
        elif split_dirs:
            class_distribution = {}
            for split_name, split_path in split_dirs.items():
                split_root = Path(split_path)
                per_class = self._class_images(split_root, class_names)
                class_distribution[split_name] = {
                    name: len(paths) for name, paths in per_class.items()
                }
        else:
            per_class_all = self._class_images(root, class_names)
            class_distribution = {
                "all": {name: len(paths) for name, paths in per_class_all.items()}
            }

        # Image size stats + duplicate + corruption scan
        all_images = list(self._iter_images(root))
        total_images = len(all_images)
        widths: List[int] = []
        heights: List[int] = []
        corrupted_paths: List[Path] = []
        md5_map: Dict[str, List[str]] = {}  # hash -> [path_str, ...]

        for idx, img_path in enumerate(all_images):
            size, valid = self._scan_image(img_path)
            if size is not None:
                widths.append(size[0])
                heights.append(size[1])
            if not valid:
                corrupted_paths.append(img_path)
            digest = self._md5(img_path)
            if digest is not None:
                if digest not in md5_map:
                    md5_map[digest] = []
                md5_map[digest].append(str(img_path))

            # Progress callback every 50 images
            if progress_callback and idx % 50 == 0:
                progress_callback({
                    "ts": int(time.time()),
                    "phase": "preprocessing",
                    "stage": "scanning",
                    "images_scanned": idx + 1,
                    "total_images": total_images,
                    "invalid_images": len(corrupted_paths),
                    "duplicates_found": sum(len(paths) - 1 for paths in md5_map.values() if len(paths) > 1),
                })

        # Size statistics
        def _mean(vals: List[int]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        def _std(vals: List[int]) -> float:
            if len(vals) < 2:
                return 0.0
            m = _mean(vals)
            return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

        image_size_stats: Dict[str, Any] = {
            "count": len(all_images),
            "min": [min(widths, default=0), min(heights, default=0)],
            "max": [max(widths, default=0), max(heights, default=0)],
            "mean": [round(_mean(widths), 1), round(_mean(heights), 1)],
            "std": [round(_std(widths), 1), round(_std(heights), 1)],
        }

        # Duplicates
        duplicate_pairs: List[List[str]] = []
        for paths in md5_map.values():
            if len(paths) > 1:
                for i in range(len(paths) - 1):
                    for j in range(i + 1, len(paths)):
                        if len(duplicate_pairs) < MAX_DUPLICATE_PAIRS:
                            duplicate_pairs.append([paths[i], paths[j]])

        duplicate_count = sum(
            len(paths) - 1 for paths in md5_map.values() if len(paths) > 1
        )

        # Final progress callback
        if progress_callback:
            progress_callback({
                "ts": int(time.time()),
                "phase": "preprocessing",
                "stage": "analyzing",
                "images_scanned": total_images,
                "total_images": total_images,
                "invalid_images": len(corrupted_paths),
                "duplicates_found": duplicate_count,
            })

        # Sample grid + OD-specific analysis
        sample_grid: Dict[str, List[str]] = {}
        od_box_counts: Dict[str, int] = {}
        od_low_box_classes: List[str] = []
        od_degenerate_count = 0
        sample_root = Path(next(iter(split_dirs.values()))) if split_dirs else root

        if task == "od":
            od_box_counts, od_low_box_classes, od_degenerate_count, od_sample_grid = self._od_scan(
                sample_root, class_names
            )
            sample_grid = {name: [str(p) for p in paths] for name, paths in od_sample_grid.items()}
        else:
            per_class_sample = self._class_images(sample_root, class_names)
            for class_name in class_names:
                paths = per_class_sample.get(class_name, [])
                sample = paths[:SAMPLE_PER_CLASS]
                if len(paths) > SAMPLE_PER_CLASS:
                    sample = self._rng.sample(paths, min(SAMPLE_PER_CLASS, len(paths)))
                sample_grid[class_name] = [str(p) for p in sample]

        result: Dict[str, Any] = {
            "task": task,
            "class_names": class_names,
            "class_distribution": class_distribution,
            "image_size_stats": image_size_stats,
            "duplicate_count": duplicate_count,
            "duplicate_sample": duplicate_pairs[:MAX_DUPLICATE_PAIRS],
            "corrupted_count": len(corrupted_paths),
            "corrupted_sample": [str(p) for p in corrupted_paths[:10]],
            "sample_grid": sample_grid,
        }

        if task == "od":
            result["od_box_counts"] = od_box_counts
            result["od_low_box_classes"] = od_low_box_classes
            result["od_degenerate_count"] = od_degenerate_count

        return result
