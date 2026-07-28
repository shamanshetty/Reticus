from __future__ import annotations

from typing import Any, List, Sequence, Tuple


SUPPORTED_FORMATS: Tuple[str, ...] = ("xyxy", "yolo", "coco", "voc")


def _to_list(boxes: Any) -> List[List[float]]:
    if boxes is None:
        return []
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    result: List[List[float]] = []
    for box in boxes:
        if hasattr(box, "tolist"):
            box = box.tolist()
        result.append([float(v) for v in box])
    return result


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def normalize_boxes(
    boxes: Any,
    fmt: str,
    img_w: float,
    img_h: float,
) -> List[List[float]]:
    """
    Convert boxes from any supported format to xyxy_abs (absolute pixel coordinates).

    Formats:
      xyxy  - absolute x1, y1, x2, y2
      voc   - same as xyxy (PASCAL VOC XML uses absolute pixel coords)
      coco  - absolute x, y, w, h  (top-left corner + dimensions)
      yolo  - normalized cx, cy, w, h  (center + dimensions, all in [0,1])

    Extra fields (class_id, confidence, ...) are preserved unchanged.
    Raises ValueError for unsupported format strings.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            "Unknown box format %r. Supported: %s" % (fmt, ", ".join(SUPPORTED_FORMATS))
        )

    raw = _to_list(boxes)
    result: List[List[float]] = []

    for box in raw:
        if len(box) < 4:
            result.append(box)
            continue

        extra = box[4:]

        if fmt in ("xyxy", "voc"):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        elif fmt == "coco":
            x, y, w, h = box[0], box[1], box[2], box[3]
            x1, y1 = x, y
            x2, y2 = x + w, y + h

        else:  # yolo
            cx, cy, w, h = box[0], box[1], box[2], box[3]
            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h

        x1 = _clamp(x1, 0.0, img_w)
        y1 = _clamp(y1, 0.0, img_h)
        x2 = _clamp(x2, 0.0, img_w)
        y2 = _clamp(y2, 0.0, img_h)

        result.append([x1, y1, x2, y2] + extra)

    return result


def box_area(box: Sequence[float]) -> float:
    if len(box) < 4:
        return 0.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w <= 0.0 or h <= 0.0:
        return 0.0
    return float(w * h)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1

    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0

    inter = inter_w * inter_h
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return float(inter / union)
