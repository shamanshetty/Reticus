from __future__ import annotations

import io
from typing import Any, Callable, Dict, List, Optional, Sequence

from PIL import Image

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


# A predictor takes a PIL image and returns either:
#   - a 1-D sequence of class probabilities (len == num classes), or
#   - a dict {class_name: prob}
Predictor = Callable[[Image.Image], Any]


def decode_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an RGB PIL image."""
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")


def _to_prob_list(raw: Any, class_names: Sequence[str]) -> List[float]:
    """Normalise a predictor's raw output into a probability list aligned to class_names."""
    n = len(class_names)

    # dict {name: prob}
    if isinstance(raw, dict):
        return [float(raw.get(name, 0.0)) for name in class_names]

    # tensor / ndarray / list
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu()
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    if _NP:
        arr = np.asarray(raw, dtype="float64").reshape(-1)
        # if it does not already look like a distribution, softmax it
        s = arr.sum()
        looks_like_probs = arr.min() >= 0.0 and abs(s - 1.0) < 1e-3
        if not looks_like_probs:
            e = np.exp(arr - arr.max())
            arr = e / e.sum() if e.sum() > 0 else arr
        vals = arr.tolist()
    else:
        vals = [float(v) for v in list(raw)]
        total = sum(vals)
        if total > 0:
            vals = [v / total for v in vals]

    if len(vals) < n:
        vals = vals + [0.0] * (n - len(vals))
    return [float(v) for v in vals[:n]]


def run_predictor(
    predictor: Predictor,
    image: Image.Image,
    class_names: Sequence[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Invoke a predictor and return a structured top-k result."""
    raw = predictor(image)
    probs = _to_prob_list(raw, class_names)

    ranked = sorted(
        ({"class_id": i, "class_name": class_names[i], "prob": p} for i, p in enumerate(probs)),
        key=lambda d: d["prob"],
        reverse=True,
    )
    top = ranked[: max(1, top_k)]
    best = ranked[0] if ranked else {"class_id": -1, "class_name": "?", "prob": 0.0}

    return {
        "pred_class_id": best["class_id"],
        "pred_class_name": best["class_name"],
        "confidence": round(float(best["prob"]), 6),
        "top_k": [{"class_name": d["class_name"], "prob": round(float(d["prob"]), 6)} for d in top],
        "all_probs": [round(float(p), 6) for p in probs],
        "class_names": list(class_names),
    }


def build_torch_classifier_predictor(
    model: Any,
    preprocess: Optional[Callable[[Image.Image], Any]] = None,
    device: str = "cpu",
    input_size: int = 224,
) -> Predictor:
    """
    Wrap a torch.nn.Module (or TorchScript module) into a Predictor that returns
    softmax probabilities. If `preprocess` is None, a standard resize->tensor->ImageNet
    normalize pipeline is used.
    """
    import torch

    model = model.to(device)
    model.eval()

    if preprocess is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def preprocess(img: Image.Image) -> Any:  # type: ignore
            img = img.resize((input_size, input_size))
            arr = np.asarray(img, dtype="float32") / 255.0  # HWC
            t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
            return (t - mean) / std

    def predict(img: Image.Image) -> Any:
        with torch.no_grad():
            x = preprocess(img)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.to(device)
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.softmax(logits.reshape(logits.shape[0], -1), dim=1)[0]
            return probs

    return predict


def load_scripted_predictor(path: str, device: str = "cpu", input_size: int = 224) -> Predictor:
    """Best-effort standalone reload of a TorchScript model saved by save_model."""
    import torch

    model = torch.jit.load(path, map_location=device)
    return build_torch_classifier_predictor(model, device=device, input_size=input_size)
