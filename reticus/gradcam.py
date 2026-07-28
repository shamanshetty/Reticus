from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageDraw

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _GRADCAM_LIB = True
except ImportError:
    _GRADCAM_LIB = False


_METHOD_MAP = {
    "gradcam": "GradCAM",
    "gradcam++": "GradCAMPlusPlus",
    "eigencam": "EigenCAM",
}


def _tensor_to_pil(image: Any) -> Image.Image:
    """Convert tensor / numpy array / PIL image to RGB PIL."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if _TORCH and isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    elif _NP:
        arr = np.asarray(image)
    else:
        raise TypeError("Cannot convert image - numpy not available")

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = arr[:, :, None].repeat(3, axis=2) if _NP else arr
    arr = arr.astype("float32")
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = arr.clip(0, 255).astype("uint8")
    return Image.fromarray(arr).convert("RGB")


def _grayscale_heatmap(image: Image.Image) -> Image.Image:
    """Fallback heatmap when pytorch-grad-cam is unavailable."""
    gray = image.convert("L")
    overlay = Image.new("RGBA", image.size, (99, 102, 241, 0))
    alpha = gray.point(lambda p: min(200, int(p * 0.8)))
    overlay.putalpha(alpha)
    base = image.convert("RGBA")
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")


def generate_gradcam(
    model: Any,
    images: Any,
    targets: Sequence[int],
    target_layer: Any,
    method: str,
    class_names: List[str],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Generate Grad-CAM overlay images. Saves both the original image and the
    heatmap overlay so the dashboard can offer a side-by-side toggle.

    Returns a list of records:
      {idx, img_overlay, img_original, true, pred_class_name, method}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    # Normalise images into a list
    if _TORCH and isinstance(images, torch.Tensor):
        img_list = [images[i] for i in range(images.shape[0])]
    elif hasattr(images, "__iter__"):
        img_list = list(images)
    else:
        img_list = [images]

    targets_list = list(targets)

    if _GRADCAM_LIB and _TORCH and model is not None and target_layer is not None:
        method_key = method.lower().replace(" ", "")
        if method_key == "gradcam++":
            cam_cls = GradCAMPlusPlus
        elif method_key == "eigencam":
            cam_cls = EigenCAM
        else:
            cam_cls = GradCAM

        try:
            with cam_cls(model=model, target_layers=[target_layer]) as cam:
                for idx, (img, true_label) in enumerate(zip(img_list, targets_list)):
                    pil_orig = _tensor_to_pil(img)
                    orig_path = output_dir / ("%d_original.png" % idx)
                    pil_orig.save(orig_path)

                    # Prepare input tensor
                    if isinstance(img, torch.Tensor):
                        inp = img.unsqueeze(0).float()
                    else:
                        inp = torch.from_numpy(
                            np.asarray(_tensor_to_pil(img)).transpose(2, 0, 1).astype("float32") / 255.0
                        ).unsqueeze(0)

                    grayscale_cam = cam(input_tensor=inp, targets=None)
                    grayscale_cam = grayscale_cam[0]

                    rgb_np = np.asarray(pil_orig).astype("float32") / 255.0
                    overlay_np = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
                    pil_overlay = Image.fromarray(overlay_np)

                    # Label overlay
                    draw = ImageDraw.Draw(pil_overlay)
                    label = class_names[true_label] if 0 <= true_label < len(class_names) else str(true_label)
                    draw.rectangle([0, 0, min(pil_overlay.width, 300), 24], fill=(0, 0, 0, 180))
                    draw.text((6, 4), "True: %s | %s" % (label, method), fill=(255, 255, 255))

                    overlay_path = output_dir / ("%d_overlay.png" % idx)
                    pil_overlay.save(overlay_path)

                    rows.append({
                        "idx": idx,
                        "img_overlay": str(Path("gradcam") / overlay_path.name),
                        "img_original": str(Path("gradcam") / orig_path.name),
                        "true": true_label,
                        "class_name": label,
                        "method": method,
                    })
        except Exception as exc:
            # Fall back to grayscale heatmap on any error
            rows = _fallback_gradcam(img_list, targets_list, class_names, output_dir, method, str(exc))
    else:
        rows = _fallback_gradcam(img_list, targets_list, class_names, output_dir, method, "pytorch-grad-cam not available")

    return rows


def _fallback_gradcam(
    img_list: List[Any],
    targets_list: List[int],
    class_names: List[str],
    output_dir: Path,
    method: str,
    reason: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, (img, true_label) in enumerate(zip(img_list, targets_list)):
        pil_orig = _tensor_to_pil(img)
        orig_path = output_dir / ("%d_original.png" % idx)
        pil_orig.save(orig_path)

        pil_overlay = _grayscale_heatmap(pil_orig)
        draw = ImageDraw.Draw(pil_overlay)
        label = class_names[true_label] if 0 <= true_label < len(class_names) else str(true_label)
        draw.rectangle([0, 0, min(pil_overlay.width, 300), 24], fill=(0, 0, 0))
        draw.text((6, 4), "True: %s [approx]" % label, fill=(255, 200, 0))

        overlay_path = output_dir / ("%d_overlay.png" % idx)
        pil_overlay.save(overlay_path)

        rows.append({
            "idx": idx,
            "img_overlay": str(Path("gradcam") / overlay_path.name),
            "img_original": str(Path("gradcam") / orig_path.name),
            "true": true_label,
            "class_name": label,
            "method": method,
            "fallback": True,
            "fallback_reason": reason,
        })
    return rows
