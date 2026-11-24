"""
Torch classifier example - requires `pip install ".[all]"` (or `".[torch]"`) from a source checkout.

The other two examples (classification_example.py, od_example.py) are
deliberately dependency-light and never touch torch. This one demonstrates
the torch-specific surface: register_torch_classifier, save_model
(TorchScript export), and log_gradcam - using a tiny real CNN and real
(synthetic) images, not numpy stand-ins.
"""
import time

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from reticus import Reticus

CLASSES = ["cat", "dog"]
N_CLASSES = len(CLASSES)
IMG_SIZE = 64
N_EPOCHS = 25
N_SAMPLES = 12


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def make_image(class_id: int) -> Image.Image:
    """A colored circle (cat) or square (dog) on a gray background - just
    enough visual structure for Grad-CAM to have something to point at."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    box = [16, 16, 48, 48]
    if class_id == 0:
        draw.ellipse(box, fill=(200, 60, 60))
    else:
        draw.rectangle(box, fill=(60, 90, 200))
    return img


def main() -> None:
    torch.manual_seed(0)
    model = TinyCNN(N_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    images = [make_image(i % N_CLASSES) for i in range(N_SAMPLES)]
    labels = torch.tensor([i % N_CLASSES for i in range(N_SAMPLES)])

    def to_tensor(img: Image.Image) -> torch.Tensor:
        import numpy as np
        arr = np.asarray(img, dtype="float32") / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    batch = torch.stack([to_tensor(im) for im in images])

    r = Reticus("torch_demo", open_browser=False)
    r.start_run(config={
        "task": "classification",
        "model": "TinyCNN",
        "optimizer": "adam",
        "lr": 1e-3,
        "class_names": CLASSES,
    })

    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()

        model.eval()
        with torch.no_grad():
            preds = model(batch).argmax(dim=1)
            acc = float((preds == labels).float().mean())
        model.train()

        r.log({
            "epoch": epoch,
            "train_loss": round(loss_val, 4),
            "val_loss": round(loss_val * 1.05, 4),
            "train_acc": round(acc, 4),
            "val_acc": round(acc, 4),
        }, total_epochs=N_EPOCHS)
        print("Epoch %d/%d  loss=%.4f  acc=%.4f" % (epoch, N_EPOCHS, loss_val, acc))
        time.sleep(0.05)

    model.eval()
    with torch.no_grad():
        pred_labels = model(batch).argmax(dim=1).tolist()

    r.log_predictions(
        images=images,
        class_names=CLASSES,
        task="classification",
        preds=pred_labels,
        targets=labels.tolist(),
    )

    # Grad-CAM - needs a real conv layer to target.
    r.log_gradcam(
        images=images,
        targets=labels.tolist(),
        class_names=CLASSES,
        model=model,
        target_layer=model.conv2,
    )

    r.end_run()

    # Save a TorchScript artifact, downloadable from the Inference tab.
    saved_path = r.save_model(model, class_names=CLASSES, scripted=True)
    print("Model saved to: %s" % saved_path)

    # Register the live model so the Inference tab can run real predictions.
    r.register_torch_classifier(model, class_names=CLASSES, input_size=IMG_SIZE)

    report_path = r.generate_report(run_id="run_1")
    print("Report written to: %s" % report_path)

    print("Done. Open dashboard at http://127.0.0.1:8765")
    print("Inference tab is live (real TinyCNN). Ctrl+C to stop.")


if __name__ == "__main__":
    main()
