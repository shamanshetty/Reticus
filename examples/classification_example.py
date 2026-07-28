"""
Classification example - numpy only, no torchvision needed.
Generates synthetic 3-class data, runs 10 epochs, logs to Reticus.
"""
import time
import numpy as np

from reticus import Reticus

CLASSES = ["cat", "dog", "bird"]
N_CLASSES = len(CLASSES)
N_EPOCHS = 10
N_PREDS = 60  # predictions logged per run


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def synthetic_epoch(epoch: int) -> dict:
    progress = epoch / N_EPOCHS
    noise = np.random.normal(0, 0.05)
    train_acc = min(0.99, 0.5 + 0.45 * progress + noise)
    val_acc = min(0.97, 0.48 + 0.42 * progress + noise * 1.2)
    train_loss = max(0.05, 1.0 - 0.9 * progress + abs(noise))
    val_loss = max(0.07, 1.05 - 0.88 * progress + abs(noise) * 1.2)
    lr = 1e-3 * (0.9 ** epoch)
    return {
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "lr": lr,
    }


def main() -> None:
    r = Reticus("classification_demo", open_browser=False)

    # Record the preprocessing pipeline (purely descriptive - no images required,
    # but passing a few gives the Preprocessing tab a before/after gallery too).
    r.log_preprocessing(steps=[
        {"name": "Resize", "params": {"size": 224}},
        {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
        {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406]}},
    ])

    r.start_run(config={
        "task": "classification",
        "model": "synthetic_linear",
        "optimizer": "adam",
        "lr": 1e-3,
        "batch_size": 32,
        "n_classes": N_CLASSES,
        "class_names": CLASSES,
    })

    for epoch in range(1, N_EPOCHS + 1):
        data = synthetic_epoch(epoch)
        r.log(data, total_epochs=N_EPOCHS)
        print(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f"
            % (epoch, N_EPOCHS, data["train_loss"], data["val_loss"], data["val_acc"])
        )
        time.sleep(0.05)

    # Log synthetic predictions (images=None -> 64x64 blank placeholder)
    progress = 1.0
    true_labels = [i % N_CLASSES for i in range(N_PREDS)]
    logits = np.random.randn(N_PREDS, N_CLASSES)
    for i, t in enumerate(true_labels):
        logits[i, t] += 3.0 * progress
    confs = softmax(logits)
    pred_labels = confs.argmax(axis=1).tolist()

    r.log_predictions(
        images=[None] * N_PREDS,
        class_names=CLASSES,
        task="classification",
        preds=pred_labels,
        targets=true_labels,
        probs=confs.tolist(),  # full prob vectors -> enables ROC + calibration
    )

    # Compute macro F1 for terminal output
    from reticus.metrics import classification_metrics
    metrics = classification_metrics(true_labels, pred_labels, CLASSES)
    macro_f1 = sum(c["f1"] for c in metrics["per_class"]) / N_CLASSES
    print("\nMacro F1: %.4f" % macro_f1)

    r.end_run()

    # Render a self-contained offline HTML report for this run.
    report_path = r.generate_report(run_id="run_1")
    print("Report written to: %s" % report_path)

    # Register a (synthetic) predictor so the Inference tab is usable in the demo.
    # Maps an image's mean brightness to a class distribution - purely illustrative.
    def demo_predictor(img):
        arr = np.asarray(img.resize((32, 32)), dtype="float32") / 255.0
        feat = float(arr.mean())
        logits = np.array([feat * 4, (1 - feat) * 4, abs(feat - 0.5) * 6])
        return softmax(logits.reshape(1, -1))[0].tolist()

    r.register_predictor(demo_predictor, class_names=CLASSES)

    print("Done. Open dashboard at http://127.0.0.1:8765")
    print("Inference tab is live (synthetic predictor). Ctrl+C to stop.")


if __name__ == "__main__":
    main()
