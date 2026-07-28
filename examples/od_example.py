"""
Object detection example - numpy only, no torch needed.
Synthetic boxes in VOC format (x1,y1,x2,y2 absolute pixels), 5 classes, 5 epochs.
"""
import random
import time

from reticus import Reticus
from reticus.metrics import detection_metrics

CLASSES = ["person", "car", "bike", "dog", "cat"]
N_CLASSES = len(CLASSES)
IMG_W, IMG_H = 640, 480
N_EPOCHS = 5
N_IMAGES = 20


def rand_box() -> list:
    x1 = random.randint(0, IMG_W - 50)
    y1 = random.randint(0, IMG_H - 50)
    x2 = random.randint(x1 + 20, min(x1 + 200, IMG_W))
    y2 = random.randint(y1 + 20, min(y1 + 200, IMG_H))
    cls = random.randint(0, N_CLASSES - 1)
    conf = round(random.uniform(0.4, 0.99), 3)
    return [float(x1), float(y1), float(x2), float(y2), cls, conf]


def rand_gt_box() -> list:
    x1 = random.randint(0, IMG_W - 50)
    y1 = random.randint(0, IMG_H - 50)
    x2 = random.randint(x1 + 20, min(x1 + 200, IMG_W))
    y2 = random.randint(y1 + 20, min(y1 + 200, IMG_H))
    cls = random.randint(0, N_CLASSES - 1)
    return [float(x1), float(y1), float(x2), float(y2), cls]


def synthetic_od_epoch(epoch: int):
    hit_rate = 0.3 + 0.12 * epoch
    preds = []
    trues = []
    for _ in range(N_IMAGES):
        gt = [rand_gt_box() for _ in range(random.randint(1, 4))]
        trues.append(gt)
        img_preds = []
        for g in gt:
            if random.random() < hit_rate:
                noise = 10.0
                p = [
                    max(0.0, g[0] + random.gauss(0, noise)),
                    max(0.0, g[1] + random.gauss(0, noise)),
                    min(float(IMG_W), g[2] + random.gauss(0, noise)),
                    min(float(IMG_H), g[3] + random.gauss(0, noise)),
                    g[4],
                    round(random.uniform(0.6, 0.99), 3),
                ]
                img_preds.append(p)
            else:
                img_preds.append(rand_box())
        preds.append(img_preds)
    return preds, trues


def main() -> None:
    r = Reticus("od_demo", open_browser=False)
    r.start_run(config={
        "task": "od",
        "model": "synthetic_detector",
        "box_format": "voc",
        "img_w": IMG_W,
        "img_h": IMG_H,
        "n_classes": N_CLASSES,
        "class_names": CLASSES,
        "optimizer": "sgd",
        "lr": 5e-3,
    })

    for epoch in range(1, N_EPOCHS + 1):
        preds, trues = synthetic_od_epoch(epoch)
        metrics = detection_metrics(preds, trues, CLASSES)
        train_loss = round(2.0 - 0.3 * epoch + random.gauss(0, 0.05), 4)
        val_loss = round(2.1 - 0.28 * epoch + random.gauss(0, 0.07), 4)

        r.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
        }, total_epochs=N_EPOCHS)

        # Log predictions (images=None -> placeholder thumbnails)
        r.log_predictions(
            images=[None] * N_IMAGES,
            class_names=CLASSES,
            task="od",
            pred_boxes=preds,
            true_boxes=trues,
            box_format="xyxy",  # our synthetic boxes are already xyxy absolute
        )

        print(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  mAP@50=%.4f  mAP@50:95=%.4f"
            % (epoch, N_EPOCHS, train_loss, val_loss, metrics["map50"], metrics["map50_95"])
        )
        time.sleep(0.05)

    # Show stub-check result
    preds_f, trues_f = synthetic_od_epoch(N_EPOCHS)
    final = detection_metrics(preds_f, trues_f, CLASSES)
    pc = final["per_class"][0]
    stub = round(pc["precision"] * pc["recall"], 4)
    real = round(pc["ap50"], 4)
    print(
        "\nStub check (%s): p*r=%.4f  real_ap50=%.4f  different=%s"
        % (pc["class_name"], stub, real, stub != real)
    )

    r.end_run()

    report_path = r.generate_report(run_id="run_1")
    print("Report written to: %s" % report_path)
    print("Done.")


if __name__ == "__main__":
    main()
