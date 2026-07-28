import unittest
from reticus.rules import (
    evaluate_data_flags,
    evaluate_training_flags,
    evaluate_results_flags,
    merge_flags,
)


class TestDataFlags(unittest.TestCase):
    def test_no_flags_balanced_dataset(self):
        analysis = {
            "class_distribution": {"train": {"A": 1000, "B": 1000}},
            "duplicate_count": 0,
            "corrupted_count": 0,
        }
        flags = evaluate_data_flags(analysis)
        self.assertEqual(flags, [])

    def test_imbalance_warning(self):
        # A=1000, B=300 -> mean=650, B=300 < 650*0.5=325 -> IMBALANCE_WARNING
        analysis = {
            "class_distribution": {"train": {"A": 1000, "B": 300}},
            "duplicate_count": 0,
            "corrupted_count": 0,
        }
        flags = evaluate_data_flags(analysis)
        ids = [f["flag_id"] for f in flags]
        self.assertIn("IMBALANCE_WARNING", ids)

    def test_critical_imbalance(self):
        analysis = {
            "class_distribution": {"train": {"A": 1000, "B": 100}},
            "duplicate_count": 0,
            "corrupted_count": 0,
        }
        flags = evaluate_data_flags(analysis)
        ids = [f["flag_id"] for f in flags]
        self.assertIn("CRITICAL_IMBALANCE", ids)

    def test_duplicates_flag(self):
        analysis = {
            "class_distribution": {},
            "duplicate_count": 5,
            "corrupted_count": 0,
        }
        flags = evaluate_data_flags(analysis)
        self.assertTrue(any(f["flag_id"] == "DUPLICATES_FOUND" for f in flags))

    def test_corrupted_flag(self):
        analysis = {
            "class_distribution": {},
            "duplicate_count": 0,
            "corrupted_count": 3,
        }
        flags = evaluate_data_flags(analysis)
        self.assertTrue(any(f["flag_id"] == "CORRUPTED_IMAGES" for f in flags))

    def test_od_low_box_count_flag(self):
        analysis = {
            "class_distribution": {},
            "duplicate_count": 0,
            "corrupted_count": 0,
            "od_low_box_classes": ["person"],
        }
        flags = evaluate_data_flags(analysis)
        self.assertTrue(any(f["flag_id"] == "OD_LOW_BOX_COUNT" for f in flags))


class TestTrainingFlags(unittest.TestCase):
    def test_no_flags_healthy_training(self):
        epochs = [
            {"epoch": i, "train_acc": 0.8, "val_acc": 0.79, "val_loss": 0.3 - i * 0.01}
            for i in range(1, 8)
        ]
        flags = evaluate_training_flags(epochs)
        self.assertEqual(flags, [])

    def test_overfitting_flag(self):
        epochs = [
            {"epoch": i, "train_acc": 0.9, "val_acc": 0.7, "val_loss": 0.5}
            for i in range(1, 6)
        ]
        flags = evaluate_training_flags(epochs)
        ids = [f["flag_id"] for f in flags]
        self.assertTrue("OVERFITTING" in ids or "SEVERE_OVERFITTING" in ids)

    def test_severe_overfitting_flag(self):
        epochs = [
            {"epoch": i, "train_acc": 0.95, "val_acc": 0.6, "val_loss": 0.5}
            for i in range(1, 6)
        ]
        flags = evaluate_training_flags(epochs)
        self.assertTrue(any(f["flag_id"] == "SEVERE_OVERFITTING" for f in flags))

    def test_plateau_flag(self):
        epochs = [{"epoch": i, "val_loss": 0.5000} for i in range(1, 8)]
        flags = evaluate_training_flags(epochs)
        self.assertTrue(any(f["flag_id"] == "VAL_LOSS_PLATEAU" for f in flags))

    def test_val_loss_increasing_flag(self):
        # 5 consecutive strictly-rising val_loss values, well above the
        # plateau delta threshold, so VAL_LOSS_INCREASING fires instead.
        epochs = [{"epoch": i, "val_loss": 0.1 * i} for i in range(1, 6)]
        flags = evaluate_training_flags(epochs)
        self.assertTrue(any(f["flag_id"] == "VAL_LOSS_INCREASING" for f in flags))

    def test_lr_too_high_flag(self):
        # Oscillating (non-monotonic) val_loss with std > 0.3 over the last
        # 5 epochs - independent of the plateau/increasing branch.
        losses = [0.1, 0.9, 0.1, 0.9, 0.1]
        epochs = [{"epoch": i + 1, "val_loss": v} for i, v in enumerate(losses)]
        flags = evaluate_training_flags(epochs)
        self.assertTrue(any(f["flag_id"] == "LR_TOO_HIGH" for f in flags))


class TestResultsFlags(unittest.TestCase):
    def test_low_f1_flag(self):
        metrics = {
            "task": "classification",
            "per_class": [
                {"class_name": "A", "f1": 0.3, "recall": 0.6, "precision": 0.3},
                {"class_name": "B", "f1": 0.9, "recall": 0.9, "precision": 0.9},
            ],
        }
        flags = evaluate_results_flags(metrics)
        self.assertTrue(any(f["flag_id"] == "LOW_CLASS_F1" for f in flags))

    def test_critical_recall_flag(self):
        metrics = {
            "task": "classification",
            "per_class": [
                {"class_name": "A", "f1": 0.4, "recall": 0.2, "precision": 0.8},
            ],
        }
        flags = evaluate_results_flags(metrics)
        self.assertTrue(any(f["flag_id"] == "CRITICAL_CLASS_RECALL" for f in flags))

    def test_no_flags_good_metrics(self):
        metrics = {
            "task": "classification",
            "per_class": [
                {"class_name": "A", "f1": 0.9, "recall": 0.88, "precision": 0.92},
                {"class_name": "B", "f1": 0.85, "recall": 0.84, "precision": 0.86},
            ],
        }
        flags = evaluate_results_flags(metrics)
        self.assertEqual(flags, [])

    def test_low_map_od_flag(self):
        metrics = {"task": "od", "map50": 0.25, "per_class": [], "overall": {"map50": 0.25}}
        flags = evaluate_results_flags(metrics)
        self.assertTrue(any(f["flag_id"] == "LOW_MAP" for f in flags))

    def test_high_precision_low_recall_flag(self):
        metrics = {
            "task": "classification",
            "per_class": [
                {"class_name": "A", "f1": 0.588, "recall": 0.45, "precision": 0.85},
            ],
        }
        flags = evaluate_results_flags(metrics)
        self.assertTrue(any(f["flag_id"] == "HIGH_PRECISION_LOW_RECALL" for f in flags))

    def test_class_confusion_flag(self):
        metrics = {
            "task": "classification",
            "per_class": [
                {"class_name": "A", "f1": 0.7, "recall": 0.6, "precision": 0.8},
                {"class_name": "B", "f1": 0.9, "recall": 0.9, "precision": 0.9},
            ],
            "class_names": ["A", "B"],
            "confusion_matrix_normalized": [[0.6, 0.4], [0.1, 0.9]],
        }
        flags = evaluate_results_flags(metrics)
        confusion = [f for f in flags if f["flag_id"] == "CLASS_CONFUSION"]
        self.assertEqual(len(confusion), 1)
        self.assertIn("A", confusion[0]["message"])
        self.assertIn("B", confusion[0]["message"])


class TestMergeFlags(unittest.TestCase):
    def test_dedup(self):
        f = {"flag_id": "X", "severity": "WARNING", "message": "msg", "suggestion": "", "stage": "data"}
        merged = merge_flags([f], [f])
        self.assertEqual(len(merged), 1)

    def test_sort_order(self):
        c = {"flag_id": "C", "severity": "CRITICAL", "message": "", "suggestion": "", "stage": "data"}
        w = {"flag_id": "W", "severity": "WARNING", "message": "", "suggestion": "", "stage": "data"}
        i = {"flag_id": "I", "severity": "INFO", "message": "", "suggestion": "", "stage": "data"}
        merged = merge_flags([w], [c], [i])
        self.assertEqual(merged[0]["severity"], "CRITICAL")
        self.assertEqual(merged[-1]["severity"], "INFO")

    def test_same_flag_id_different_stage_not_deduped(self):
        a = {"flag_id": "X", "severity": "WARNING", "message": "msg", "suggestion": "", "stage": "data"}
        b = {"flag_id": "X", "severity": "WARNING", "message": "msg", "suggestion": "", "stage": "results"}
        merged = merge_flags([a], [b])
        # Different stage -> not deduped
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
