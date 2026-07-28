import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from reticus.data_analyzer import DataAnalyzer


def _make_image(path: Path, width: int = 64, height: int = 64, color: int = 100) -> None:
    img = Image.new("RGB", (width, height), color=(color, color, color))
    img.save(str(path))


class TestClassDistribution(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.analyzer = DataAnalyzer(seed=42)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_class_counts_correct(self):
        for cls, count in [("cat", 3), ("dog", 5)]:
            d = self.tmp / cls
            d.mkdir()
            for i in range(count):
                _make_image(d / ("img_%d.png" % i))

        result = self.analyzer.analyze(
            str(self.tmp), ["cat", "dog"], task="classification"
        )
        dist = result["class_distribution"]["all"]
        self.assertEqual(dist["cat"], 3)
        self.assertEqual(dist["dog"], 5)

    def test_split_dirs(self):
        train = self.tmp / "train"
        val = self.tmp / "val"
        for split in [train, val]:
            for cls in ["A", "B"]:
                d = split / cls
                d.mkdir(parents=True)
                _make_image(d / "img.png")

        result = self.analyzer.analyze(
            str(self.tmp),
            ["A", "B"],
            split_dirs={"train": str(train), "val": str(val)},
        )
        self.assertIn("train", result["class_distribution"])
        self.assertIn("val", result["class_distribution"])
        self.assertEqual(result["class_distribution"]["train"]["A"], 1)


class TestDuplicateDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.analyzer = DataAnalyzer(seed=7)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_no_duplicates(self):
        for i in range(3):
            _make_image(self.tmp / ("img_%d.png" % i), color=i * 30)
        result = self.analyzer.analyze(str(self.tmp), [], task="classification")
        self.assertEqual(result["duplicate_count"], 0)

    def test_detects_exact_duplicate(self):
        src = self.tmp / "original.png"
        dup = self.tmp / "copy.png"
        _make_image(src)
        shutil.copy2(str(src), str(dup))
        result = self.analyzer.analyze(str(self.tmp), [], task="classification")
        self.assertGreater(result["duplicate_count"], 0)
        self.assertEqual(len(result["duplicate_sample"]), 1)

    def test_duplicate_count_multiple(self):
        src = self.tmp / "base.png"
        _make_image(src)
        for i in range(3):
            shutil.copy2(str(src), str(self.tmp / ("copy_%d.png" % i)))
        result = self.analyzer.analyze(str(self.tmp), [], task="classification")
        # 3 copies = 3 extra = duplicate_count 3
        self.assertEqual(result["duplicate_count"], 3)


class TestCorruptedImages(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.analyzer = DataAnalyzer(seed=1)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_corrupted_file_detected(self):
        bad = self.tmp / "bad.png"
        bad.write_bytes(b"not an image file at all")
        result = self.analyzer.analyze(str(self.tmp), [], task="classification")
        self.assertGreater(result["corrupted_count"], 0)

    def test_valid_images_not_corrupted(self):
        for i in range(3):
            _make_image(self.tmp / ("ok_%d.png" % i))
        result = self.analyzer.analyze(str(self.tmp), [], task="classification")
        self.assertEqual(result["corrupted_count"], 0)


class TestSampleGrid(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.analyzer = DataAnalyzer(seed=99)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_sample_grid_max_6_per_class(self):
        d = self.tmp / "catX"
        d.mkdir()
        for i in range(12):
            _make_image(d / ("img_%d.png" % i))

        result = self.analyzer.analyze(str(self.tmp), ["catX"], task="classification")
        samples = result["sample_grid"].get("catX", [])
        self.assertLessEqual(len(samples), 6)

    def test_sample_grid_fewer_than_6(self):
        d = self.tmp / "rare"
        d.mkdir()
        for i in range(2):
            _make_image(d / ("img_%d.png" % i))

        result = self.analyzer.analyze(str(self.tmp), ["rare"], task="classification")
        self.assertEqual(len(result["sample_grid"]["rare"]), 2)


def _write_yolo_label(path: Path, boxes) -> None:
    """boxes: list of (class_id, cx, cy, w, h), all normalized 0-1."""
    with open(path, "w") as f:
        for cid, cx, cy, w, h in boxes:
            f.write("%d %.6f %.6f %.6f %.6f\n" % (cid, cx, cy, w, h))


class TestOdDatasetAnalysis(unittest.TestCase):
    """OD datasets are flat (images + YOLO .txt labels), not
    subdirectory-per-class like classification - class_distribution and
    sample_grid must be derived from label scans, not directory names."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.analyzer = DataAnalyzer(seed=3)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_flat_layout_box_counts_and_samples(self):
        # image + .txt side by side, flat directory (no class subfolders)
        for i in range(3):
            _make_image(self.tmp / ("%d.png" % i))
            _write_yolo_label(self.tmp / ("%d.txt" % i), [(0, 0.5, 0.5, 0.2, 0.2)])
        _make_image(self.tmp / "3.png")
        _write_yolo_label(self.tmp / "3.txt", [(1, 0.3, 0.3, 0.1, 0.1)])

        result = self.analyzer.analyze(str(self.tmp), ["cat", "dog"], task="od")
        dist = result["class_distribution"]["all"]
        self.assertEqual(dist["cat"], 3)
        self.assertEqual(dist["dog"], 1)
        self.assertEqual(result["od_box_counts"], dist)
        self.assertEqual(len(result["sample_grid"]["cat"]), 3)
        self.assertEqual(len(result["sample_grid"]["dog"]), 1)

    def test_images_labels_sibling_dirs(self):
        # the other dominant real-world YOLO layout: images/ + labels/
        (self.tmp / "images").mkdir()
        (self.tmp / "labels").mkdir()
        for i in range(2):
            _make_image(self.tmp / "images" / ("%d.png" % i))
            _write_yolo_label(self.tmp / "labels" / ("%d.txt" % i), [(0, 0.5, 0.5, 0.2, 0.2)])

        result = self.analyzer.analyze(str(self.tmp), ["cat"], task="od")
        self.assertEqual(result["class_distribution"]["all"]["cat"], 2)
        samples = result["sample_grid"]["cat"]
        self.assertEqual(len(samples), 2)
        # samples must resolve to the actual image files, not the .txt labels
        for p in samples:
            self.assertTrue(Path(p).exists())
            self.assertIn("images", Path(p).parts)

    def test_degenerate_box_detected(self):
        _make_image(self.tmp / "0.png")
        _write_yolo_label(self.tmp / "0.txt", [(0, 0.5, 0.5, 0.0, 0.2)])  # zero width
        result = self.analyzer.analyze(str(self.tmp), ["cat"], task="od")
        self.assertEqual(result["od_degenerate_count"], 1)

    def test_split_dirs_use_per_split_box_counts(self):
        train = self.tmp / "train"
        val = self.tmp / "val"
        train.mkdir()
        val.mkdir()
        _make_image(train / "0.png")
        _write_yolo_label(train / "0.txt", [(0, 0.5, 0.5, 0.2, 0.2)] * 5)
        _make_image(val / "0.png")
        _write_yolo_label(val / "0.txt", [(0, 0.5, 0.5, 0.2, 0.2)] * 2)

        result = self.analyzer.analyze(
            str(self.tmp), ["cat"],
            split_dirs={"train": str(train), "val": str(val)},
            task="od",
        )
        self.assertEqual(result["class_distribution"]["train"]["cat"], 5)
        self.assertEqual(result["class_distribution"]["val"]["cat"], 2)


class TestImageSizeStats(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.analyzer = DataAnalyzer(seed=0)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_size_stats_populated(self):
        _make_image(self.tmp / "a.png", 64, 64)
        _make_image(self.tmp / "b.png", 128, 128)
        result = self.analyzer.analyze(str(self.tmp), [], task="classification")
        stats = result["image_size_stats"]
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["min"][0], 64)
        self.assertEqual(stats["max"][0], 128)


if __name__ == "__main__":
    unittest.main()
