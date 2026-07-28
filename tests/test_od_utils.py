import unittest
from reticus.od_utils import normalize_boxes, box_iou, box_area


class TestNormalizeBoxes(unittest.TestCase):
    def test_xyxy_passthrough(self):
        boxes = normalize_boxes([[10, 20, 50, 80]], "xyxy", 200, 200)
        self.assertEqual(boxes[0][:4], [10.0, 20.0, 50.0, 80.0])

    def test_voc_passthrough(self):
        boxes = normalize_boxes([[10, 20, 50, 80]], "voc", 200, 200)
        self.assertEqual(boxes[0][:4], [10.0, 20.0, 50.0, 80.0])

    def test_yolo_center_format(self):
        # cx=0.5, cy=0.5, w=0.5, h=0.5 at 200x200 -> x1=50, y1=50, x2=150, y2=150
        boxes = normalize_boxes([[0.5, 0.5, 0.5, 0.5]], "yolo", 200, 200)
        self.assertAlmostEqual(boxes[0][0], 50.0, places=3)
        self.assertAlmostEqual(boxes[0][1], 50.0, places=3)
        self.assertAlmostEqual(boxes[0][2], 150.0, places=3)
        self.assertAlmostEqual(boxes[0][3], 150.0, places=3)

    def test_coco_format(self):
        # x=10, y=20, w=40, h=60 -> x1=10, y1=20, x2=50, y2=80
        boxes = normalize_boxes([[10, 20, 40, 60]], "coco", 200, 200)
        self.assertEqual(boxes[0][:4], [10.0, 20.0, 50.0, 80.0])

    def test_extra_fields_preserved(self):
        boxes = normalize_boxes([[10, 20, 50, 80, 3, 0.95]], "xyxy", 200, 200)
        self.assertEqual(boxes[0][4], 3.0)
        self.assertAlmostEqual(boxes[0][5], 0.95)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            normalize_boxes([[1, 2, 3, 4]], "bad_format", 100, 100)

    def test_empty_input(self):
        self.assertEqual(normalize_boxes([], "xyxy", 100, 100), [])

    def test_yolo_clamps_to_image(self):
        # cx slightly out of bounds
        boxes = normalize_boxes([[1.1, 0.5, 0.2, 0.2]], "yolo", 100, 100)
        self.assertLessEqual(boxes[0][2], 100.0)


class TestBoxIou(unittest.TestCase):
    def test_perfect_overlap(self):
        box = [10, 10, 50, 50]
        self.assertAlmostEqual(box_iou(box, box), 1.0)

    def test_no_overlap(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_partial_overlap(self):
        iou = box_iou([0, 0, 10, 10], [5, 0, 15, 10])
        self.assertAlmostEqual(iou, 0.333, places=2)

    def test_zero_area(self):
        self.assertAlmostEqual(box_iou([5, 5, 5, 5], [0, 0, 10, 10]), 0.0)


if __name__ == "__main__":
    unittest.main()
