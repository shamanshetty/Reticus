import pytest

pytest.importorskip("fastapi")

from reticus.server import _matches_filter


class TestMatchesFilterClassification:
    def test_class_filter(self):
        row = {"true": 0, "pred": 0, "correct": True, "conf": 0.9}
        assert _matches_filter(row, filter_class=0, filter_correct=None, conf_min=0.0, conf_max=1.0)
        assert not _matches_filter(row, filter_class=1, filter_correct=None, conf_min=0.0, conf_max=1.0)

    def test_correct_filter(self):
        row = {"true": 0, "pred": 1, "correct": False, "conf": 0.4}
        assert _matches_filter(row, filter_class=None, filter_correct=False, conf_min=0.0, conf_max=1.0)
        assert not _matches_filter(row, filter_class=None, filter_correct=True, conf_min=0.0, conf_max=1.0)

    def test_conf_range(self):
        row = {"true": 0, "pred": 0, "correct": True, "conf": 0.3}
        assert not _matches_filter(row, filter_class=None, filter_correct=None, conf_min=0.5, conf_max=1.0)
        assert _matches_filter(row, filter_class=None, filter_correct=None, conf_min=0.0, conf_max=0.5)


class TestMatchesFilterDetection:
    """OD prediction rows have true_boxes/pred_boxes + iou, not true/pred/
    correct/conf - filters must be reinterpreted, not silently dead-end."""

    def _od_row(self, iou, true_cls=(0,), pred_cls=(0,)):
        return {
            "true_boxes": [[0, 0, 10, 10, c] for c in true_cls],
            "pred_boxes": [[0, 0, 10, 10, c, 0.9] for c in pred_cls],
            "iou": iou,
        }

    def test_class_filter_matches_boxes_not_missing_true_pred_keys(self):
        row = self._od_row(iou=0.8, true_cls=(1,), pred_cls=(1,))
        # class 1 appears in this row's boxes -> must match
        assert _matches_filter(row, filter_class=1, filter_correct=None, conf_min=0.0, conf_max=1.0)
        # class 0 never appears -> must not match
        assert not _matches_filter(row, filter_class=0, filter_correct=None, conf_min=0.0, conf_max=1.0)

    def test_correct_filter_uses_iou_threshold(self):
        good = self._od_row(iou=0.8)
        bad = self._od_row(iou=0.1)
        assert _matches_filter(good, filter_class=None, filter_correct=True, conf_min=0.0, conf_max=1.0)
        assert not _matches_filter(bad, filter_class=None, filter_correct=True, conf_min=0.0, conf_max=1.0)
        assert _matches_filter(bad, filter_class=None, filter_correct=False, conf_min=0.0, conf_max=1.0)
        assert not _matches_filter(good, filter_class=None, filter_correct=False, conf_min=0.0, conf_max=1.0)

    def test_conf_range_uses_iou(self):
        row = self._od_row(iou=0.8)
        assert _matches_filter(row, filter_class=None, filter_correct=None, conf_min=0.5, conf_max=1.0)
        assert not _matches_filter(row, filter_class=None, filter_correct=None, conf_min=0.0, conf_max=0.5)
