import shutil
import tempfile
import unittest
from pathlib import Path

from reticus.storage import (
    build_index,
    count_jsonl_lines,
    read_jsonl_page,
    stream_jsonl,
    write_jsonl_line,
)


class TestOffsetPagination(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write_lines(self, path, n):
        for i in range(n):
            write_jsonl_line(path, {"i": i, "v": i * 2})

    def test_linear_matches_offset_small(self):
        path = self.tmp / "test.jsonl"
        self._write_lines(path, 50)
        index = build_index(path, chunk_size=10)
        linear = read_jsonl_page(path, 15, 10)
        offset = read_jsonl_page(path, 15, 10, index=index)
        self.assertEqual(linear, offset)

    def test_linear_matches_offset_large(self):
        path = self.tmp / "large.jsonl"
        self._write_lines(path, 1000)
        index = build_index(path, chunk_size=100)
        self.assertEqual(index["total"], 1000)
        self.assertEqual(len(index["offsets"]), 11)  # 0, 100, 200, ..., 1000

        linear = read_jsonl_page(path, 250, 20)
        offset_based = read_jsonl_page(path, 250, 20, index=index)
        self.assertEqual(linear, offset_based)
        self.assertEqual(linear[0]["i"], 250)
        self.assertEqual(linear[-1]["i"], 269)

    def test_page_beyond_end_returns_partial(self):
        path = self.tmp / "short.jsonl"
        self._write_lines(path, 5)
        rows = read_jsonl_page(path, 3, 10)
        self.assertEqual(len(rows), 2)

    def test_stream_yields_all(self):
        path = self.tmp / "stream.jsonl"
        self._write_lines(path, 30)
        rows = list(stream_jsonl(path))
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows[0]["i"], 0)
        self.assertEqual(rows[-1]["i"], 29)

    def test_count_lines(self):
        path = self.tmp / "count.jsonl"
        self._write_lines(path, 77)
        self.assertEqual(count_jsonl_lines(path), 77)

    def test_index_total_correct(self):
        path = self.tmp / "idx.jsonl"
        self._write_lines(path, 501)
        index = build_index(path, chunk_size=500)
        self.assertEqual(index["total"], 501)

    def test_missing_file_returns_empty(self):
        rows = read_jsonl_page(self.tmp / "nonexistent.jsonl", 0, 10)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
