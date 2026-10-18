"""Progress counts downloaded archives without mistaking sparse holes for data."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.download_progress import DownloadProgressLog, snapshot


class DownloadProgressTests(unittest.TestCase):
    def test_counts_deleted_verified_and_extracted_archives_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution = root / "execution"
            incoming = execution / "incoming"
            incoming.mkdir(parents=True)
            records = [{"run_id": name, "archive": name + ".tar.zst",
                        "archive_bytes": 100, "archive_sha256": name + "-sha"}
                       for name in ("verified", "extracted", "partial")]
            catalog = execution / "delivery-catalog.json"
            catalog.write_text(json.dumps({"archives": records}))
            (execution / "retrieval.json").write_text(json.dumps({
                "verified_runs": ["verified"], "extracted_archives": {"extracted": "extracted-sha"}}))
            # Both copies remain briefly during extraction; neither is counted twice.
            (incoming / "verified.tar.zst").write_bytes(b"v" * 100)
            (incoming / "extracted.tar.zst").write_bytes(b"e" * 100)
            (incoming / "partial.tar.zst.download").write_bytes(b"p" * 30)
            (incoming / "partial.tar.zst.download.aria2").write_bytes(b"control" * 100)
            (incoming / "metadata.tar.zst.download").write_bytes(b"metadata" * 100)
            progress = snapshot(root, catalog)
            self.assertEqual(progress["downloaded_bytes"], 230)
            self.assertEqual(progress["total_bytes"], 300)
            self.assertEqual(progress["verified_runs"], 1)

    def test_sparse_final_file_length_is_not_downloaded_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming = root / "execution/incoming"
            incoming.mkdir(parents=True)
            size = 100_000_000
            path = incoming / "partial.tar.zst.download"
            with path.open("wb") as archive:
                archive.seek(size - 1)
                archive.write(b"x")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"archives": [{"run_id": "partial",
                "archive": "partial.tar.zst", "archive_bytes": size, "archive_sha256": "expected"}]}))
            self.assertEqual(path.stat().st_size, size)
            self.assertLess(snapshot(root, catalog)["downloaded_bytes"], size / 100)

    def test_logs_utc_gb_and_waits_ten_minutes(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = DownloadProgressLog(temporary)
            logger.stop = Mock()
            logger.stop.is_set.return_value = False
            progress = {"downloaded_bytes": 7_800_000_000, "total_bytes": 21_640_000_000,
                        "verified_runs": 80, "expected_runs": 140, "complete": False}
            now = datetime(2026, 10, 18, 6, 0, tzinfo=timezone.utc)
            with (patch("tools.download_progress.snapshot", side_effect=[progress,
                        {**progress, "downloaded_bytes": progress["total_bytes"], "complete": True}]),
                  patch("tools.download_progress.datetime", wraps=datetime) as clock):
                clock.now.side_effect = [now, now + timedelta(minutes=10)]
                logger.run()
            logger.stop.wait.assert_called_once_with(600)
            lines = logger.path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith("2026-10-18T06:00:00Z  7.80 GB / 21.64 GB"))
            self.assertTrue(lines[1].startswith("2026-10-18T06:10:00Z"))
            self.assertTrue(lines[1].endswith("  complete"))

    def test_restart_appends_at_original_interval_instead_of_resetting_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = DownloadProgressLog(temporary)
            logger.path.parent.mkdir(parents=True)
            previous = "2026-10-18T06:00:00Z  7.80 GB / 21.64 GB  (36.0%)  verified=80/140\n"
            logger.path.write_text(previous)
            logger.stop = Mock()
            logger.stop.is_set.return_value = False
            progress = {"downloaded_bytes": 8_000_000_000, "total_bytes": 21_640_000_000,
                        "verified_runs": 81, "expected_runs": 140, "complete": False}
            restarted = datetime(2026, 10, 18, 6, 5, tzinfo=timezone.utc)
            with (patch("tools.download_progress.snapshot", side_effect=[progress, {**progress, "complete": True}]),
                  patch("tools.download_progress.datetime", wraps=datetime) as clock):
                clock.now.side_effect = [restarted, restarted + timedelta(minutes=5)]
                logger.run()
            logger.stop.wait.assert_called_once_with(300)
            self.assertTrue(logger.path.read_text().startswith(previous))
            self.assertEqual(len(logger.path.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
