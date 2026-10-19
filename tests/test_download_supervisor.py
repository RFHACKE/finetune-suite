"""A temporary billing outage must not destroy downloads or bypass the cap."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    from tools import watch_large_ranks as watcher
except ModuleNotFoundError as exc:
    if exc.name == "modal":
        raise unittest.SkipTest("Cloud supervisor checks require the optional Modal SDK") from exc
    raise


class DownloadSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.root = self.workspace / "Finetune-result/large-ranks"
        self.execution = self.root / "execution"
        self.execution.mkdir(parents=True)
        self.timeout = subprocess.TimeoutExpired(["modal", "billing", "summary"], 30)

    def cached(self, cost="46.0", age=0):
        path = self.execution / "billing-raw-delivery.json"
        path.write_text(json.dumps({"metered_cost": cost}))
        os.utime(path, (time.time() - age, time.time() - age))
        return path

    def test_success_updates_checked_balance_and_status(self):
        response = SimpleNamespace(returncode=0, stdout='{"metered_cost":"46.1"}')
        with patch.object(watcher.subprocess, "run", return_value=response) as command:
            metered, cached = watcher.download_billing(self.execution, 51.4)
        self.assertFalse(cached)
        self.assertEqual(metered["metered_cost"], "46.1")
        self.assertEqual(command.call_args.kwargs["timeout"], 30)
        self.assertEqual(json.loads((self.execution / "download-billing-status.json").read_text())["source"], "live")

    def test_timeout_uses_recent_balance_without_refreshing_its_age(self):
        path = self.cached(age=120)
        before = path.stat().st_mtime_ns
        with patch.object(watcher.subprocess, "run", side_effect=self.timeout):
            metered, cached = watcher.download_billing(self.execution, 51.4)
        self.assertTrue(cached)
        self.assertEqual(metered["metered_cost"], "46.0")
        self.assertEqual(path.stat().st_mtime_ns, before)
        status = json.loads((self.execution / "download-billing-status.json").read_text())
        self.assertEqual(status["source"], "recent_cache")
        self.assertGreaterEqual(status["cache_age_s"], 120)
        self.assertEqual(status["retry_after_s"], 30)

    def test_stale_balance_cannot_keep_transfers_running(self):
        self.cached(age=901)
        with patch.object(watcher.subprocess, "run", side_effect=self.timeout):
            with self.assertRaisesRegex(RuntimeError, "no recent checked balance"):
                watcher.download_billing(self.execution, 51.4)
        status = json.loads((self.execution / "download-billing-status.json").read_text())
        self.assertTrue(status["downloads_must_pause"])

    def test_cached_balance_near_the_limit_is_not_accepted(self):
        self.cached(cost="51.0")
        with patch.object(watcher.subprocess, "run", side_effect=self.timeout):
            with self.assertRaises(RuntimeError):
                watcher.download_billing(self.execution, 51.4)

    def test_unavailable_or_invalid_first_check_cannot_launch_transfers(self):
        for response in (self.timeout, SimpleNamespace(returncode=0, stdout='{"metered_cost":"NaN"}')):
            options = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
            with patch.object(watcher.subprocess, "run", **options):
                with self.assertRaises(RuntimeError):
                    watcher.download_billing(self.execution, 51.4)
        self.assertFalse((self.execution / "billing-raw-delivery.json").exists())

    def test_fresh_balance_at_the_limit_stops_even_if_old_balance_was_low(self):
        self.cached()
        response = SimpleNamespace(returncode=0, stdout='{"metered_cost":"51.2"}')
        with patch.object(watcher.subprocess, "run", return_value=response):
            metered, cached = watcher.download_billing(self.execution, 51.4)
        self.assertFalse(cached)
        self.assertTrue(watcher.download_budget_exhausted(self.root, 51.4, metered))
        self.assertEqual(json.loads((self.execution / "download-supervisor.json").read_text())["phase"], "budget_stop")

    def test_resumed_status_preserves_old_error_as_history(self):
        path = self.execution / "download-supervisor.json"
        path.write_text(json.dumps({"phase": "error", "error": "billing timeout", "traceback": "old traceback", "reason": "old interruption"}))
        watcher.supervisor_status(self.root, "downloading", downloaded_bytes=100)
        status = json.loads(path.read_text())
        self.assertNotIn("error", status)
        self.assertNotIn("reason", status)
        self.assertEqual(status["last_error"]["error"], "billing timeout")

    def test_exhausted_transport_attempts_raise_for_systemd_recovery(self):
        original = self.workspace / "Finetune-result/execution"
        original.mkdir(parents=True)
        (original / "retrieval.json").write_text('{"retrieval_finished":true}')
        (original / "active-budget.json").write_text('{"approved_total_usd":51.4}')
        (self.execution / "delivery-catalog.json").write_text('{}')
        child = Mock(returncode=1)
        child.poll.return_value = 1
        with (patch.object(watcher, "download_billing", return_value=({"metered_cost": "46"}, False)),
              patch.object(watcher.subprocess, "run", return_value=SimpleNamespace(returncode=0)),
              patch.object(watcher.subprocess, "Popen", return_value=child) as starts,
              patch.object(watcher.modal.Function, "from_name") as endpoint,
              patch.object(watcher.time, "sleep")):
            endpoint.return_value.get_web_url.return_value = "https://example.invalid"
            with self.assertRaisesRegex(RuntimeError, "Three transport attempts exhausted"):
                watcher._collect_raw(self.workspace, self.root, {"run": "test-cohort"})
        self.assertEqual(starts.call_count, 3)


if __name__ == "__main__":
    unittest.main()
