"""Transport changes preserve bytes, credentials and completed CPU results."""
import hashlib
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec("modal") is None:
    raise unittest.SkipTest("Optional Modal client SDK is required for cloud-helper tests")

from tools.collect_delivery import SegmentedStream, extract_metadata


class DeliveryTests(unittest.TestCase):
    def compressed(self, root, members):
        raw = root / "fixture.tar"
        with tarfile.open(raw, "w") as archive:
            for name, payload in members:
                member = tarfile.TarInfo(name)
                if isinstance(payload, bytes):
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
                else:
                    member.type = tarfile.SYMTYPE
                    member.linkname = payload
                    archive.addfile(member)
        subprocess.run(["zstd", "-q", "-f", str(raw)], check=True)
        return raw.with_suffix(".tar.zst")

    def test_metadata_preserves_completed_spectral_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "out/runs/run/result.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({"intruder_status": "completed", "evidence": "retained"}))
            archive = self.compressed(root, [("protocol.json", b'{"source":"frozen"}'),
                ("runs/run/result.json", b'{"intruder_status":"pending_cpu"}')])
            extract_metadata(archive, root / "out")
            self.assertEqual(json.loads(result.read_text())["evidence"], "retained")
            self.assertEqual(json.loads((root / "out/protocol.json").read_text())["source"], "frozen")

    def test_metadata_rejects_traversal_and_symlink(self):
        for name, payload in [("../escape", b"invalid"), ("execution/link", "/tmp/escape")]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = self.compressed(root, [(name, payload)])
                (root / "out").mkdir()
                with self.assertRaises(ValueError):
                    extract_metadata(archive, root / "out")

    def test_segmented_resume_keeps_control_and_secret_out_of_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "usr/bin/aria2c"
            executable.parent.mkdir(parents=True)
            executable.touch()
            token = root / "access"
            token.write_text("a-private-test-token-with-more-than-thirty-two-characters")
            target = root / "partial.download"
            target.write_bytes(b"old incomplete content")
            target.with_name(target.name + ".aria2").write_bytes(b"resume control")
            data = b"exact original archive bytes"
            expected = hashlib.sha256(data).hexdigest()
            stream = SegmentedStream("https://private.modal.run", token, executable, 16)
            def download(command, **kwargs):
                self.assertNotIn(token.read_text(), " ".join(command))
                config = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--input-file=")))
                self.assertEqual(config.stat().st_mode & 0o777, 0o600)
                self.assertIn(token.read_text(), config.read_text())
                self.assertTrue(target.with_name(target.name + ".aria2").exists())
                target.write_bytes(data)
                return SimpleNamespace(returncode=0)
            with patch("tools.collect_delivery.subprocess.run", side_effect=download):
                stream.download("/cohort/archives/model/run.tar.zst", SimpleNamespace(size=len(data)), target, expected)
            self.assertEqual(target.read_bytes(), data)

    def test_completed_call_waits_for_valid_fresh_status_before_metadata_delivery(self):
        from tools import watch_large_ranks as watcher
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution = root / "execution"
            execution.mkdir()
            (execution / "main-calls.json").write_text(json.dumps({
                "call_id": "main", "results_volume": "results", "run": "cohort"}))
            (execution / "archive-recovery-call.json").write_text('{"call_id":"archive"}')
            (execution / "status.json").write_text('{"completed_sequences":0,"archived_sequences":0}')
            finished = {"counts": {"completed": 40}, "completed_sequences": 40,
                        "archived_sequences": 40, "training_finished": True, "runs": []}
            volume = SimpleNamespace(read_file=Mock(side_effect=[[b""], [json.dumps(finished).encode()]]))
            call = SimpleNamespace(get=lambda timeout: {"complete": True})
            with (patch.object(watcher.modal.FunctionCall, "from_id", return_value=call),
                  patch.object(watcher.modal.Volume, "from_name", return_value=volume),
                  patch.object(watcher.time, "sleep"),
                  patch.object(watcher.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run,
                  patch("sys.argv", ["watcher", "--root", str(root), "--fetch-metadata"]),
                  redirect_stdout(io.StringIO())):
                watcher.main()
            self.assertEqual(volume.read_file.call_count, 2)
            self.assertEqual(run.call_count, 1)
            self.assertIn("--inline-metadata", run.call_args.args[0])
            self.assertTrue((execution / "watch-finished.json").is_file())
            self.assertFalse((execution / "metadata-retrieval-deferred.json").exists())

    def test_raw_transfer_waits_for_original_and_stops_its_own_endpoint_at_budget(self):
        from tools import watch_large_ranks as watcher
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            original = workspace / "Finetune-result/execution"
            original.mkdir(parents=True)
            (original / "retrieval.json").write_text('{"retrieval_finished":false}')
            (original / "active-budget.json").write_text('{"approved_total_usd":46}')
            root = workspace / "Finetune-result/large-ranks"
            (root / "execution").mkdir(parents=True)
            (root / "execution/delivery-catalog.json").write_text('{}')
            child = SimpleNamespace(pid=12345, returncode=None)
            child.poll = lambda: child.returncode
            def finish(timeout):
                child.returncode = -2
                return -2
            child.wait = Mock(side_effect=finish)
            def previous_finished(_seconds):
                (original / "retrieval.json").write_text('{"retrieval_finished":true}')
            def command(args, **kwargs):
                if "billing" in args:
                    return SimpleNamespace(returncode=0, stdout='{"metered_cost":"45.90"}')
                return SimpleNamespace(returncode=0)
            endpoint = SimpleNamespace(get_web_url=lambda: "https://isolated.modal.run")
            with (patch.object(watcher.subprocess, "run", side_effect=command) as run,
                  patch.object(watcher.subprocess, "Popen", return_value=child) as popen,
                  patch.object(watcher.modal.Function, "from_name", return_value=endpoint),
                  patch.object(watcher.time, "sleep", side_effect=previous_finished) as sleep,
                  patch.object(watcher.os, "killpg") as kill):
                watcher.collect_raw(workspace, root, {"run": "extension"})
            self.assertEqual(sleep.call_count, 1)
            self.assertIn("tools.collect_delivery", popen.call_args.args[0])
            self.assertTrue((root / "execution/raw-delivery-budget-stop.json").exists())
            self.assertFalse((root / "execution/raw-delivery-finished.json").exists())
            self.assertEqual(kill.call_count, 1)
            self.assertEqual(run.call_args.args[0][-1], "fyp-large-rank-stream")
            self.assertTrue(all("modal_large_ranks.py" not in call.args[0] for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
