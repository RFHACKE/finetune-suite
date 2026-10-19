"""Transport changes preserve bytes, credentials and completed CPU results."""
import hashlib
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import signal
import subprocess
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec("modal") is None:
    raise unittest.SkipTest("Optional Modal client SDK is required for cloud-helper tests")

from tools.collect_delivery import SegmentedStream, collect, extract_metadata
from tools.collect_modal_results import Collector, canonical


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
                self.assertIn("--continue=true", command)
                self.assertIn("--auto-save-interval=10", command)
                target.write_bytes(data)
                return Mock(returncode=0, communicate=lambda: (b"", b""))
            with patch("tools.collect_delivery.subprocess.Popen", side_effect=download):
                stream.download("/cohort/archives/model/run.tar.zst", SimpleNamespace(size=len(data)), target, expected)
            self.assertEqual(target.read_bytes(), data)

    def test_complete_delivery_needs_no_endpoint_or_cloud_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution = root / "execution"
            execution.mkdir()
            protocol = {"frozen": "controls"}
            (root / "protocol.json").write_text(json.dumps(protocol))
            catalog = {"schema": "frw-completed-delivery-v1", "run": "cohort", "training_finished": True,
                       "expected_runs": 1, "protocol_sha256": canonical(protocol),
                       "archives": [{"run_id": "verified"}], "metadata": {"sha256": "metadata"}}
            (execution / "delivery-catalog.json").write_text(json.dumps(catalog))
            state = {"run": "cohort", "verified_runs": ["verified"], "retrieval_finished": True,
                     "metadata_sha256": "metadata"}
            state_path = execution / "retrieval.json"
            state_path.write_text(json.dumps(state))
            original = state_path.read_bytes()
            args = SimpleNamespace(target=str(root), catalog=str(execution / "delivery-catalog.json"))
            with (patch("tools.collect_delivery.Collector", side_effect=AssertionError("Unexpected cloud client")),
                  redirect_stdout(io.StringIO()) as output):
                collect(args)
            self.assertTrue(json.loads(output.getvalue())["already_complete"])
            self.assertEqual(state_path.read_bytes(), original)

    def test_pause_during_native_start_retains_partial_and_signals_new_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "aria2c"
            executable.touch()
            token = root / "token"
            token.write_text("a-private-test-token-with-more-than-thirty-two-characters")
            target = root / "partial.download"
            target.write_bytes(b"saved")
            control = root / "partial.download.aria2"
            control.write_bytes(b"segments")
            stream = SegmentedStream("https://example.modal.run", token, executable, 1)
            diagnostic = ("Authorization: Bearer " + token.read_text()
                          + " URI=https://example.modal.run/file?__modal_attempt_token=private-attempt-token\n").encode()
            child = Mock(returncode=1, communicate=lambda: (diagnostic, b""))
            def start(*_args, **_kwargs):
                stream.stop()
                return child
            with (patch("tools.collect_delivery.subprocess.Popen", side_effect=start),
                  self.assertRaises(OSError)):
                stream.download("/cohort/archives/model/run.tar.zst", SimpleNamespace(size=100), target, "unused")
            child.send_signal.assert_called_once_with(signal.SIGINT)
            self.assertEqual(target.read_bytes(), b"saved")
            self.assertEqual(control.read_bytes(), b"segments")
            recorded = target.with_name(target.name + ".error.log").read_text()
            self.assertNotIn(token.read_text(), recorded)
            self.assertNotIn("private-attempt-token", recorded)

    def test_resume_keeps_partial_bytes_skips_verified_and_commits_before_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution = root / "execution"
            incoming = execution / "incoming"
            incoming.mkdir(parents=True)
            (execution / "unattended").mkdir()
            protocol = {"frozen": "controls"}
            (root / "protocol.json").write_text(json.dumps(protocol))
            records = [{"run_id": name, "archive": name + ".tar.zst", "remote_path":
                        "cohort/archives/model/" + name + ".tar.zst", "archive_bytes": 7,
                        "archive_sha256": hashlib.sha256(b"payload").hexdigest(),
                        "includes_base_projections": False, "model": "model"}
                       for name in ("verified", "pending")]
            catalog = {"schema": "frw-completed-delivery-v1", "run": "cohort", "training_finished": True,
                       "expected_runs": 2, "protocol_sha256": canonical(protocol),
                       "archives": records, "metadata": {"sha256": "metadata"}}
            catalog_path = execution / "delivery-catalog.json"
            catalog_path.write_text(json.dumps(catalog))
            (execution / "unattended/expected-grid.json").write_text(json.dumps(records))
            state_path = execution / "retrieval.json"
            state_path.write_text(json.dumps({"run": "cohort", "volume": "fyp-finetune-results", "files": {},
                "verified_runs": ["verified"], "retrieval_finished": False, "metadata_sha256": "metadata"}))
            for name in ("verified", "pending"):
                folder = root / "runs" / name
                folder.mkdir(parents=True)
                (folder / "result.json").write_text(json.dumps({"run_id": name, "status": "completed"}))
            partial = incoming / "pending.tar.zst.download"
            partial.write_bytes(b"pay")
            control = partial.with_name(partial.name + ".aria2")
            control.write_bytes(b"saved segments")
            def download(remote, entry, destination, expected_sha):
                self.assertEqual(destination, partial)
                self.assertEqual(destination.read_bytes(), b"pay")
                self.assertEqual(control.read_bytes(), b"saved segments")
                destination.write_bytes(b"payload")
            stream = SimpleNamespace(download=Mock(side_effect=download), stop=Mock(),
                                     stop_requested=threading.Event())
            args = SimpleNamespace(target=str(root), catalog=str(catalog_path), workers=1,
                archive_volume="archives", archive_url="https://example.modal.run", access_file="unused", aria2="unused",
                connections=1)
            def verify(_client, candidates, verified):
                verified.update(candidates)
            original_unlink = Path.unlink
            def interrupted_cleanup(path, *args, **kwargs):
                if path == incoming / "pending.tar.zst":
                    raise KeyboardInterrupt("Simulated interruption after verification")
                return original_unlink(path, *args, **kwargs)
            with (patch("tools.collect_modal_results.modal.Volume.from_name", return_value=SimpleNamespace()),
                  patch("tools.collect_delivery.SegmentedStream", return_value=stream),
                  patch.object(Collector, "extract"), patch.object(Collector, "verify_candidates", new=verify),
                  redirect_stdout(io.StringIO())):
                with patch.object(Path, "unlink", new=interrupted_cleanup), self.assertRaises(KeyboardInterrupt):
                    collect(args)
                self.assertEqual(json.loads(state_path.read_text())["verified_runs"], ["pending", "verified"])
                collect(args)
            stream.download.assert_called_once()
            self.assertTrue(json.loads(state_path.read_text())["retrieval_finished"])

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
            (root / "execution/delivery-catalog.json").write_text('{"archives":[]}')
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

    def test_supervisor_records_unexpected_exit_reason(self):
        from tools import watch_large_ranks as watcher
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "execution").mkdir()
            with (patch.object(watcher, "_collect_raw", side_effect=RuntimeError("billing fixture unavailable")),
                  self.assertRaises(RuntimeError)):
                watcher.collect_raw(root, root, {"run": "fixture"})
            recorded = json.loads((root / "execution/download-supervisor.json").read_text())
            self.assertEqual(recorded["phase"], "error")
            self.assertIn("billing fixture unavailable", recorded["traceback"])

    def test_stalled_transfer_retries_and_keeps_partial_data(self):
        from tools import watch_large_ranks as watcher
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            original = workspace / "Finetune-result/execution"
            original.mkdir(parents=True)
            (original / "retrieval.json").write_text('{"retrieval_finished":true}')
            (original / "active-budget.json").write_text('{"approved_total_usd":46}')
            root = workspace / "Finetune-result/large-ranks"
            execution = root / "execution"
            execution.mkdir(parents=True)
            (execution / "delivery-catalog.json").write_text('{"archives":[]}')
            partial = execution / "saved.download"
            partial.write_bytes(b"saved before retry")
            clock = [0]
            children = []
            def popen(*_args, **_kwargs):
                child = SimpleNamespace(pid=12345, returncode=None)
                child.poll = lambda: child.returncode
                def finish(timeout):
                    child.returncode = -2
                    return -2
                child.wait = Mock(side_effect=finish)
                children.append(child)
                return child
            def command(args, **_kwargs):
                return SimpleNamespace(returncode=0, stdout='{"metered_cost":"40"}' if "billing" in args else '')
            endpoint = SimpleNamespace(get_web_url=lambda: "https://isolated.modal.run")
            with (patch.object(watcher.subprocess, "run", side_effect=command),
                  patch.object(watcher.subprocess, "Popen", side_effect=popen),
                  patch.object(watcher.modal.Function, "from_name", return_value=endpoint),
                  patch.object(watcher.time, "monotonic", side_effect=lambda: clock[0]),
                  patch.object(watcher.time, "sleep", side_effect=lambda s: clock.__setitem__(0, clock[0]+s)),
                  patch.object(watcher.os, "killpg") as kill):
                watcher.collect_raw(workspace, root, {"run": "extension"})
            self.assertEqual(len(children), 3)
            self.assertEqual(kill.call_count, 3)
            self.assertEqual(partial.read_bytes(), b"saved before retry")
            recorded = json.loads((execution / "download-supervisor.json").read_text())
            self.assertEqual(recorded["phase"], "deferred")
            self.assertTrue(json.loads((execution / "raw-delivery-attempt.json").read_text())["stalled"])


if __name__ == "__main__":
    unittest.main()
