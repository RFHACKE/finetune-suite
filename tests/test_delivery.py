"""Transport changes preserve bytes, credentials and completed CPU results."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
