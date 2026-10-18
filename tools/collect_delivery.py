"""Retrieve a finished cohort with segmented, checksum-verified downloads.

Requires the authenticated archive endpoint and CPU-generated delivery catalog.
The frozen scientific implementation is unchanged; this handles transport only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
from urllib.parse import quote

from tools.collect_modal_results import ArchiveStream, Collector, atomic_json, canonical, sha256
from tools.download_progress import DownloadProgressLog


class SegmentedStream(ArchiveStream):
    def __init__(self, url, access_file, executable, connections):
        super().__init__(url, access_file)
        self.executable = str(Path(executable).resolve())
        self.connections = connections
        self.stop_requested = threading.Event()
        self.process_lock = threading.Lock()
        self.processes = set()
        if not Path(self.executable).is_file():
            raise ValueError("Install aria2c or provide its executable with --aria2")

    def download(self, remote, entry, destination, expected_sha):
        if self.stop_requested.is_set():
            raise InterruptedError("Download paused; partial data and resume controls retained")
        control = destination.with_name(destination.name + ".aria2")
        if destination.exists() and not control.exists() and destination.stat().st_size >= entry.size:
            if destination.stat().st_size == entry.size and sha256(destination) == expected_sha:
                return
            destination.unlink()
        environment = os.environ.copy()
        library = Path(self.executable).parents[1] / "lib/x86_64-linux-gnu"
        if library.is_dir():
            environment["LD_LIBRARY_PATH"] = str(library) + ":" + environment.get("LD_LIBRARY_PATH", "")
        with tempfile.TemporaryDirectory(prefix="fyp-aria2-") as temporary:
            config = Path(temporary) / "input.txt"
            config.write_text(self.url + quote(remote, safe="/") + "\n"
                              + "  header=Authorization: Bearer " + self.token + "\n"
                              + "  dir=" + str(destination.parent) + "\n"
                              + "  out=" + destination.name + "\n"
                              + "  checksum=sha-256=" + expected_sha + "\n")
            config.chmod(0o600)
            command = [self.executable, "--continue=true", "--check-integrity=true",
                       "--allow-overwrite=true", "--auto-file-renaming=false", "--file-allocation=none",
                       "--auto-save-interval=10",
                       "--check-certificate=true",
                       f"--max-connection-per-server={self.connections}", f"--split={self.connections}",
                       "--min-split-size=1M", "--max-tries=8", "--retry-wait=3", "--connect-timeout=30",
                       "--timeout=300", "--lowest-speed-limit=0", "--summary-interval=0",
                       "--download-result=hide", "--console-log-level=error", "--quiet=true",
                       "--input-file=" + str(config)]
            with self.process_lock:
                if self.stop_requested.is_set():
                    raise InterruptedError("Download paused; partial data retained")
                result = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.processes.add(result)
            try:
                stdout, stderr = result.communicate()
            finally:
                with self.process_lock:
                    self.processes.discard(result)
        if result.returncode:
            diagnostic = (stdout + stderr).decode(errors="replace").replace(self.token, "[redacted]")
            destination.with_name(destination.name + ".error.log").write_text(diagnostic)
            raise OSError(f"Segmented archive download failed (aria2 exit {result.returncode}); partial data retained")
        if destination.stat().st_size != entry.size or sha256(destination) != expected_sha:
            raise ValueError("Segmented archive failed the independent cloud size/SHA-256 check")

    def stop(self):
        self.stop_requested.set()
        with self.process_lock:
            for process in self.processes:
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass


@contextmanager
def download_signals(stream):
    def pause(_signum, _frame):
        stream.stop()
    previous = {sig: signal.signal(sig, pause) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def active_download(root):
    """Also recognize collectors started before the per-target lock existed."""
    root = Path(root).resolve()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode().split("\0")
            if "tools.collect_delivery" not in args:
                continue
            target = args[args.index("--target") + 1] if "--target" in args else "Finetune-result"
            if (process / "cwd" / target).resolve() == root:
                return int(process.name)
        except (OSError, ValueError, UnicodeError):
            continue
    return None


@contextmanager
def download_lock(root):
    path = Path(root) / "execution/download.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A downloader is already using this target; follow execution/download-progress.log") from None
        pid = active_download(root)
        if pid:
            raise SystemExit(f"This download is already running (PID {pid}); follow execution/download-progress.log")
        yield


def saved_options(args):
    root = Path(args.target).resolve()
    path = root / "execution/download-settings.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    args.catalog = args.catalog or saved.get("catalog", str(root / "execution/delivery-catalog.json"))
    defaults = {"archive_url": None, "archive_volume": "fyp-finetune-archive",
                "access_file": ".modal-state/stream-token", "aria2": ".bootstrap/aria2/usr/bin/aria2c",
                "workers": 16, "connections": 16}
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, saved.get(name, default))
    return args


def save_settings(args, run):
    # Save paths/settings, never the bearer token itself.
    settings = {name: getattr(args, name) for name in
                ("archive_url", "archive_volume", "workers", "connections")}
    settings.update({name: str(Path(getattr(args, name)).resolve()) for name in
                     ("catalog", "access_file", "aria2")})
    atomic_json(Path(args.target) / "execution/download-settings.json",
                {"schema": "frw-download-settings-v1", "run": run, **settings})


def extract_metadata(archive_path, root):
    """Extract checked metadata, preserving already completed CPU diagnostics."""
    command = shutil.which("zstd")
    if not command:
        raise ValueError("Install zstd before retrieving delivery archives")
    allowed = {"lr_defaults.json", "lr_test", "calibration", "benchmark", "execution", "runs",
               "protocol.json", "software.json", "data", "hardware", "invocations", "batches", "logs",
               "overfitting"}
    decoder = subprocess.Popen([command, "-d", "--long=30", "-q", "-c", str(archive_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=decoder.stdout, mode="r|") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if not path.parts:
                    continue
                if (path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir())
                        or (path.parts[0] not in allowed and not path.parts[0].startswith("failures-"))):
                    raise ValueError("Unexpected metadata archive member")
                destination = root / str(path)
                if path.parts[0] == "runs" and path.name == "result.json" and destination.exists():
                    current = json.loads(destination.read_text())
                    if current.get("intruder_status") == "completed":
                        continue
                archive.extract(member, path=root, set_attrs=False)
        decoder.stdout.close()
        error = decoder.stderr.read().decode(errors="replace")
        if decoder.wait():
            raise OSError("Metadata decompression failed: " + error)
    finally:
        if decoder.poll() is None:
            decoder.terminate()
            decoder.wait()
        decoder.stdout.close()
        decoder.stderr.close()


def collect(args):
    catalog = json.loads(Path(args.catalog).read_text())
    if catalog.get("schema") != "frw-completed-delivery-v1" or not catalog.get("training_finished"):
        raise ValueError("Use the CPU-generated catalog for a finished cohort")
    records = catalog["archives"]
    if len(records) != catalog["expected_runs"] or len({r["run_id"] for r in records}) != len(records):
        raise ValueError("Incomplete or duplicated delivery catalog")
    root = Path(args.target).resolve()
    state_path = root / "execution/retrieval.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("run", catalog["run"]) != catalog["run"]:
        raise ValueError("Use a separate target directory for a different cloud cohort")
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and canonical(json.loads(protocol_path.read_text())) != catalog["protocol_sha256"]:
        raise ValueError("Local scientific controls differ from the delivery catalog")
    if (state.get("retrieval_finished") and protocol_path.exists()
            and set(state.get("verified_runs", [])) == {r["run_id"] for r in records}
            and state.get("metadata_sha256") == catalog["metadata"]["sha256"]):
        print(json.dumps({"retrieval_finished": True, "already_complete": True,
                          "verified_runs": len(records), "network_downloads": 0}), flush=True)
        return
    if not args.archive_url:
        raise ValueError("Supply --archive-url once; subsequent restarts reuse execution/download-settings.json")
    client = Collector(args.target, catalog["run"], "fyp-finetune-results", args.workers,
                       prefer_archives=True, archive_volume=args.archive_volume)
    try:
        client.stream = SegmentedStream(args.archive_url, args.access_file, args.aria2, args.connections)
        save_settings(args, catalog["run"])
        with download_signals(client.stream):
            _collect_into(client, args, catalog)
    finally:
        client.pool.shutdown(wait=True)


def _collect_into(client, args, catalog):
    records = catalog["archives"]
    existing_protocol = client.root / "protocol.json"
    if existing_protocol.exists() and canonical(json.loads(existing_protocol.read_text())) != catalog["protocol_sha256"]:
        raise ValueError("Local scientific controls differ from the delivery catalog")
    client.state.update({"expected_completed_runs": len(records), "training_finished": True,
                         "retrieval_finished": False, "transport": "segmented_lossless_archives"})
    atomic_json(client.state_path, client.state)
    metadata = catalog["metadata"]
    incoming = client.root / "execution/incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    if client.state.get("metadata_sha256") != metadata["sha256"] or not existing_protocol.exists():
        destination = incoming / "metadata.tar.zst.download"
        client.stream.download("/" + metadata["remote_path"], SimpleNamespace(size=metadata["bytes"]),
                               destination, metadata["sha256"])
        extract_metadata(destination, client.root)
        destination.unlink()
        client.state["metadata_sha256"] = metadata["sha256"]
        atomic_json(client.state_path, client.state)
    if canonical(json.loads(existing_protocol.read_text())) != catalog["protocol_sha256"]:
        raise ValueError("Extracted controls disagree with catalog")
    expected = json.loads((client.root / "execution/unattended/expected-grid.json").read_text())
    if {r["run_id"] for r in expected} != {r["run_id"] for r in records}:
        raise ValueError("Delivery catalog differs from the preserved requested grid")
    verified = set(client.state["verified_runs"])
    candidates = {r["run_id"]: json.loads((client.root / "runs" / r["run_id"] / "result.json").read_text())
                  for r in records}
    pending = sorted((r for r in records if r["run_id"] not in verified),
                     key=lambda r: (not r["includes_base_projections"], r["archive_bytes"], r["model"]))
    futures = {}
    for record in pending:
        path = incoming / record["archive"]
        client.archive_checksums[str(path)] = record["archive_sha256"]
        entry = SimpleNamespace(path=record["remote_path"], size=record["archive_bytes"], mtime=0)
        futures[client.pool.submit(client.fetch, (entry, path))] = (record, path)
    errors = []
    extracted_pending, extracted_paths = {}, {}
    start = time.monotonic()
    print(json.dumps({"metadata_ready": True, "verified_completed_runs": len(verified),
                      "queued_archives": len(pending), "connections_per_archive": args.connections}), flush=True)
    for future in as_completed(futures):
        record, path = futures[future]
        try:
            key, file_record = future.result()
            client.state["files"][key] = file_record
            client.extract(path, record)
            client.state.setdefault("extracted_archives", {})[record["run_id"]] = record["archive_sha256"]
            extracted_pending[record["run_id"]] = candidates[record["run_id"]]
            extracted_paths[record["run_id"]] = path
            # A small run archive may arrive before its model's much larger
            # pretrained cache archive. Keep it pending rather than treating
            # that normal dependency as a corrupt or failed download.
            client.verify_candidates(extracted_pending, verified)
            # Commit verified identities before deleting their compressed
            # copies, so an interruption cannot force a verified re-download.
            client.state["verified_runs"] = sorted(verified)
            atomic_json(client.state_path, client.state)
            for ready in set(extracted_pending) & verified:
                extracted_paths.pop(ready).unlink(missing_ok=True)
                del extracted_pending[ready]
            print(json.dumps({"archive_ready": record["run_id"], "verified_completed_runs": len(verified),
                              "elapsed_s": time.monotonic() - start}), flush=True)
        except Exception as exc:
            errors.append({"run_id": record["run_id"], "error": repr(exc)})
            print(json.dumps(errors[-1]), flush=True)
        client.state.update({"verified_runs": sorted(verified), "last_cycle_unix": time.time(),
                             "last_cycle_errors": errors})
        atomic_json(client.state_path, client.state)
    if extracted_pending:
        client.verify_candidates(extracted_pending, verified)
        client.state["verified_runs"] = sorted(verified)
        atomic_json(client.state_path, client.state)
        for run_id in extracted_pending:
            if run_id in verified:
                extracted_paths[run_id].unlink(missing_ok=True)
            else:
                errors.append({"run_id": run_id, "error": "Saved projection/base checksum validation incomplete"})
    complete = len(verified) == len(records) and not errors
    client.state.update({"retrieval_finished": complete, "verified_runs": sorted(verified)})
    atomic_json(client.state_path, client.state)
    atomic_json(client.root / "execution/delivery-download-errors.json", errors)
    if not complete:
        raise SystemExit("Delivery incomplete; partial downloads retained and failures recorded for resumption")
    print(json.dumps({"retrieval_finished": True, "verified_runs": len(verified)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", help="Default: TARGET/execution/delivery-catalog.json")
    parser.add_argument("--target", default="Finetune-result")
    parser.add_argument("--archive-volume")
    parser.add_argument("--archive-url", help="Required only until endpoint settings have been saved")
    parser.add_argument("--access-file")
    parser.add_argument("--aria2")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--connections", type=int)
    args = saved_options(parser.parse_args())
    if args.workers < 1 or not 1 <= args.connections <= 16:
        parser.error("workers must be positive; connections must be in [1,16]")
    with download_lock(args.target), DownloadProgressLog(args.target, args.catalog):
        collect(args)


if __name__ == "__main__":
    main()
