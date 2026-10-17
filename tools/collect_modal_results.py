"""Incrementally retrieve immutable completed runs; never start cloud compute.

Run with the Python environment containing modal. An optional separate CPU
analysis process can consume execution/retrieval.json's verified run IDs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import time
from urllib.parse import quote, urlsplit

import modal

MODELS = ["llama3.2-1b", "qwen2.5-1.5b", "granite3.3-2b", "qwen2.5-3b"]


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


class ArchiveStream:
    """Optional user-authorized HTTPS transport; cloud archive SHA stays binding."""
    def __init__(self, url, access_file):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".modal.run"):
            raise ValueError("Use the HTTPS URL of the authorized Modal archive endpoint")
        self.url = url.rstrip("/")
        self.token = Path(access_file).read_text().strip()
        self.curl = shutil.which("curl")
        if not self.curl or len(self.token) < 32:
            raise ValueError("Archive streaming requires curl and the generated access token file")

    def download(self, remote, entry, destination, expected_sha):
        if destination.exists() and destination.stat().st_size >= entry.size:
            if destination.stat().st_size == entry.size and sha256(destination) == expected_sha:
                return
            destination.unlink()
        with tempfile.TemporaryDirectory(prefix="fyp-stream-") as temporary:
            config = Path(temporary) / "curl.cfg"
            config.write_text("url = " + json.dumps(self.url + quote(remote, safe="/")) + "\nheader = "
                              + json.dumps("Authorization: Bearer " + self.token) + "\n")
            os.chmod(config, 0o600)
            for attempt in range(3):
                result = subprocess.run([self.curl, "--fail", "--silent", "--show-error", "--location",
                                         "--proto", "=https", "--proto-redir", "=https", "--max-redirs", "5",
                                         "--connect-timeout", "30", "--speed-time", "60", "--speed-limit", "1024",
                                         "--continue-at", "-", "--config", str(config),
                                         "--output", str(destination)], capture_output=True)
                if not result.returncode:
                    break
                if result.returncode == 33:
                    destination.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(3)
            if result.returncode:
                raise OSError(f"Authenticated archive stream failed (curl exit {result.returncode})")
        if destination.stat().st_size != entry.size or sha256(destination) != expected_sha:
            destination.unlink(missing_ok=True)
            raise ValueError("Streamed archive failed the independent cloud size/SHA-256 check")


class Collector:
    def __init__(self, target, run, volume, workers, prefer_archives=False,
                 archive_volume="fyp-finetune-archive", archive_url="", access_file=".modal-state/stream-token"):
        self.root = Path(target).resolve()
        self.prefix = PurePosixPath("/") / run
        self.volume = modal.Volume.from_name(volume)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.prefer_archives = prefer_archives
        self.archive_volume = modal.Volume.from_name(archive_volume) if prefer_archives else None
        self.stream = ArchiveStream(archive_url, access_file) if archive_url else None
        self.archive_checksums = {}
        self.state_path = self.root / "execution" / "retrieval.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "run": run, "volume": volume, "files": {}, "verified_runs": [], "training_finished": False}
        if self.state.get("run", run) != run or self.state.get("volume", volume) != volume:
            raise ValueError("Use a new target directory for a different cloud cohort")

    def read_json(self, remote, volume=None):
        return json.loads(b"".join((volume or self.volume).read_file(str(remote))))

    def entries(self, prefix, volume=None):
        try:
            return [e for e in (volume or self.volume).iterdir(str(prefix), recursive=True)
                    if e.type.name == "FILE" and not e.path.endswith(".lock")
                    and not e.path.endswith("/resume.pt")]
        except (FileNotFoundError, modal.exception.NotFoundError):
            return []

    def fetch(self, job):
        entry, target = job
        key = entry.path
        previous = self.state["files"].get(key, {})
        version = [entry.size, entry.mtime]
        if target.exists() and previous.get("remote_version") == version:
            return key, previous
        expected_archive = self.archive_checksums.get(str(target))
        if (expected_archive and target.exists() and target.stat().st_size == entry.size
                and sha256(target) == expected_archive):
            return key, {"remote_version": version, "sha256_at_retrieval": expected_archive,
                         "local_path": str(target.relative_to(self.root)), "bytes": entry.size}
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".download")
        h, size = hashlib.sha256(), 0
        remote = PurePosixPath("/" + key.lstrip("/"))
        archive_prefix = self.prefix / "archives"
        if self.stream and remote.is_relative_to(archive_prefix) and key.endswith(".tar.zst"):
            self.stream.download(str(remote), entry, temp, self.archive_checksums[str(target)])
            size = temp.stat().st_size
            checksum = sha256(temp)
        else:
            source_volume = self.archive_volume if self.prefer_archives and remote.is_relative_to(archive_prefix) else self.volume
            with temp.open("wb") as out:
                for chunk in source_volume.read_file(str(remote)):
                    out.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            checksum = h.hexdigest()
        if size != entry.size:
            temp.unlink(missing_ok=True)
            raise OSError(f"Remote file changed during retrieval: {key}")
        if target.suffix == ".json":
            try:
                json.loads(temp.read_text())
            except json.JSONDecodeError as exc:
                temp.unlink(missing_ok=True)
                raise OSError(f"Remote JSON was updated during retrieval: {key}") from exc
        os.replace(temp, target)
        return key, {"remote_version": version, "sha256_at_retrieval": checksum,
                     "local_path": str(target.relative_to(self.root)), "bytes": size}

    def extract(self, path, manifest):
        if sha256(path) != manifest["archive_sha256"]:
            raise ValueError("Compressed archive checksum mismatch")
        command = shutil.which("zstd")
        if not command:
            raise ValueError("Install the zstd CLI before using --prefer-archives")
        decoder = subprocess.Popen([command, "-d", "--long=30", "-q", "-c", str(path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=decoder.stdout, mode="r|") as archive:
                for member in archive:
                    parts = PurePosixPath(member.name).parts
                    valid_run = len(parts) >= 2 and parts[:2] == ("runs", manifest["run_id"])
                    valid_base = manifest.get("includes_base_projections") and parts[:2] == ("cache", "projections")
                    if (not (valid_run or valid_base) or ".." in parts or member.name.startswith("/")
                            or not (member.isdir() or member.isfile())):
                        raise ValueError("Archive contains an unexpected path/type")
                    archive.extract(member, path=self.root, set_attrs=False)
            decoder.stdout.close()
            error = decoder.stderr.read().decode(errors="replace")
            if decoder.wait():
                raise OSError(f"Archive decompression failed: {error}")
        finally:
            if decoder.poll() is None:
                decoder.terminate()
                decoder.wait()

    def verify_candidates(self, candidates, verified):
        for run_id, result in candidates.items():
            if run_id in verified:
                continue
            run_dir = self.root / "runs" / run_id
            try:
                base = self.root / "cache" / "projections" / result["model"] / result["hf_revision"] / "base.pt"
                base_meta = json.loads(base.with_suffix(".json").read_text())
                based = self.state.setdefault("verified_base", {})
                if based.get(result["model"]) != base_meta["sha256"]:
                    if sha256(base) != base_meta["sha256"]:
                        raise ValueError("Pretrained projection checksum mismatch")
                    based[result["model"]] = base_meta["sha256"]
                if all(sha256(run_dir / b["path"] / "projections.pt") == b["projections_sha256"]
                       for b in result["boundaries"]):
                    verified.add(run_id)
            except FileNotFoundError:
                pass

    def cycle(self):
        jobs, candidates, archives = {}, {}, {}
        verified = set(self.state["verified_runs"])
        for folder in ("defaults", "calibration", "benchmark", "execution"):
            prefix = self.prefix / folder
            for entry in self.entries(prefix):
                rel = PurePosixPath("/" + entry.path.lstrip("/")).relative_to(prefix)
                target = self.root / ("" if folder == "defaults" else folder) / str(rel)
                jobs[str(target)] = (entry, target)
        protocol = None
        for model in MODELS:
            prefix = self.prefix / "main" / model
            entries = self.entries(prefix)
            paths = {str(PurePosixPath("/" + e.path.lstrip("/")).relative_to(prefix)): e
                     for e in entries}
            if "protocol.json" not in paths:
                continue
            controls = self.read_json(prefix / "protocol.json")
            current_protocol = self.root / "protocol.json"
            if current_protocol.exists() and json.loads(current_protocol.read_text()) != controls:
                raise ValueError("Local results already have different scientific controls")
            if protocol is not None and controls != protocol:
                raise ValueError("Remote main workers have conflicting scientific controls")
            protocol = controls
            ready = set()
            for rel, entry in paths.items():
                parts = PurePosixPath(rel).parts
                if len(parts) == 3 and parts[0] == "runs" and parts[2] == "result.json":
                    local_result = self.root / rel
                    result = (json.loads(local_result.read_text()) if parts[1] in verified and local_result.exists()
                              else self.read_json(prefix / rel))
                    if result.get("status") != "completed":
                        continue
                    if result["protocol_sha256"] != canonical(controls):
                        raise ValueError("Result protocol does not match worker controls")
                    required = [str(PurePosixPath("runs") / parts[1] / b["path"] / "projections.pt")
                                for b in result["boundaries"]]
                    if all(path in paths for path in required):
                        ready.add(parts[1])
                        candidates[parts[1]] = result
            for rel, entry in paths.items():
                parts = PurePosixPath(rel).parts
                if parts[0] == "runs":
                    if self.prefer_archives:
                        continue
                    if parts[1] not in ready or parts[-1] == "resume.pt":
                        continue
                    target = self.root / rel
                elif parts[0] in ("cache", "hardware", "data", "invocations", "batches"):
                    if self.prefer_archives and parts[0] == "cache":
                        continue
                    target = self.root / rel
                elif rel in ("protocol.json", "software.json"):
                    target = self.root / rel
                elif parts[0] == "logs":
                    target = self.root / "logs" / model / str(PurePosixPath(*parts[1:]))
                elif rel.startswith("worker-"):
                    target = self.root / "execution" / f"main-{model}-{rel}"
                elif len(parts) == 1 and rel.startswith("failures") and rel.endswith(".jsonl"):
                    target = self.root / f"failures-{model}-{rel}"
                else:
                    continue
                jobs[str(target)] = (entry, target)
        if self.prefer_archives:
            extracted = self.state.setdefault("extracted_archives", {})
            for model in MODELS:
                prefix = self.prefix / "archives" / model
                entries = self.entries(prefix, self.archive_volume)
                indexed = {PurePosixPath(e.path).name: e for e in entries}
                for name, entry in indexed.items():
                    if not name.endswith(".json") or name[:-5] in verified:
                        continue
                    manifest = self.read_json(prefix / name, self.archive_volume)
                    if protocol is None or manifest["protocol_sha256"] != canonical(protocol):
                        raise ValueError("Archive protocol does not match the current cohort")
                    target = self.root / "execution" / "archive-manifests" / model / name
                    jobs[str(target)] = (entry, target)
                    if extracted.get(manifest["run_id"]) == manifest["archive_sha256"]:
                        continue
                    archive_entry = indexed.get(manifest["archive"])
                    if archive_entry:
                        target = self.root / "execution" / "incoming" / manifest["archive"]
                        self.archive_checksums[str(target)] = manifest["archive_sha256"]
                        jobs[str(target)] = (archive_entry, target)
                        archives[manifest["run_id"]] = (target, manifest)
        failures = []
        archives_by_path = {str(path.relative_to(self.root)): (run_id, path, manifest)
                            for run_id, (path, manifest) in archives.items()}
        futures = [self.pool.submit(self.fetch, job) for job in jobs.values()]
        for future in as_completed(futures):
            try:
                key, record = future.result()
                self.state["files"][key] = record
                ready_archive = archives_by_path.get(record["local_path"])
                if ready_archive:
                    run_id, path, manifest = ready_archive
                    self.extract(path, manifest)
                    self.state["extracted_archives"][run_id] = manifest["archive_sha256"]
                    path.unlink()
                    before = len(verified)
                    self.verify_candidates(candidates, verified)
                    # Publish each complete archive immediately. A slower
                    # unrelated download must not hold up local CPU analysis.
                    self.state["verified_runs"] = sorted(verified)
                    atomic_json(self.state_path, self.state)
                    if len(verified) != before:
                        print(json.dumps({"archive_ready": run_id,
                                          "verified_completed_runs": len(verified)}), flush=True)
            except Exception as exc:
                failures.append(repr(exc))
        self.verify_candidates(candidates, verified)
        monitor = self.root / "execution" / "main-workers.json"
        try:
            journal = json.loads(monitor.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            journal = {}
        self.state.update({"verified_runs": sorted(verified), "last_cycle_unix": time.time(),
                           "training_finished": len(journal.get("returned", {})) == len(MODELS),
                           "last_cycle_errors": failures})
        atomic_json(self.state_path, self.state)
        status = {"verified_completed_runs": len(verified), "retrieved_files": len(self.state["files"]),
                  "training_finished": self.state["training_finished"], "errors": failures,
                  "pending_completed_runs": len(set(candidates) - verified)}
        print(json.dumps(status), flush=True)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Finetune-result")
    parser.add_argument("--run", default="frw-20261018")
    parser.add_argument("--volume", default="fyp-finetune-results")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--prefer-archives", action="store_true", help="Use lossless CPU-packaged run archives")
    parser.add_argument("--archive-volume", default="fyp-finetune-archive")
    parser.add_argument("--archive-url", default="", help="Explicitly authorized authenticated Modal archive endpoint")
    parser.add_argument("--access-file", default=".modal-state/stream-token", help="Local archive endpoint token file")
    args = parser.parse_args()
    if args.workers < 1 or args.interval < 5:
        parser.error("workers must be positive and interval at least five seconds")
    if args.archive_url and not args.prefer_archives:
        parser.error("--archive-url requires --prefer-archives")
    collector = Collector(args.target, args.run, args.volume, args.workers, args.prefer_archives,
                          args.archive_volume, args.archive_url, args.access_file)
    while True:
        status = collector.cycle()
        if not args.watch or (status["training_finished"] and not status["errors"]
                              and not status["pending_completed_runs"]):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
