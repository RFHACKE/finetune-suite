"""Append timestamped archive-download progress every ten minutes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import threading

INTERVAL_SECONDS = 600


def snapshot(root, catalog_path):
    root = Path(root)
    catalog = json.loads(Path(catalog_path).read_text())
    state_path = root / "execution/retrieval.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    verified = set(state.get("verified_runs", []))
    extracted = state.get("extracted_archives", {})
    downloaded, total = 0, 0
    records = catalog["archives"]
    for record in records:
        size = record["archive_bytes"]
        total += size
        run_id = record["run_id"]
        if run_id in verified or extracted.get(run_id) == record["archive_sha256"]:
            # Verified archives are deleted after extraction; keep counting them.
            downloaded += size
            continue
        # Segmented transfers can make sparse files whose logical length is
        # already the final size. Count allocated bytes, capped at archive size.
        received = 0
        archive_path = root / "execution/incoming" / record["archive"]
        for path in (archive_path, archive_path.with_name(archive_path.name + ".download")):
            try:
                info = path.stat()
            except FileNotFoundError:
                continue
            received = max(received, min(size, info.st_size, info.st_blocks * 512))
        downloaded += received
    return {"downloaded_bytes": downloaded, "total_bytes": total,
            "verified_runs": len(verified & {r["run_id"] for r in records}),
            "expected_runs": len(records), "complete": bool(state.get("retrieval_finished"))}


class DownloadProgressLog:
    def __init__(self, root, catalog=None):
        self.root = Path(root)
        self.catalog = Path(catalog) if catalog else self.root / "execution/delivery-catalog.json"
        self.path = self.root / "execution/download-progress.log"
        self.stop = threading.Event()
        self.thread = None

    def _write(self, log, progress, now):
        total = progress["total_bytes"]
        percent = 100 * progress["downloaded_bytes"] / total if total else 0
        log.write(f"{now.isoformat(timespec='seconds').replace('+00:00', 'Z')}  "
                  f"{progress['downloaded_bytes'] / 1e9:.2f} GB / {total / 1e9:.2f} GB  "
                  f"({percent:.1f}%)  verified={progress['verified_runs']}/{progress['expected_runs']}"
                  + ("  complete" if progress["complete"] else "") + "\n")
        log.flush()

    def run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+") as log:
            try:
                # The standalone observer and a restarted downloader may overlap.
                # Exactly one process owns this cohort's progress log.
                fcntl.flock(log, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            log.seek(0, 2)
            end = log.tell()
            log.seek(max(0, end - 2048))
            lines = log.read().splitlines()
            last_time = None
            already_complete = False
            if lines:
                try:
                    last_time = datetime.fromisoformat(lines[-1].split()[0].replace("Z", "+00:00"))
                    already_complete = lines[-1].endswith("  complete")
                except (ValueError, IndexError):
                    pass
            while True:
                try:
                    progress = snapshot(self.root, self.catalog)
                    now = datetime.now(timezone.utc)
                    remaining = INTERVAL_SECONDS - (now - last_time).total_seconds() if last_time else 0
                    if remaining <= 0 or (progress["complete"] and not already_complete):
                        self._write(log, progress, now)
                        last_time, remaining = now, INTERVAL_SECONDS
                    if progress["complete"] or self.stop.is_set():
                        return
                except (OSError, ValueError, KeyError) as exc:
                    with self.path.with_name("download-progress-errors.log").open("a") as errors:
                        errors.write(f"{datetime.now(timezone.utc).isoformat()}  {exc!r}\n")
                    if self.stop.is_set():
                        return
                    remaining = 30
                # A stop wakes the thread so a successful transfer gets a final
                # completion entry even between scheduled ten-minute entries.
                self.stop.wait(max(1, remaining))

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, name="download-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Finetune-result")
    parser.add_argument("--catalog", help="Default: TARGET/execution/delivery-catalog.json")
    args = parser.parse_args()
    DownloadProgressLog(args.target, args.catalog).run()


if __name__ == "__main__":
    main()
