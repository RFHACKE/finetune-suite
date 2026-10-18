"""Temporary authenticated, CPU-only streaming of immutable result archives.

Deploy only with explicit authorization: modal deploy tools/modal_stream.py
Stop the deployment after collection. The access token stays in ignored
.modal-state/stream-token and is supplied only as an ephemeral Modal secret.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import threading
from urllib.parse import unquote, urlsplit

import modal

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("FYP_MODAL_RUN", "frw-20261018")
APP_NAME = os.environ.get("FYP_STREAM_APP", "fyp-archive-stream")
vault = modal.Volume.from_name("fyp-finetune-archive")
app = modal.App(APP_NAME, include_source=False)


def access_secret():
    if not modal.is_local():
        return modal.Secret.from_dict({})
    path = ROOT / ".modal-state" / "stream-token"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_urlsafe(32))
        path.chmod(0o600)
    return modal.Secret.from_dict({"FYP_STREAM_TOKEN": path.read_text().strip()})


image = (modal.Image.debian_slim(python_version="3.11")
         .env({"PYTHONPATH": "/app", "FYP_MODAL_RUN": RUN, "FYP_STREAM_APP": APP_NAME})
         .add_local_file(__file__, "/app/modal_stream.py", copy=True))


@app.function(image=image, volumes={"/vault": vault}, secrets=[access_secret()],
              cpu=(0.125, 1), memory=(256, 1024), timeout=1800,
              max_containers=16, scaledown_window=5)
@modal.concurrent(max_inputs=16)
@modal.web_server(8000)
def stream_archives():
    threading.Thread(target=start_server, daemon=True).start()


@app.function(image=image, volumes={"/vault": vault}, secrets=[access_secret()],
              cpu=(0.125, 1), memory=(256, 1024), timeout=1800,
              max_containers=1, scaledown_window=5,
              region="ap-south", routing_region="ap-south")
@modal.concurrent(max_inputs=16)
@modal.web_server(8000)
def stream_archives_ap():
    threading.Thread(target=start_server, daemon=True).start()


def start_server():
    guard = threading.Lock()
    active = 0
    token = os.environ["FYP_STREAM_TOKEN"]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def reject(self, code):
            self.send_response(code)
            self.send_header("Content-Length", "0")
            if code == 503:
                self.send_header("Retry-After", "5")
            self.end_headers()

        def do_GET(self):
            nonlocal active
            if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                return self.reject(401)
            relative = PurePosixPath(unquote(urlsplit(self.path).path).lstrip("/"))
            if (len(relative.parts) != 4 or ".." in relative.parts
                    or relative.parts[:2] != (RUN, "archives")
                    or not relative.name.endswith(".tar.zst")):
                return self.reject(404)
            path = Path("/vault") / str(relative)
            with guard:
                if not path.is_file():
                    if active:
                        return self.reject(503)
                    try:
                        vault.reload()
                    except Exception:
                        return self.reject(503)
                if not path.is_file():
                    return self.reject(404)
                active += 1
            try:
                size = path.stat().st_size
                first, last = 0, size - 1
                requested = self.headers.get("Range")
                if requested:
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                    if not match:
                        return self.reject(416)
                    first = int(match[1])
                    last = min(int(match[2]) if match[2] else last, last)
                    if first > last:
                        return self.reject(416)
                self.send_response(206 if requested else 200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(last - first + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "private, no-store")
                if requested:
                    self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
                self.end_headers()
                with path.open("rb") as source:
                    source.seek(first)
                    remaining = last - first + 1
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise OSError("Archive changed during streaming")
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with guard:
                    active -= 1

    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
