#!/usr/bin/env python3
"""Standalone Telegram notifier for long-running training batches.

This file is deliberately NOT part of the research suite: it uses only the
Python standard library, is never imported by rank_suite/, and is never
referenced by run.sh / setup.sh. Deleting notify.py (and run_notify.sh)
removes all notification behavior without affecting the project.

Configuration (environment variables, or a `notify.env` file next to this
script with KEY=VALUE lines):

    TG_BOT_TOKEN   token from @BotFather (https://t.me/BotFather)
    TG_CHAT_ID     your chat id (message @userinfobot, or use the
                   getUpdates snippet printed by: python3 notify.py chatid)

Usage:

    python3 notify.py test                 # send a test message
    python3 notify.py send "custom text"   # send an arbitrary message
    python3 notify.py chatid               # helper: print your chat id
    python3 notify.py run -- ./run.sh run --models qwen2.5-1.5b
                                           # run a command; notify on start,
                                           # completion, failure or interrupt

`notify.py run` streams the command's output unchanged (interactive prompts
work) and exits with the command's own exit code. Notification failures are
never fatal and never alter the wrapped command's behavior.

For an unattended VPS batch, detach it so the run survives SSH drops:

    nohup ./run_notify.sh run --models qwen2.5-1.5b --track both \
        > train.out 2>&1 &        # or: tmux/screen
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT = 20  # seconds; a dead network must never hang the batch


# ----------------------------------------------------------------- config --
def load_config() -> tuple:
    env_file = HERE / "notify.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    return os.environ.get("TG_BOT_TOKEN", "").strip(), \
        os.environ.get("TG_CHAT_ID", "").strip()


# ------------------------------------------------------------------- send --
def send(text: str) -> bool:
    """Best-effort Telegram message. Returns True on success; never raises."""
    token, chat = load_config()
    if not token or not chat:
        print("notify: TG_BOT_TOKEN / TG_CHAT_ID not set — message skipped",
              file=sys.stderr)
        return False
    data = urllib.parse.urlencode({
        "chat_id": chat,
        "text": text[:4000],          # Telegram hard limit
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(
        API.format(token=token, method="sendMessage"), data=data)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            payload = json.loads(resp.read().decode())
            if not payload.get("ok"):
                print(f"notify: API error: {payload}", file=sys.stderr)
                return False
            return True
    except Exception as exc:  # noqa: BLE001 - notifications are best-effort
        print(f"notify: send failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False


# ---------------------------------------------------------------- helpers --
def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def read_batch_summary(output_dir: str):
    path = Path(output_dir) / "batch_summary.json"
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def parse_output_dir(argv: list) -> str:
    for flag in ("--output", "-o"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
    return "results"


# --------------------------------------------------------------- commands --
def cmd_test() -> int:
    ok = send(f"🔔 notifier test from {socket.gethostname()} — "
              f"configuration OK")
    print("sent" if ok else "failed (see message above)")
    return 0 if ok else 1


def cmd_send(argv: list) -> int:
    if not argv:
        print("usage: notify.py send <message>", file=sys.stderr)
        return 2
    return 0 if send(" ".join(argv)) else 1


def cmd_chatid() -> int:
    token, _ = load_config()
    if not token:
        print("set TG_BOT_TOKEN first (or put it in notify.env)", file=sys.stderr)
        return 1
    req = urllib.request.Request(API.format(token=token, method="getUpdates"))
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        print(f"getUpdates failed: {exc}", file=sys.stderr)
        return 1
    chats = []
    for upd in payload.get("result", []):
        for where in ("message", "edited_message", "channel_post"):
            msg = upd.get(where) or {}
            chat = msg.get("chat") or {}
            if chat.get("id") is not None:
                chats.append((chat.get("id"), chat.get("title")
                              or chat.get("username") or chat.get("first_name")))
    if not chats:
        print("no chats found — send any message to your bot first, "
              "then re-run")
    for cid, name in dict.fromkeys(chats):
        print(f"chat id: {cid}   ({name})")
    return 0


def cmd_run(argv: list) -> int:
    if not argv:
        print("usage: notify.py run -- <command> [args...]", file=sys.stderr)
        return 2
    host = socket.gethostname()
    output_dir = parse_output_dir(argv)
    started = time.time()
    send(f"🚀 training started on {host}\n"
         f"$ {' '.join(argv)}\n"
         f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")

    interrupted = False
    exit_code = 1
    proc = subprocess.Popen(argv)   # inherit stdio: prompts/streaming work

    def _on_term(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_term)
    try:
        exit_code = proc.wait()
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        print("\nnotify: interrupting command...", file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()
        exit_code = proc.returncode if proc.returncode is not None else 1
    finally:
        elapsed = fmt_duration(time.time() - started)
        if interrupted:
            head = f"⚠️ training INTERRUPTED on {host}"
        elif exit_code == 0:
            head = f"✅ training COMPLETE on {host}"
        else:
            head = f"❌ training FAILED on {host} (exit {exit_code})"
        body = (f"$ {' '.join(argv)}\n"
                f"duration: {elapsed}\n"
                f"results: {output_dir}/")
        summary = read_batch_summary(output_dir)
        if summary:
            counts = ", ".join(
                f"{k} {v}" for k, v in summary.items()
                if k in ("completed", "skipped", "failed"))
            if counts:
                body += f"\nruns: {counts}"
        send(f"{head}\n{body}")

    return exit_code


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "run" and rest and rest[0] == "--":
        rest = rest[1:]
    handlers = {
        "test": lambda: cmd_test(),
        "send": lambda: cmd_send(rest),
        "chatid": lambda: cmd_chatid(),
        "run": lambda: cmd_run(rest),
    }
    if cmd not in handlers:
        print(f"unknown command: {cmd}\n{__doc__}", file=sys.stderr)
        return 2
    return handlers[cmd]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
