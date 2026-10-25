"""Multi-account helpers: read modal.env (never printed) and build per-account Modal clients/env."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "modal.env"


def accounts() -> dict:
    """{n: (token_id, token_secret)} for every acN-token-id / acN-token-secret pair."""
    vals = {}
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip("\"'")
    out = {}
    for k, v in vals.items():
        if k.startswith("ac") and k.endswith("-token-id"):
            n = int(k[2:-len("-token-id")])
            sec = vals.get(f"ac{n}-token-secret")
            if sec:
                out[n] = (v, sec)
    return dict(sorted(out.items()))


def env_for(n: int, base=None) -> dict:
    tid, sec = accounts()[n]
    env = dict(base if base is not None else os.environ)
    env.update(MODAL_TOKEN_ID=tid, MODAL_TOKEN_SECRET=sec)
    env.pop("MODAL_PROFILE", None)
    return env


def client(n: int):
    import modal
    tid, sec = accounts()[n]
    return modal.Client.from_credentials(tid, sec)


def hf_token() -> str:
    for line in (ROOT / "hf.env").read_text().splitlines():
        k, _, v = line.removeprefix("export ").partition("=")
        if k.strip() == "HF_TOKEN":
            return v.strip().strip("\"'")
    raise KeyError("HF_TOKEN missing from hf.env")
