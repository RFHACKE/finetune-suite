"""Validated, cohort-local calibrated defaults. This module needs only stdlib."""
from __future__ import annotations

import math
from pathlib import Path
import re

from . import config, utils
from .protocol import digest, source_digest

SCHEMA = "frw-lr-defaults-v1"


def calibration_context(spec):
    return {"protocol": config.PROTOCOL_VERSION, "source_sha256": source_digest(),
            "models": config.MODEL_REVISIONS, "data_revision": config.DATA_REVISION,
            "encoding": config.ENCODING_VERSION, "max_len": config.MAX_LEN,
            "train_samples": spec.train_samples, "epochs": spec.epochs,
            "micro_batch": spec.micro_batch, "accum": spec.accum,
            "eval_batch": spec.eval_batch, "optimizer": spec.optimizer,
            "deterministic": spec.deterministic}


def load_defaults(spec):
    path = Path(spec.output_dir) / "lr_defaults.json"
    payload = utils.load_json(path)
    if payload is None:
        return {}
    signed = {k: v for k, v in payload.items() if k != "sha256"}
    if (payload.get("schema") != SCHEMA or payload.get("status") != "complete"
            or payload.get("sha256") != digest(signed)):
        raise ValueError(f"Invalid calibrated defaults: {path}")
    if payload.get("context") != calibration_context(spec):
        raise ValueError(f"Calibrated defaults have different code/training controls: {path}. "
                         "Use the original controls or a new --output directory.")
    choices = payload.get("selected", {})
    if not choices or any(m not in config.MODELS for m in choices):
        raise ValueError("Calibrated defaults contain invalid models")
    evidence = {}
    for model, tracks in choices.items():
        if not tracks or any(t not in ("lora", "full") for t in tracks):
            raise ValueError("Invalid calibrated adaptation track")
        for track, choice in tracks.items():
            for key in ("lr", "head_lr_multiplier"):
                value = choice.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid calibrated {key}")
            cal_id = choice.get("calibration_id", "")
            if not isinstance(cal_id, str) or not re.fullmatch(r"[a-f0-9]{20}", cal_id):
                raise ValueError("Calibrated choice is missing evidence identity")
            if cal_id not in evidence:
                evidence[cal_id] = utils.load_json(Path(spec.output_dir) / "lr_test" / "experiments" / cal_id / "report.json", {})
            report = evidence[cal_id]
            if (report.get("status") != "complete" or not report.get("activated")
                    or report.get("selected", {}).get(model, {}).get(track) != choice
                    or report.get("plan", {}).get("context") != payload["context"]):
                raise ValueError("Calibrated defaults are missing matching completed evidence; preserve the lr_test directory")
    return choices
