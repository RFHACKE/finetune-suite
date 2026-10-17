"""Finalize the requested cohort only after all 140 runs and spectra verify.

Run in the suite's CPU environment with --watch. --stop-stream shuts down the
previously authorized temporary Modal archive endpoint after retrieval finishes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

from rank_suite.analysis import run_analysis
from rank_suite.protocol import source_digest
from tools.audit_requested_results import audit, sha256


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def finalize(root):
    report = audit(root)
    save(root / "requested-completeness.json", report)
    if report["status"] != "complete":
        raise ValueError("Requested-cohort audit is incomplete; aggregate finalization refused")
    if source_digest() != report["scientific_source_sha256"]:
        raise ValueError("Current scientific source differs from the trained cohort")
    analysis = run_analysis(root)
    save(root / "execution" / "final-analysis.json", analysis)
    source = root / "source"
    shutil.copytree("rank_suite", source / "rank_suite", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree("tools", source / "tools", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2("requirements.txt", source / "requirements.txt")
    for name in ("MODAL_EXECUTION.md", "PROTOCOL_V3.md", "PROTOCOL_V2.md"):
        shutil.copy2(Path("docs") / name, source / name)
    save(source / "provenance.json", {
        "scientific_source_sha256": source_digest(),
        "git_revision_at_finalization": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_runner_evidence": "../execution/runner-snapshots/ (actual launch snapshots)",
        "note": "Scientific package matches training. Transport helpers include subsequent reliability fixes."})
    policies = report["selected_policies"]
    rows, evidence = [], []
    for model, value in policies.items():
        policy = value["lora"]
        directory = Path("lr_test/experiments") / policy["calibration_id"]
        calibration = read(root / directory / "report.json")
        edge = any(g.get("search_boundary_warning") for g in calibration["groups"])
        rows.append(f"| {model} | {policy['lr']:.6f} | {policy['head_lr_multiplier']:g}× | {'Yes' if edge else 'No'} |")
        evidence.append(f"- [{model} calibration JSON]({directory}/report.json) and [charts]({directory}/charts/).")
    text = f"""# Finetune results

Completed and verified **140 LoRA sequences**: four fixed models × ranks 4, 8, 16, 32, 64, 128, 256 × seeds 42, 123, 456, 789, 1024. Every sequence trains RTE → MNLI → SST-2 for three epochs per task with 2,490 training examples per task and full official validation (277 / 9,815 / 872). All 140 sequences have completed CPU spectral diagnostics at all three task boundaries.

The training-only calibration completed 48 cases without selecting on official validation. These are the best eligible tested LoRA policies, fixed across all main ranks. An upper-edge selection warrants wider LR testing in a separate future cohort; it is not proof of a global optimum.

| Model | LoRA LR | Head multiplier | Upper search edge |
| --- | ---: | ---: | --- |
{chr(10).join(rows)}

Start with [the analysis report](analysis/report.md), [requested-cohort verification](requested-completeness.json), and [aggregate tables](aggregates/). [Selected defaults](lr_defaults.json) retain their matching calibration evidence. Resource and cost evidence is under [execution/](execution/).

{chr(10).join(evidence)}

`runs/` retains per-seed accuracy matrices, predictions, optimizer logs, adapter exports, classification heads, projection factors, and spectral diagnostics. `cache/` retains verified pretrained projections and reusable CPU reference spectra. `source/` snapshots the scientific code and pinned requirements. `execution/archive-manifests/` records independent cloud archive checksums; lossless transport preserves exported tensors byte for byte. Recovery-only obsolete `resume.pt` files remain in the cloud. The final local [artifact manifest](artifact-manifest.json) records every delivered scientific file's size and SHA-256.

This execution covers the requested expanded LoRA track. Full-FT comparisons and baseline-dependent feasible-rank certification require an additional baseline cohort. The suite's original 120-run audit therefore remains separate from the requested 140-run completeness audit. Gradient-spike flags remain visible in the raw results and verification report.

Scientific source SHA-256: `{report['scientific_source_sha256']}`. Common protocol SHA-256: `{report['protocol_sha256']}`. Generated {datetime.now(timezone.utc).isoformat()}.
"""
    (root / "README.md").write_text(text)
    records = []
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.name.endswith((".lock", ".tmp", ".download"))
                or path.name == "artifact-manifest.json"):
            continue
        records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                        "sha256": sha256(path)})
    save(root / "artifact-manifest.json", {"generated_utc": datetime.now(timezone.utc).isoformat(),
         "files": records, "total_bytes": sum(r["bytes"] for r in records),
         "verified_sequences": 140, "completed_spectral_sequences": 140})
    return {"status": "complete", "runs": 140, "figures": len(analysis["figures"]), "manifest_files": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Finetune-result")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--stop-stream", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    stopped = False
    while True:
        state = read(root / "execution" / "retrieval.json", {})
        results = [read(path) for path in (root / "runs").glob("*/result.json")]
        spectral = sum(r.get("intruder_status") == "completed" for r in results)
        retrieved = len(state.get("verified_runs", [])) == 140 and state.get("training_finished")
        if retrieved and args.stop_stream and not stopped:
            subprocess.run(["modal", "app", "stop", "--yes", "fyp-archive-stream"], check=True)
            stopped = True
        if retrieved and spectral == 140:
            break
        if not args.watch:
            raise SystemExit(f"Not ready: verified={len(state.get('verified_runs', []))}, spectra={spectral}")
        time.sleep(max(5, args.interval))
    billing = subprocess.run(["modal", "billing", "summary", "--json"], capture_output=True, text=True)
    if billing.returncode == 0:
        save(root / "execution" / "billing-final.json", json.loads(billing.stdout))
    print(json.dumps(finalize(root)), flush=True)


if __name__ == "__main__":
    main()
