"""Failure classification must never disguise a partial overnight cohort."""
import json
from pathlib import Path

from rank_suite import config
from tools.preservation import canonical, csv_summary, expected_grid, inventory


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def defaults():
    return {"selected": {model: {"lora": {"lr": .0002, "head_lr_multiplier": 5.0}}
                         for model in config.MODELS}}


def test_expected_grid_preserves_all_requested_identities():
    rows = expected_grid(defaults())
    assert len(rows) == len({row["run_id"] for row in rows}) == 140
    assert {row["rank"] for row in rows} == {4, 8, 16, 32, 64, 128, 256}
    assert {row["seed"] for row in rows} == set(config.DEFAULT_SEEDS)


def test_pending_is_distinct_from_failed_and_budget_deferred(tmp_path):
    expected = expected_grid(defaults())
    first, second = expected[:2]
    root = tmp_path / "main" / first["model"]
    partial = root / "runs" / first["run_id"]
    (partial / "checkpoints" / "task1_rte").mkdir(parents=True)
    (partial / "resume.pt").write_bytes(b"preserved checkpoint")
    (partial / "train_log.jsonl").write_text('{"step": 468}\n')
    active = inventory(tmp_path, tmp_path / "vault", expected, {}, {})
    assert active["runs"][0]["state"] == "in_progress"
    assert active["runs"][0]["saved_task_directories"] == ["task1_rte"]
    assert active["counts"]["pending"] == 139
    assert active["retry_candidates"] == []
    (root / "failures-worker-0.jsonl").write_text(json.dumps({
        "run_id": second["run_id"], "error": "RuntimeError('out of memory')",
        "traceback": "a retained traceback"}) + '\n{"partial":')
    failed = inventory(tmp_path, tmp_path / "vault", expected, {}, {})
    assert failed["runs"][1]["state"] == "failed"
    assert failed["failure_events"][0]["traceback"] == "a retained traceback"
    assert failed["read_issues"]
    done = inventory(tmp_path, tmp_path / "vault", expected, {},
                     {first["model"]: {"status": "budget_deferred"}})
    assert done["runs"][0]["state"] == "incomplete"
    assert done["runs"][0]["resume_checkpoint_available"]
    assert done["runs"][0]["last_saved_optimizer_step"] == 468
    assert len(done["retry_candidates"]) == 35
    assert not done["training_finished"]


def test_complete_exports_and_missing_archive_are_reported_separately(tmp_path):
    expected = expected_grid(defaults())[:1]
    row = expected[0]
    root = tmp_path / "main" / row["model"]
    run = root / "runs" / row["run_id"]
    protocol = {"frozen": True}
    write(root / "protocol.json", protocol)
    sizes, boundaries = {}, []
    for i, task in enumerate(config.TASK_KEYS):
        folder = Path("checkpoints") / f"task{i + 1}_{task}"
        boundaries.append({"path": str(folder), "task": task})
        names = ["adapter_model.safetensors", "adapter_config.json", "heads.pt", "projections.pt"]
        names += [f"predictions_{key}.npz" for key in config.TASK_KEYS[:i + 1]]
        for name in names:
            path = run / folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            sizes[str(folder / name)] = 7
    (run / "train_log.jsonl").write_text(''.join(json.dumps({"step": i}) + '\n' for i in range(1, 1405)))
    write(run / "result.json", {**row, "status": "completed", "track": "lora", "mode": "main",
          "epochs": 3, "train_samples": 2490, "hf_revision": config.MODELS[row["model"]].revision,
          "protocol_sha256": canonical(protocol), "boundaries": boundaries, "checkpoint_sizes": sizes})
    report = inventory(tmp_path, tmp_path / "vault", expected, {}, {})
    assert report["completed_sequences"] == 1
    assert report["archived_sequences"] == 0
    assert report["retry_candidates"] == []
    manifest = tmp_path / "vault" / "archives" / row["model"] / (row["run_id"] + ".json")
    write(manifest, {"archive": "result.tar.zst", "archive_bytes": 3,
                     "archive_sha256": "transport checksum", "protocol_sha256": canonical(protocol)})
    (manifest.parent / "result.tar.zst").write_bytes(b"abc")
    report = inventory(tmp_path, tmp_path / "vault", expected, {}, {})
    assert report["archived_sequences"] == 1
    assert row["run_id"] in csv_summary(report["runs"])
    (run / "checkpoints" / "task3_sst2" / "heads.pt").unlink()
    broken = inventory(tmp_path, tmp_path / "vault", expected, {},
                       {row["model"]: {"completed": 1}})
    assert broken["runs"][0]["state"] == "artifact_invalid"
    assert broken["completed_sequences"] == 0
    assert len(broken["retry_candidates"]) == 1
