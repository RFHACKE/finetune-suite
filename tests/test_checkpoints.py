"""Lossless restoration and fail-closed pruning of real serialized tensors."""
import json

import pytest
from safetensors.torch import save_file
import torch

from rank_suite import config, utils
from rank_suite.checkpoints import (FORMAT, STORAGE_NAME, adapter_payload, checkpoint_inventory_errors,
                                    factor_digest, load_boundary_projections, maintain_checkpoints,
                                    restore_boundary, sha256, validate_boundary_storage,
                                    verify_boundary, write_adapter_manifest, compacted_projection_record)
from rank_suite.protocol import digest


def checkpoint_fixture(root, seed=42, native=False):
    run = config.RunSpec("qwen2.5-1.5b", "lora", 2, .001, seed)
    identity = json.loads(json.dumps(run.identity))
    directory = root / "runs" / run.run_id
    checkpoint = directory / "checkpoints" / "task1_rte"
    checkpoint.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed)
    matrices = {f"model.layers.0.self_attn.{key}": {
        "A": torch.randn(2, 7, generator=generator),
        "B": torch.randn(out, 2, generator=generator), "scale": 2.0}
        for key, out in (("q_proj", 5), ("v_proj", 3))}
    payload = {"track": "lora", "identity": identity, "matrices": matrices}
    save_file({f"base_model.model.{name}.lora_{letter}.weight": value[letter]
               for name, value in matrices.items() for letter in ("A", "B")},
              checkpoint / "adapter_model.safetensors")
    utils.save_json(checkpoint / "adapter_config.json", {
        "peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none",
        "target_modules": ["q_proj", "v_proj"], "revision": identity["revision"]})
    torch.save([{"weight": torch.ones(2, 7)}], checkpoint / "heads.pt")
    boundary = {"task": "rte", "path": "checkpoints/task1_rte"}
    if native:
        record = write_adapter_manifest(checkpoint, identity)
        boundary.update(projection_storage=FORMAT, adapter_factors_sha256=record["factors_sha256"])
    else:
        torch.save(payload, checkpoint / "projections.pt")
        boundary["projections_sha256"] = sha256(checkpoint / "projections.pt")
    result = {"status": "completed", "track": "lora", "mode": "main", "run_id": run.run_id,
              "model": run.model, "hf_revision": identity["revision"], "protocol_sha256": "fixture",
              "run_identity": identity, "boundaries": [boundary], "checkpoint_sizes": {
                  str(p.relative_to(directory)): p.stat().st_size
                  for p in checkpoint.iterdir() if p.name != STORAGE_NAME}}
    result["checkpoint_total_bytes"] = sum(result["checkpoint_sizes"].values())
    utils.save_json(directory / "result.json", result)
    return directory, checkpoint, result, boundary, payload


def test_bit_exact_factors_and_cpu_roundtrip_without_changing_original_provenance(tmp_path):
    directory, checkpoint, result, boundary, original = checkpoint_fixture(tmp_path)
    original_result = (directory / "result.json").read_bytes()
    manifest = verify_boundary(directory, result, boundary)
    assert manifest["n_matrices"] == 2
    assert factor_digest(original) == factor_digest(adapter_payload(checkpoint, result["run_identity"]))
    report = maintain_checkpoints([tmp_path], "prune")
    assert report["removed_files"] == 1 and report["reclaimed_bytes"] > 0
    assert not (checkpoint / "projections.pt").exists()
    assert factor_digest(load_boundary_projections(directory, result, boundary)) == factor_digest(original)
    assert not checkpoint_inventory_errors(directory, result, verify_hashes=True)
    assert restore_boundary(directory, result, boundary)
    assert (checkpoint / "projections.pt").is_file()
    assert factor_digest(load_boundary_projections(directory, result, boundary)) == factor_digest(original)
    stored = utils.load_json(checkpoint / STORAGE_NAME)
    assert stored["materialized_sha256"] == sha256(checkpoint / "projections.pt")
    assert stored["original_projection_sha256"] == boundary["projections_sha256"]
    assert not checkpoint_inventory_errors(directory, result, verify_hashes=True)
    assert (directory / "result.json").read_bytes() == original_result
    assert maintain_checkpoints([tmp_path], "prune")["removed_files"] == 1
    assert maintain_checkpoints([tmp_path], "prune")["removed_files"] == 0


def test_all_selected_pairs_pass_before_any_file_is_deleted(tmp_path):
    _, first, _, _, _ = checkpoint_fixture(tmp_path)
    _, second, _, _, _ = checkpoint_fixture(tmp_path, seed=123)
    # Still a valid safetensors file, with the same keys/shapes and wrong values.
    from safetensors.torch import load_file
    path = second / "adapter_model.safetensors"
    tensors = load_file(path)
    tensors[next(iter(tensors))][0, 0] += 1
    save_file(tensors, path)
    with pytest.raises(ValueError, match="factors/keys/shapes/scaling differ"):
        maintain_checkpoints([tmp_path], "prune")
    assert (first / "projections.pt").is_file()
    assert (second / "projections.pt").is_file()
    reports = list((tmp_path / "execution" / "checkpoint-storage").glob("*.json"))
    assert len(reports) == 1 and utils.load_json(reports[0])["removed_files"] == 0


def test_absent_projections_without_attestation_is_an_error(tmp_path):
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path)
    (checkpoint / "projections.pt").unlink()
    with pytest.raises(ValueError, match="manifest"):
        load_boundary_projections(directory, result, boundary)
    assert checkpoint_inventory_errors(directory, result)


def test_corrupt_retained_adapter_is_detected_even_when_size_is_unchanged(tmp_path):
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path)
    maintain_checkpoints([tmp_path], "prune")
    path = checkpoint / "adapter_model.safetensors"
    with path.open("r+b") as target:
        target.seek(-1, 2)
        value = target.read(1)
        target.seek(-1, 2)
        target.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_boundary_projections(directory, result, boundary)
    assert checkpoint_inventory_errors(directory, result, verify_hashes=True)


def test_manifest_is_bound_to_run_identity_and_original_checksum(tmp_path):
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path)
    maintain_checkpoints([tmp_path], "prune")
    result["run_identity"]["seed"] += 1
    with pytest.raises(ValueError, match="mismatched"):
        load_boundary_projections(directory, result, boundary)
    result["run_identity"]["seed"] -= 1
    boundary["projections_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mismatched"):
        load_boundary_projections(directory, result, boundary)


def test_native_adapter_only_storage_supports_restore_and_prune(tmp_path):
    directory, checkpoint, result, boundary, original = checkpoint_fixture(tmp_path, native=True)
    assert not (checkpoint / "projections.pt").exists()
    assert factor_digest(load_boundary_projections(directory, result, boundary)) == factor_digest(original)
    assert validate_boundary_storage(directory, result, boundary, verify_hashes=True)["format"] == FORMAT
    assert restore_boundary(directory, result, boundary)
    assert not checkpoint_inventory_errors(directory, result, verify_hashes=True)
    assert maintain_checkpoints([tmp_path], "prune")["removed_files"] == 1
    assert not (checkpoint / "projections.pt").exists()
    assert digest(result["run_identity"]) == utils.load_json(checkpoint / STORAGE_NAME)["identity_sha256"]


@pytest.mark.parametrize("variant", ["use_dora", "use_rslora", "modules_to_save"])
def test_unsupported_adapter_variants_are_not_compacted(tmp_path, variant):
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path)
    cfg = utils.load_json(checkpoint / "adapter_config.json")
    cfg[variant] = ["head"] if variant == "modules_to_save" else True
    utils.save_json(checkpoint / "adapter_config.json", cfg)
    with pytest.raises(ValueError, match="Unsupported adapter"):
        verify_boundary(directory, result, boundary)
    assert (checkpoint / "projections.pt").exists()


def test_cannot_compact_a_busy_run(tmp_path):
    from rank_suite.protocol import file_lock
    directory, checkpoint, _, _, _ = checkpoint_fixture(tmp_path)
    with file_lock(directory / "run.lock"):
        with pytest.raises(ValueError, match="busy"):
            maintain_checkpoints([tmp_path], "prune")
    assert (checkpoint / "projections.pt").exists()


def test_unknown_run_id_and_no_runs_fail_before_mutation(tmp_path):
    _, checkpoint, _, _, _ = checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match="Unknown or ineligible"):
        maintain_checkpoints([tmp_path], "prune", run_ids=["missing"])
    assert (checkpoint / "projections.pt").exists()


def test_resumed_download_recognizes_intentionally_compacted_original(tmp_path):
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path)
    assert compacted_projection_record(checkpoint / "projections.pt") is None
    maintain_checkpoints([tmp_path], "prune")
    record = compacted_projection_record(checkpoint / "projections.pt")
    assert record["sha256"] == boundary["projections_sha256"] and record["retained_as"] == FORMAT
    assert record["bytes"] > 0
    (checkpoint / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="retained source"):
        compacted_projection_record(checkpoint / "projections.pt")


def test_a_present_but_truncated_original_is_not_excused_by_a_valid_manifest(tmp_path):
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path)
    verify_boundary(directory, result, boundary)
    (checkpoint / "projections.pt").write_bytes(b"broken")
    assert checkpoint_inventory_errors(directory, result)
    with pytest.raises(ValueError, match="checksum"):
        load_boundary_projections(directory, result, boundary)


def test_subsequent_source_corruption_does_not_allow_reverification(tmp_path):
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path)
    verify_boundary(directory, result, boundary)
    manifest = utils.load_json(checkpoint / STORAGE_NAME)
    manifest["factors_sha256"] = "0" * 64
    utils.save_json(checkpoint / STORAGE_NAME, manifest)
    with pytest.raises(ValueError, match="Factor checksum disagrees"):
        verify_boundary(directory, result, boundary)
    assert (checkpoint / "projections.pt").is_file()


def test_cli_parses_all_checkpoint_options_and_legacy_export_switch():
    from rank_suite.cli import _build_parser, _make_spec
    parser = _build_parser()
    args = parser.parse_args(["checkpoints", "verify", "--output", "a", "b",
                              "--run-id", "run1", "run2", "--tasks", "rte", "sst2"])
    assert args.output == ["a", "b"] and args.run_id == ["run1", "run2"]
    assert args.tasks == ["rte", "sst2"]
    assert not _make_spec(parser.parse_args(["plan"])).save_lora_projections
    assert _make_spec(parser.parse_args(["plan", "--save-lora-projections"])).save_lora_projections


def test_collector_does_not_redownload_a_verified_compacted_projection(tmp_path):
    pytest.importorskip("modal")
    from types import SimpleNamespace
    from tools.collect_modal_results import Collector
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path)
    size = (checkpoint / "projections.pt").stat().st_size
    maintain_checkpoints([tmp_path], "prune")
    client = Collector.__new__(Collector)
    client.root, client.state = tmp_path, {"files": {}}
    entry = SimpleNamespace(path="original/remote/projections.pt", size=size, mtime=123)
    key, record = client.fetch((entry, checkpoint / "projections.pt"))
    assert record["retained_as"] == FORMAT and record["sha256_at_retrieval"] == boundary["projections_sha256"]
    assert not (checkpoint / "projections.pt").exists()
    entry.size += 1
    with pytest.raises(ValueError, match="Remote projection differs"):
        client.fetch((entry, checkpoint / "projections.pt"))


def test_collector_verifies_adapter_only_candidates(tmp_path):
    pytest.importorskip("modal")
    from tools.collect_modal_results import Collector
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path, native=True)
    base = tmp_path / "cache/projections" / result["model"] / result["hf_revision"] / "base.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"fixture base")
    utils.save_json(base.with_suffix(".json"), {"sha256": sha256(base)})
    client = Collector.__new__(Collector)
    client.root, client.state = tmp_path, {}
    verified = set()
    client.verify_candidates({result["run_id"]: result}, verified)
    assert verified == {result["run_id"]}


def test_local_archive_packager_accepts_verified_compaction(tmp_path, monkeypatch):
    pytest.importorskip("modal")
    import io
    import subprocess
    import tarfile
    from types import SimpleNamespace
    from tools import modal_archive
    monkeypatch.setattr(modal_archive, "vault", SimpleNamespace(commit=lambda: None))
    directory, checkpoint, result, boundary, _ = checkpoint_fixture(tmp_path)
    maintain_checkpoints([tmp_path], "prune")
    destination = tmp_path / "archives"
    assert modal_archive.package_run(tmp_path, directory, destination)
    manifest = utils.load_json(destination / f"{result['run_id']}.json")
    assert manifest["original_checkpoint_export_bytes"] == result["checkpoint_total_bytes"]
    raw = subprocess.check_output(["zstd", "-dq", "--long=30", "-c", str(destination / manifest["archive"])])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        names = archive.getnames()
        assert any(name.endswith(STORAGE_NAME) for name in names)
        assert any(name.endswith("adapter_model.safetensors") for name in names)
        assert not any(name.endswith("projections.pt") for name in names)
