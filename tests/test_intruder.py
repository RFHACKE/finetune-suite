import numpy as np
import pytest

from rank_suite.intruder import intruder_counts, left_vectors


def test_reference_includes_vectors_outside_pretrained_top_ten():
    base = np.diag(np.arange(20, 0, -1)).astype(np.float32)
    tuned = base.copy()
    tuned[15, 15] = 25
    all_ref = left_vectors(base)
    assert intruder_counts(tuned, all_ref)[0]["0.9"] == 0
    assert intruder_counts(tuned, all_ref[:, :10])[0]["0.9"] == 1
    assert intruder_counts(tuned, -all_ref)[0]["0.9"] == 0


@pytest.mark.parametrize("shape", [(32, 32), (16, 48)])
def test_subset_eigensolve_matches_direct_svd(shape):
    rng = np.random.default_rng(42)
    base = rng.normal(size=shape).astype(np.float32)
    tuned = rng.normal(size=shape).astype(np.float32)
    ref = left_vectors(base)
    a, sims_a = intruder_counts(tuned, ref, solver="eigh")
    b, sims_b = intruder_counts(tuned, ref, solver="svd")
    assert a == b
    np.testing.assert_allclose(sims_a, sims_b, atol=1e-5)


def test_rotated_vectors_trigger_threshold_sensitivity():
    angle = np.pi / 4
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tuned = rot @ np.diag([3., 1.])
    counts, _ = intruder_counts(tuned, np.eye(2), k=2)
    assert counts["0.6"] == 0
    assert counts["0.9"] == 2


def test_interrupted_base_cache_is_rebuilt_from_pristine_weights(tmp_path):
    from dataclasses import replace
    import torch.nn as nn
    from rank_suite import config, utils
    from rank_suite.intruder import export_base_projections, file_sha256
    model = nn.Module()
    model.q_proj, model.v_proj = nn.Linear(4, 4), nn.Linear(4, 2)
    spec = replace(config.MODELS["qwen2.5-1.5b"], layers=1, d_model=4)
    path = export_base_projections(model, tmp_path, spec)
    path.write_bytes(b"interrupted write")
    export_base_projections(model, tmp_path, spec)
    assert utils.load_json(path.with_suffix(".json"))["sha256"] == file_sha256(path)


def test_update_metrics_match_dense_svd_and_factor_rescaling():
    from rank_suite.intruder import update_diagnostics
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(4, 15)), rng.normal(size=(12, 4))
    base = rng.normal(size=(12, 15))
    value = {'A': a, 'B': b, 'scale': 2.0}
    measured = update_diagnostics(base, value, 'lora')
    delta = 2 * b @ a
    dense = update_diagnostics(base, base + delta, 'full')
    scaled = update_diagnostics(base, {'A': a * 7, 'B': b / 7, 'scale': 2}, 'lora')
    for key in ('frobenius_norm', 'relative_frobenius', 'spectral_norm', 'stable_rank', 'effective_rank', 'energy_rank_90'):
        assert measured[key] == pytest.approx(dense[key])
        assert measured[key] == pytest.approx(scaled[key])
    assert measured['effective_rank'] <= 4.000001
    zero = update_diagnostics(base, {'A': a, 'B': b * 0, 'scale': 2}, 'lora')
    assert zero['zero_update'] and zero['effective_rank'] == zero['stable_rank'] == 0


def test_basis_rotation_can_change_intruders_without_moving_subspace():
    from rank_suite.intruder import subspace_diagnostics
    angle = np.pi / 4
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    reference = np.eye(3)
    top = np.eye(3)
    top[:2, :2] = rot
    diag = subspace_diagnostics(reference, np.array([3., 2., 1.]), top,
                                np.array([3., 2., 1.]), requested=(2,))['2']
    assert diag['distance'] == pytest.approx(0, abs=1e-12)
    assert diag['tuned_cutoff_gap']['relative'] == pytest.approx(.5)
    assert intruder_counts(top @ np.diag([3., 2., 1.]), reference, k=2)[0]['0.9'] == 2
    changed = subspace_diagnostics(reference, np.array([3., 2., 1.]),
                reference[:, [0, 2, 1]], np.array([3., 2., 1.]), requested=(2,))['2']
    assert changed['distance'] == pytest.approx(.5)
