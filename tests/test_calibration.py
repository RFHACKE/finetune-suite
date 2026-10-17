"""Orchestration fixtures are synthetic; they never claim checkpoint accuracy."""
from dataclasses import replace
from pathlib import Path

import pytest

from rank_suite import calibration, config, metrics, train, utils
from rank_suite.runner import build_grid

BEST_RATIO = {'llama3.2-1b': .5, 'qwen2.5-1.5b': 1, 'granite3.3-2b': 2, 'qwen2.5-3b': .5}

def test_selection_uses_distinct_rates_and_cannot_reward_underlearning():
    screen = [{'status': 'completed', 'lr': lr, 'head_lr_multiplier': head,
               'metrics': {'final_task_acc': score}, 'NLL': [[1 - score]]}
              for lr, head, score in [(1e-4, 1, .85), (1e-4, 5, .86), (2e-4, 1, .84), (4e-4, 1, .83)]]
    chosen = calibration.choose_finalists(screen)
    assert [(c['lr'], c['head_lr_multiplier']) for c in chosen] == [(1e-4, 5), (2e-4, 1)]
    candidates = [dict(eligible=False, mean_final_accuracy=.99),
                  dict(eligible=True, mean_final_accuracy=.75, mean_diagonal=.8,
                       mean_final_nll=.5, lr=2e-4, head_lr_multiplier=5)]
    assert calibration.choose_winner(candidates)['lr'] == 2e-4
    assert calibration.choose_winner(candidates[:1]) is None


def test_calibration_plan_is_bounded_and_separate_from_main():
    spec = config.ExperimentSpec()
    plan = calibration.make_plan(spec)
    assert plan['workload']['maximum_runs'] == 88
    assert plan['workload']['maximum_optimizer_steps'] == 57000
    assert plan['seeds'] == [42, 123]
    assert plan['confirmation_tasks'] == ['rte', 'mnli', 'sst2']
    assert plan['endpoint_ranks'] == [4, 64]
    expanded = calibration.make_plan(replace(spec, track='lora', ranks=[4, 8, 16, 32, 64, 128, 256]))
    assert expanded['endpoint_ranks'] == [4, 256]
    assert expanded['workload']['maximum_runs'] == 48
    assert expanded['workload']['maximum_optimizer_steps'] == 30000
    assert len(build_grid(spec)) == 120
    for change in (replace(spec, num_shards=2), replace(spec, mode='smoke')):
        with pytest.raises(ValueError):
            calibration.make_plan(change)
    with pytest.raises(ValueError, match='two distinct'):
        calibration.make_plan(spec, seeds=[42])


@pytest.fixture
def synthetic_calibration(monkeypatch):
    from rank_suite import data, model_utils
    calls = []
    monkeypatch.setattr(model_utils, 'load_tokenizer', lambda spec: spec.key)
    def prepare(tokenizer, **kwargs):
        assert kwargs['calibration'] is True
        return {t: {'fixture': 'synthetic'} for t in config.TASK_KEYS}
    monkeypatch.setattr(data, 'prepare_task_data', prepare)
    def run(run, directory, data, tokenizer, mspec, **kwargs):
        assert kwargs['experiment'].mode == 'lr_test'
        calls.append(run)
        ratio = run.lr / getattr(mspec, f'default_{run.track}_lr')
        preferred = BEST_RATIO[run.model] if run.track == 'lora' else 2 / BEST_RATIO[run.model]
        # Keep the full-FT preferred rate inside the tested grid.
        preferred = min(preferred, 2)
        score = .72 + (.04 if ratio == preferred else 0) + (.01 if run.head_lr_multiplier == 5 else 0)
        score += 0 if run.seed == 42 else -.01
        size = len(run.task_keys)
        A = [[score + .01 * j - .01 * (i - j) if j <= i else None for j in range(size)] for i in range(size)]
        NLL = [[1.5 - value if value is not None else None for value in row] for row in A]
        result = {'status': 'completed', 'mode': 'lr_test', 'run_id': run.run_id,
                  'run_identity': run.identity, 'protocol_sha256': kwargs['protocol_sha'],
                  'model': run.model, 'track': run.track, 'rank': run.rank,
                  'lr': run.lr, 'head_lr_multiplier': run.head_lr_multiplier, 'seed': run.seed,
                  'A': A, 'NLL': NLL, 'metrics': metrics.compute_run_metrics(A, run.task_keys),
                  'train_summary': {}, 'flags': {'grad_spike_steps': []},
                  'data': {t: {'majority_eval_accuracy': 1 / config.TASK_BY_KEY[t].num_labels,
                               'calibration': True} for t in run.task_keys},
                  'hardware': {'name': 'synthetic-fixture'}, 'train_time_s': 1}
        utils.save_json(Path(directory) / 'result.json', result)
        return result
    monkeypatch.setattr(train, 'run_single', run)
    return calls


def test_full_selection_saves_charts_defaults_and_resumes_without_retraining(tmp_path, synthetic_calibration):
    spec = config.ExperimentSpec(output_dir=str(tmp_path), optimizer='adamw')
    result = calibration.run_calibration(spec, device='cpu')
    assert result['status'] == 'complete' and result['activated']
    assert len(synthetic_calibration) == 88
    report = utils.load_json(result['report'])
    assert len(report['trials']) == 88 and len(report['charts']) == 32
    assert all((Path(result['report']).parent / name).stat().st_size > 1000 for name in report['charts'])
    assert report['plan']['context']['data_revision'] == config.DATA_REVISION
    policy = config.ExperimentSpec(output_dir=str(tmp_path), optimizer='adamw')
    grid = build_grid(policy)
    assert len(grid) == 120
    assert all(r.head_lr_multiplier == 5 for r in grid)
    for run in grid:
        ratio = BEST_RATIO[run.model] if run.track == 'lora' else min(2 / BEST_RATIO[run.model], 2)
        assert run.lr == pytest.approx(ratio * getattr(config.MODELS[run.model], f'default_{run.track}_lr'))
    again = calibration.run_calibration(spec, device='cpu')
    assert again['completed'] == 0 and again['skipped'] == 88
    assert len(synthetic_calibration) == 88
    with pytest.raises(ValueError, match='different code/training controls'):
        config.ExperimentSpec(output_dir=str(tmp_path), optimizer='adamw', micro_batch=8)
    # Changing evidence cannot make a fabricated learning rate become a default.
    report_path = Path(again['report'])
    report = utils.load_json(report_path)
    report['selected']['qwen2.5-3b']['lora']['lr'] *= 2
    utils.save_json(report_path, report)
    with pytest.raises(ValueError, match='matching completed evidence'):
        config.ExperimentSpec(output_dir=str(tmp_path), optimizer='adamw')


def test_forced_recalibration_preserves_active_evidence_until_replacement(tmp_path, synthetic_calibration):
    spec = config.ExperimentSpec(models=['qwen2.5-1.5b'], track='lora',
                                 output_dir=str(tmp_path), optimizer='adamw')
    initial = calibration.run_calibration(spec, device='cpu')
    old_path = Path(initial['report'])
    previous_evidence = old_path.read_bytes()
    forced = calibration.run_calibration(replace(spec, force=True), device='cpu')
    assert forced['activated'] and forced['completed'] == 12
    assert forced['report'] != initial['report']
    assert old_path.read_bytes() == previous_evidence
    loaded = config.ExperimentSpec(output_dir=str(tmp_path), optimizer='adamw')
    assert loaded.resolved_head_multiplier('qwen2.5-1.5b', 'lora') == 5


def test_budget_stop_and_started_main_cohort_do_not_activate_defaults(tmp_path, monkeypatch, synthetic_calibration):
    monkeypatch.setattr('rank_suite.cost.scheduling_limit_hours', lambda spec: 0)
    spec = config.ExperimentSpec(models=['qwen2.5-1.5b'], output_dir=str(tmp_path), optimizer='adamw')
    result = calibration.run_calibration(spec, device='cpu')
    assert result['status'] == 'deferred' and not result['activated']
    assert not (tmp_path / 'lr_defaults.json').exists()
    assert synthetic_calibration == []
    utils.save_json(tmp_path / 'protocol.json', {'fixture': 'already-started'})
    with pytest.raises(ValueError, match='already frozen'):
        calibration.run_calibration(spec, device='cpu')


def test_unknown_runtime_failure_preserves_report_and_cannot_select_alternative(tmp_path, monkeypatch, synthetic_calibration):
    def fail(*args, **kwargs):
        raise RuntimeError('infrastructure fixture failure')
    monkeypatch.setattr(train, 'run_single', fail)
    spec = config.ExperimentSpec(models=['qwen2.5-1.5b'], output_dir=str(tmp_path), optimizer='adamw')
    with pytest.raises(RuntimeError, match='infrastructure'):
        calibration.run_calibration(spec, device='cpu')
    latest = utils.load_json(tmp_path / 'lr_test/latest.json')
    report = utils.load_json(tmp_path / 'lr_test' / latest['report'])
    assert report['status'] == 'failed' and 'infrastructure' in report['error']
    assert not (tmp_path / 'lr_defaults.json').exists()
