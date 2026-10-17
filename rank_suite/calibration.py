"""Training-only, resumable LR/head-rate selection with frozen cohort defaults.

Screen on RTE, confirm two distinct rates over the full sequence/two seeds,
then diagnose the chosen LoRA policy at rank 4 and 64 on RTE. No official
validation score participates in selection. Calibration never starts main runs.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from statistics import mean, stdev
import time
import traceback
import uuid

from . import config, utils
from .lr_defaults import SCHEMA, calibration_context, load_defaults
from .protocol import digest, file_lock, immutable_json

CALIBRATION_VERSION = 'training-holdout-sequential-v1'
SOURCES = [
    {'title': 'Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning',
     'url': 'https://arxiv.org/html/2602.04998v2',
     'use': 'Method/model/task-dependent LR sensitivity; not an exact-suite optimum.'},
    {'title': 'LoRA Learns Less and Forgets Less',
     'url': 'https://arxiv.org/html/2405.09673v2',
     'use': 'Learning/retention and optimization-policy sensitivity; not an exact small-model LR rule.'},
    {'title': 'On the Stability of Fine-tuning BERT',
     'url': 'https://arxiv.org/abs/2006.04884v3',
     'use': 'Replicated fine-tuning and checks for optimization failure.'},
]


def make_plan(spec, seeds=(42, 123)):
    seeds = tuple(dict.fromkeys(seeds))
    if len(seeds) < 2 or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('LR confirmation needs at least two distinct valid seeds')
    if spec.num_shards != 1:
        raise ValueError('--test-lr uses one GPU worker; shard the main experiment after calibration')
    if spec.mode != 'main':
        raise ValueError('--test-lr cannot be combined with smoke, sanity or pilot mode')
    groups = []
    heads = [spec.head_lr_multiplier] if spec.head_lr_multiplier is not None else [1.0, 5.0]
    for model in sorted(spec.models, key=lambda m: config.MODELS[m].params_total, reverse=True):
        for track in ('full', 'lora'):
            if spec.track not in ('both', track):
                continue
            explicit = spec.lora_lrs if track == 'lora' else spec.full_lrs
            prior = getattr(config.MODELS[model], f'default_{track}_lr')
            rates = sorted(set(explicit[model] if explicit and explicit.get(model)
                               else [prior * factor for factor in (.5, 1, 2)]))
            if len(rates) < 2:
                raise ValueError(f'LR selection requires at least two distinct rates: {model}/{track}')
            groups.append({'model': model, 'track': track, 'rates': rates,
                           'head_multipliers': heads, 'rank': 16 if track == 'lora' else None})
    screening = sum(len(g['rates']) * len(heads) for g in groups)
    confirmation = len(groups) * 2 * len(seeds)
    endpoints = 2 * sum(g['track'] == 'lora' for g in groups)
    steps_per_task = math.ceil(int(.8 * spec.train_samples) / config.TRAIN_BATCH) * spec.epochs
    plan = {'version': CALIBRATION_VERSION, 'context': calibration_context(spec),
            'groups': groups, 'seeds': list(seeds), 'screen_task': 'rte',
            'confirmation_tasks': list(config.TASK_KEYS), 'finalists_per_track': 2,
            'endpoint_ranks': [4, 64], 'endpoint_task': 'rte', 'sources': SOURCES,
            'selection': {'screen': 'RTE accuracy, then NLL; best head per rate; top two distinct rates',
                'confirmation': 'highest mean final-row accuracy across tasks/seeds among candidates whose mean diagonal beats each holdout majority baseline; ties: mean diagonal, NLL, lower LR, lower head multiplier',
                'endpoint': 'RTE checks at r=4,64; underlearning is reported; numerical instability blocks activation',
                'uncertainty': 'sample SD and every paired seed score; two seeds do not prove a universal optimum'},
            'workload': {'screening_runs': screening, 'confirmation_runs': confirmation,
                'endpoint_runs': endpoints, 'maximum_runs': screening + confirmation + endpoints,
                'maximum_optimizer_steps': (screening + 3 * confirmation + endpoints) * steps_per_task,
                'note': 'Three epochs on 80% training splits; these are calibration-only runs. Step counts are not wall-time or dollar estimates.'}}
    plan['calibration_id'] = digest(plan)[:20]
    return plan


def _run_spec(spec, group, rate, head, seed, tasks):
    optimizer = 'adamw8bit' if spec.optimizer == 'standard' and group['track'] == 'full' else 'adamw'
    return config.RunSpec(group['model'], group['track'], group['rank'], rate, seed,
                          task_keys=tuple(tasks), mode='lr_test', epochs=spec.epochs,
                          train_samples=spec.train_samples, optimizer=optimizer,
                          head_lr_multiplier=head)


def _trial_record(result, stage):
    keys = ['run_id', 'model', 'track', 'rank', 'lr', 'head_lr_multiplier', 'seed',
            'status', 'A', 'NLL', 'metrics', 'flags', 'train_summary', 'data',
            'train_time_s', 'peak_vram_gb', 'error']
    return {'stage': stage, **{k: result.get(k) for k in keys}}


def _screen_key(result):
    return (-result['metrics']['final_task_acc'], result['NLL'][-1][-1],
            result['lr'], result['head_lr_multiplier'])


def choose_finalists(screen):
    """Select distinct rates; do not spend both slots on one LR's two heads."""
    by_rate = {}
    for result in sorted((r for r in screen if r['status'] == 'completed'), key=_screen_key):
        by_rate.setdefault(result['lr'], result)
    return sorted(by_rate.values(), key=_screen_key)[:2]


def summarize_candidate(results, seeds):
    if len(results) != len(seeds) or {r['seed'] for r in results} != set(seeds):
        raise ValueError('Incomplete/duplicate confirmation seed block')
    success = all(r['status'] == 'completed' for r in results)
    first = results[0]
    out = {k: first[k] for k in ('model', 'track', 'lr', 'head_lr_multiplier')}
    out.update({'run_ids': [r['run_id'] for r in results], 'complete': success,
                'eligible': False, 'per_seed': []})
    if not success:
        return out
    out['per_seed'] = [{'seed': r['seed'], 'final_mean_accuracy': r['metrics']['avg_final_row'],
                       'mean_diagonal': r['metrics']['avg_diagonal'], 'bwt': r['metrics']['bwt'],
                       'final_mean_nll': mean(r['NLL'][-1])} for r in results]
    out['mean_final_accuracy'] = mean(x['final_mean_accuracy'] for x in out['per_seed'])
    out['sd_final_accuracy'] = stdev(x['final_mean_accuracy'] for x in out['per_seed'])
    out['mean_diagonal'] = mean(x['mean_diagonal'] for x in out['per_seed'])
    out['mean_final_nll'] = mean(x['final_mean_nll'] for x in out['per_seed'])
    out['learning_checks'] = {task: {
        'mean_diagonal': mean(r['A'][i][i] for r in results),
        'majority_accuracy': mean(r['data'][task]['majority_eval_accuracy'] for r in results)}
        for i, task in enumerate(config.TASK_KEYS)}
    out['eligible'] = all(v['mean_diagonal'] > v['majority_accuracy'] for v in out['learning_checks'].values())
    out['gradient_spike_steps'] = sum(len(r.get('flags', {}).get('grad_spike_steps', [])) for r in results)
    return out


def choose_winner(candidates):
    eligible = [c for c in candidates if c['eligible']]
    return min(eligible, key=lambda c: (-c['mean_final_accuracy'], -c['mean_diagonal'],
                                      c['mean_final_nll'], c['lr'], c['head_lr_multiplier'])) if eligible else None


class AdmissionStop(Exception):
    pass


def run_calibration(spec, seeds=(42, 123), *, device='cuda'):
    root = Path(spec.output_dir)
    plan = make_plan(spec, seeds)
    with file_lock(root / 'defaults.lock', blocking=False) as acquired:
        if not acquired:
            raise ValueError('Another calibration/main launch is using this output directory')
        if (root / 'protocol.json').exists():
            raise ValueError('Main training has already frozen this cohort. Run --test-lr in a new --output directory.')
        previous_report = utils.load_json(root / 'lr_test' / 'experiments'
                                          / plan['calibration_id'] / 'report.json', {})
        if spec.force or previous_report.get('activated'):
            # Active evidence must remain valid while a repeat is incomplete.
            # Cached trial identities still permit an ordinary repeat to skip training.
            plan['repeat_of'] = plan['calibration_id']
            plan['attempt_id'] = uuid.uuid4().hex
            plan['calibration_id'] = digest(plan)[:20]
        return _execute_calibration(spec, plan, device)


def _execute_calibration(spec, plan, device):
    from . import train
    from .data import prepare_task_data
    from .model_utils import load_tokenizer
    from .protocol import register_hardware, register_invocation
    from .cost import scheduling_limit_hours
    from .hardware import check_environment
    from .calibration_plots import write_report
    root = Path(spec.output_dir)
    calroot = root / 'lr_test'
    report_dir = calroot / 'experiments' / plan['calibration_id']
    immutable_json(report_dir / 'plan.json', plan)
    runtime = replace(spec, output_dir=str(calroot), mode='lr_test',
                      calibrated_defaults={}, head_lr_multiplier=None)
    if str(device).startswith('cuda'):
        check_environment(require_gpu=True, optimizer='adamw8bit' if spec.optimizer == 'standard'
                          and spec.track in ('full', 'both') else 'adamw')
    protocol_sha = register_invocation(calroot, runtime, utils.capture_environment())
    limit = scheduling_limit_hours(spec)
    start = time.monotonic()
    counts = {'completed': 0, 'skipped': 0, 'unstable': 0, 'failed': 0}
    report = {'plan': plan, 'status': 'running', 'activated': False, 'trials': [],
              'groups': [], 'defaults_file': str(root / 'lr_defaults.json')}
    selected, data_cache, hardware_cache = {}, {}, {}
    pending_payload = None

    def trial(group, lr, head, seed, tasks, stage):
        run = _run_spec(spec, group, lr, head, seed, tasks)
        run_dir = calroot / 'runs' / run.run_id
        old = utils.load_json(run_dir / 'result.json', {})
        if old.get('status') in ('completed', 'unstable') and not spec.force:
            if old.get('protocol_sha256') != protocol_sha or old.get('run_identity') != run.identity:
                # JSON tuples are lists on disk; compare canonical digests.
                if old.get('protocol_sha256') != protocol_sha or digest(old.get('run_identity')) != digest(run.identity):
                    raise ValueError('Calibration result identity/controls mismatch')
            counts['skipped'] += 1
            hardware_cache.setdefault(run.model, old.get('hardware', {}))
            report['trials'].append(_trial_record(old, stage))
            return old
        if limit is not None:
            elapsed = (time.monotonic() - start) / 3600
            measured = [(r['train_time_s'] / len(r['A'])) for r in report['trials']
                        if r['status'] == 'completed' and r['model'] == group['model']
                        and r['track'] == group['track'] and r.get('train_time_s') and r.get('A')]
            estimate = mean(measured) * len(tasks) / 3600 if measured else 0
            if elapsed >= limit or (estimate and elapsed + 1.25 * estimate > limit):
                raise AdmissionStop('Scheduling limit reached; no defaults activated')
        model = group['model']
        if model not in data_cache:
            mspec = config.MODELS[model]
            hardware_cache[model] = register_hardware(calroot, model) if str(device).startswith('cuda') else {'name': 'cpu-test'}
            tok = load_tokenizer(mspec)
            data_cache[model] = (tok, prepare_task_data(tok, n_train=spec.train_samples,
                selection_dir=calroot / 'data' / 'calibration', calibration=True))
        tok, data = data_cache[model]
        with file_lock(run_dir / 'run.lock', blocking=False) as acquired:
            if not acquired:
                raise AdmissionStop('Calibration case is locked by another worker')
            if spec.force:
                (run_dir / 'result.json').unlink(missing_ok=True)
                (run_dir / 'resume.pt').unlink(missing_ok=True)
            try:
                result = train.run_single(run, run_dir, data, tok, config.MODELS[model],
                    experiment=runtime, protocol_sha=protocol_sha,
                    hardware=hardware_cache[model], device=device)
                counts['completed'] += 1
            except FloatingPointError as exc:
                result = {'status': 'unstable', 'run_id': run.run_id, 'run_identity': run.identity,
                    'protocol_sha256': protocol_sha, 'model': model, 'track': group['track'],
                    'lr': lr, 'rank': group['rank'], 'seed': seed, 'head_lr_multiplier': head,
                    'error': str(exc)}
                utils.save_json(run_dir / 'result.json', result)
                counts['unstable'] += 1
            except Exception as exc:
                if spec.fail_fast:
                    raise
                result = {'status': 'failed', 'run_id': run.run_id, 'run_identity': run.identity,
                    'protocol_sha256': protocol_sha, 'model': model, 'track': group['track'],
                    'lr': lr, 'rank': group['rank'], 'seed': seed, 'head_lr_multiplier': head,
                    'error': repr(exc), 'traceback': traceback.format_exc()}
                utils.save_json(run_dir / 'result.json', result)
                counts['failed'] += 1
        report['trials'].append(_trial_record(result, stage))
        # Save partial evidence after each paid run, before another one is admitted.
        utils.save_json(report_dir / 'report.json', report)
        return result

    error = None
    try:
        for group in plan['groups']:
            model, track = group['model'], group['track']
            screen = [trial(group, lr, head, plan['seeds'][0], ('rte',), 'screen')
                      for lr in group['rates'] for head in group['head_multipliers']]
            finalists = choose_finalists(screen)
            if len(finalists) < 2:
                report['groups'].append({'model': model, 'track': track, 'status': 'insufficient_stable_rates'})
                continue
            candidates = []
            for finalist in finalists:
                confirmed = [trial(group, finalist['lr'], finalist['head_lr_multiplier'], seed,
                                   config.TASK_KEYS, 'confirm') for seed in plan['seeds']]
                candidates.append(summarize_candidate(confirmed, plan['seeds']))
            winner = choose_winner(candidates)
            info = {'model': model, 'track': track, 'candidates': candidates,
                    'winner': winner, 'status': 'selected' if winner else 'no_healthy_candidate'}
            report['groups'].append(info)
            if winner:
                info['search_boundary_warning'] = winner['lr'] in (min(group['rates']), max(group['rates']))
                endpoint_ok = True
                info['endpoints'] = []
                if track == 'lora':
                    for rank in plan['endpoint_ranks']:
                        checked = trial({**group, 'rank': rank}, winner['lr'], winner['head_lr_multiplier'],
                                        plan['seeds'][0], ('rte',), 'endpoint')
                        ok = checked['status'] == 'completed'
                        endpoint_ok &= ok
                        info['endpoints'].append({'rank': rank, 'run_id': checked['run_id'],
                            'numerically_stable': ok, 'above_majority': bool(ok and checked['A'][0][0]
                                                    > checked['data']['rte']['majority_eval_accuracy'])})
                if endpoint_ok:
                    selected.setdefault(model, {})[track] = {
                        'lr': winner['lr'], 'head_lr_multiplier': winner['head_lr_multiplier'],
                        'calibration_id': plan['calibration_id']}
                else:
                    info['status'] = 'endpoint_instability'
            # Avoid retaining all tokenizers/datasets on a small rental CPU host.
            data_cache.clear()
        expected = {(g['model'], g['track']) for g in plan['groups']}
        actual = {(m, t) for m, tracks in selected.items() for t in tracks}
        report['status'] = 'failed' if counts['failed'] else 'complete' if expected == actual else 'no_eligible_candidate'
        report['selected'] = selected
        if report['status'] == 'complete':
            if (root / 'protocol.json').exists():
                raise ValueError('Main controls appeared during calibration; refusing to change defaults')
            previous = load_defaults(spec)
            merged = {m: dict(tracks) for m, tracks in previous.items()}
            for model, tracks in selected.items():
                merged.setdefault(model, {}).update(tracks)
            payload = {'schema': SCHEMA, 'status': 'complete', 'context': calibration_context(spec),
                       'selected': merged, 'evidence': str(report_dir.relative_to(root) / 'report.json')}
            payload['sha256'] = digest(payload)
            # Evidence is written first; only a complete selection can become active.
            pending_payload = payload
    except AdmissionStop as exc:
        report['status'], report['stop_reason'] = 'deferred', str(exc)
    except Exception as exc:
        report['status'], report['error'] = 'failed', repr(exc)
        report['traceback'] = traceback.format_exc()
        error = exc
    finally:
        report['counts'] = counts
        report['elapsed_hours'] = (time.monotonic() - start) / 3600
        report['estimated_rental_usd'] = report['elapsed_hours'] * spec.hourly_rate if spec.hourly_rate else None
        report['hardware'] = hardware_cache
        report['completed_trial_wall_seconds'] = sum(r.get('train_time_s') or 0 for r in report['trials'])
        utils.save_json(report_dir / 'report.json', report)
        report['charts'] = write_report(report, report_dir, calroot)
        if pending_payload is not None:
            report['activated'] = True
            utils.save_json(report_dir / 'report.json', report)
            utils.save_json(root / 'lr_defaults.json', pending_payload)
            # Refresh the human report's activation status without regenerating figures.
            text = (report_dir / 'report.md').read_text()
            (report_dir / 'report.md').write_text(text.replace('Defaults activated: **False**',
                                                             'Defaults activated: **True**'))
        utils.save_json(report_dir / 'report.json', report)
        utils.save_json(calroot / 'latest.json', {'report': str(report_dir.relative_to(calroot) / 'report.json')})
        utils.save_json(calroot / 'batches' / f'{uuid.uuid4().hex}.json', {
            'calibration_id': plan['calibration_id'], 'status': report['status'], 'counts': counts,
            'elapsed_hours': report['elapsed_hours'], 'estimated_rental_usd': report['estimated_rental_usd']})
    if error:
        raise error
    return {'status': report['status'], 'activated': report['activated'], **counts,
            'report': str(report_dir / 'report.json'), 'charts': report['charts'],
            'defaults': str(root / 'lr_defaults.json') if report['activated'] else None,
            'selected': selected, 'elapsed_hours': report['elapsed_hours']}
