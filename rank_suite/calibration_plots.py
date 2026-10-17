"""Human-reviewable calibration charts; all numeric chart inputs remain JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .figures import plt as plotting, save


def write_report(report, directory, calroot):
    directory, calroot = Path(directory), Path(calroot)
    directory.mkdir(parents=True, exist_ok=True)
    charts_dir = directory / 'charts'
    charts_dir.mkdir(exist_ok=True)
    plt = plotting()
    charts = []
    trials = report['trials']
    for group in report['plan']['groups']:
        model, track = group['model'], group['track']
        stem = f'{model}_{track}'
        rows = [r for r in trials if r['model'] == model and r['track'] == track]
        screen = [r for r in rows if r['stage'] == 'screen']
        if screen:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for head in group['head_multipliers']:
                good = sorted([r for r in screen if r['status'] == 'completed'
                               and r['head_lr_multiplier'] == head], key=lambda r: r['lr'])
                if not good:
                    continue
                x = [r['lr'] for r in good]
                axes[0].plot(x, [r['A'][0][0] * 100 for r in good], marker='o', label=f'head LR ×{head:g}')
                axes[1].plot(x, [r['NLL'][0][0] for r in good], marker='o', label=f'head LR ×{head:g}')
            for ax, ylabel in zip(axes, ('RTE holdout accuracy (%)', 'RTE holdout NLL')):
                ax.set_xscale('log')
                ax.set_xlabel('Backbone/adapter peak learning rate')
                ax.set_ylabel(ylabel)
                if ax.lines:
                    ax.legend(fontsize=8)
            failed = sum(r['status'] != 'completed' for r in screen)
            fig.suptitle(f'{model} / {track}: screening (training holdout); non-completed: {failed}')
            charts.append('charts/' + save(fig, charts_dir, f'screen_{stem}.png'))
        info = next((g for g in report['groups'] if g['model'] == model and g['track'] == track), {})
        candidates = [c for c in info.get('candidates', []) if c['complete']]
        if candidates:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
            names = [f"LR {c['lr']:.2e}\nhead ×{c['head_lr_multiplier']:g}"
                     + ('\nlearning check failed' if not c['eligible'] else '') for c in candidates]
            x = np.arange(len(candidates))
            axes[0].bar(x, [c['mean_final_accuracy'] * 100 for c in candidates],
                        yerr=[c['sd_final_accuracy'] * 100 for c in candidates], capsize=4, alpha=.6)
            for i, c in enumerate(candidates):
                for seed in c['per_seed']:
                    axes[0].scatter(i, seed['final_mean_accuracy'] * 100, color='black', s=20)
                    axes[1].scatter(seed['mean_diagonal'] * 100, -seed['bwt'] * 100,
                                    label=f"candidate {i + 1}, seed {seed['seed']}")
            axes[0].set_xticks(x, names, fontsize=8)
            axes[0].set_ylabel('Mean final accuracy (%) ± sample seed SD')
            axes[1].set_xlabel('Mean diagonal / learning (%)')
            axes[1].set_ylabel('−BWT (pp; lower is better retention)')
            axes[1].legend(fontsize=6)
            fig.suptitle(f'{model} / {track}: full-sequence training-holdout confirmation')
            charts.append('charts/' + save(fig, charts_dir, f'confirmation_{stem}.png'))
        winner = info.get('winner')
        if winner:
            chosen = [r for r in rows if r['run_id'] in winner['run_ids'] and r['status'] == 'completed']
            values = np.stack([np.asarray([[np.nan if v is None else v for v in row] for row in r['A']]) for r in chosen])
            # Future-task entries are unobserved, so do not fill them with zeros.
            matrix = np.full((3, 3), np.nan)
            for i in range(3):
                for j in range(i + 1):
                    matrix[i, j] = values[:, i, j].mean() * 100
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            image = ax.imshow(np.ma.masked_invalid(matrix), cmap='viridis', vmin=0, vmax=100)
            for i in range(3):
                for j in range(i + 1):
                    ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', color='white')
            ax.set_xticks(range(3), [t.upper() for t in config.TASK_KEYS])
            ax.set_yticks(range(3), ['after ' + t.upper() for t in config.TASK_KEYS])
            ax.set_xlabel('Evaluated task (training holdout)')
            ax.set_title(f'{model} / {track}: selected candidate\nLR={winner["lr"]:.2e}, head ×{winner["head_lr_multiplier"]:g}')
            fig.colorbar(image, ax=ax, label='Mean accuracy (%)')
            charts.append('charts/' + save(fig, charts_dir, f'selected_matrix_{stem}.png'))
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
            for result in chosen:
                log = calroot / 'runs' / result['run_id'] / 'train_log.jsonl'
                records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
                for ax, task in zip(axes, config.TASK_KEYS):
                    points = [r for r in records if r['task'] == task]
                    if points:
                        ax.plot(range(1, len(points) + 1), [r['loss'] for r in points],
                                alpha=.7, label=f'seed {result["seed"]}')
                    ax.set_title(task.upper())
                    ax.set_xlabel('Optimizer step within task')
                    ax.set_ylabel('Training cross entropy')
                    if ax.lines:
                        ax.legend(fontsize=7)
            fig.suptitle(f'{model} / {track}: selected-candidate training curves')
            charts.append('charts/' + save(fig, charts_dir, f'selected_loss_{stem}.png'))
    text = ['# Learning-rate calibration review', '',
            f"Status: **{report['status']}**. Defaults activated: **{report['activated']}**.", '',
            'Selection uses only stratified 20% holdouts from the fixed training subsets. '
            'Official validation is excluded. These are the best tested finalists under the declared rule, '
            'not guaranteed global optima. A fixed common LoRA rate is used across main ranks.', '',
            'Screening compares rates and head multipliers on RTE; the two best distinct rates '
            'are confirmed on RTE → MNLI → SST-2 across the declared seeds. Every task must '
            'beat its holdout majority baseline in mean diagonal accuracy. Mean final-row accuracy '
            'is the selection objective, with declared deterministic tie breakers. '
            'Endpoint underlearning is reported without deleting difficult ranks.', '',
            '| Model | Track | Selected LR | Head multiplier | Final mean | Seed SD | Status |',
            '|---|---|---:|---:|---:|---:|---|']
    for group in report['groups']:
        c = group.get('winner')
        text.append(f"| {group['model']} | {group['track']} | " +
                    (f"{c['lr']:.4g} | {c['head_lr_multiplier']:g} | {100*c['mean_final_accuracy']:.2f}% | "
                     f"{100*c['sd_final_accuracy']:.2f}pp | {group['status']} |" if c else f"— | — | — | — | {group['status']} |"))
    text += ['', 'The sample SD is descriptive; two seeds provide limited uncertainty resolution. '
             'Inspect individual seed scores, all task diagonals, BWT, failures, endpoint checks, '
             'and any search-boundary warning in report.json. Calibration runs are excluded from main results.', '']
    text.extend(f'![{Path(name).stem}]({name})\n' for name in charts)
    (directory / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    return charts
