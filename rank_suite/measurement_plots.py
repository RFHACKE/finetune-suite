"""Additional per-task and spectral plots; the run is the replication unit."""
from __future__ import annotations

import numpy as np

from . import config, metrics
from .figures import plt as plotting, save


def measurement_figures(frame, results, directory):
    plt = plotting()
    names = []
    for model in sorted(frame['model'].unique()):
        lora = frame[(frame.model == model) & (frame.track == 'lora')]
        if lora.empty:
            continue
        full = frame[(frame.model == model) & (frame.track == 'full')
                     & np.isclose(frame.lr, metrics.declared_lr(frame, model, 'full'), rtol=1e-10, atol=0)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        if full.empty:
            axes[0].text(.5, .5, 'Full-FT baseline unavailable', ha='center', va='center',
                         transform=axes[0].transAxes)
        for lr, group in lora.groupby('lr'):
            ranks = sorted(group['rank'].unique())
            for task_index, task in enumerate(config.TASK_KEYS):
                for mode, ax in enumerate(axes):
                    if mode and task_index == 2:
                        continue
                    col = f'A_{task}_{task}'
                    points, errors, counts = [], [], []
                    for rank in ranks:
                        part = group[group['rank'] == rank].set_index('seed')
                        if mode:
                            values = part[f'A_{config.TASK_KEYS[-1]}_{task}'] - part[col]
                        else:
                            ref = full.set_index('seed')
                            common = sorted(set(part.index) & set(ref.index))
                            values = part.loc[common, col] - ref.loc[common, col] if common else part[col].iloc[:0]
                        points.append(float(values.mean()) * 100 if len(values) else np.nan)
                        errors.append(float(values.std(ddof=1)) * 100 if len(values) > 1 else np.nan)
                        counts.append(len(values))
                    x, y, e = np.asarray(ranks), np.asarray(points), np.asarray(errors)
                    ax.plot(x, y, marker='o', label=f'{task.upper()}, LR {lr:.2e}')
                    valid = np.isfinite(e)
                    ax.fill_between(x, y - e, y + e, where=valid, alpha=.12)
        for ax, title in zip(axes, ('Learning gap to full-FT diagonal', 'Final-minus-diagonal retention')):
            ax.axhline(-2, color='red', ls='--', label='supplementary −2pp gate')
            ax.set_xscale('log', base=2)
            # A partial LoRA-only cohort has no finite learning-gap y values.
            # Explicit positive limits keep log minor ticks valid in that case.
            ax.set_xlim(float(lora['rank'].min()) / 1.4, float(lora['rank'].max()) * 1.4)
            ax.set_xticks(sorted(lora['rank'].unique()), [str(int(r)) for r in sorted(lora['rank'].unique())])
            ax.set_xlabel('LoRA rank')
            ax.set_ylabel('Change (pp; mean ± sample seed SD when n≥2)')
            ax.set_title(title)
            ax.legend(fontsize=6)
        fig.suptitle(f'{model}: every-task learning and retention (descriptive)')
        names.append(save(fig, directory, f'all_task_gates_{model}.png'))
        rows = []
        for result in results:
            if result['model'] != model or result['track'] != 'lora':
                continue
            final = result.get('intruder', {}).get('after_' + result['task_order'][-1], {})
            for kind in config.INTRUDER_MATRICES:
                matrices = [d for name, d in final.get('diagnostics_by_matrix', {}).items() if name.endswith(kind)]
                if not matrices:
                    continue
                rows.append({'rank': result['rank'], 'lr': result['lr'], 'seed': result['seed'], 'type': kind,
                    'relative_update': np.mean([d['update']['relative_frobenius'] for d in matrices]),
                    'capacity_used': np.mean([d['update']['effective_rank_fraction'] for d in matrices]),
                    'subspace_k10': np.mean([d['subspace']['10']['distance'] for d in matrices]),
                    'subspace_k20': np.mean([d['subspace']['20']['distance'] for d in matrices])})
        if not rows:
            continue
        import pandas as pd
        spectral = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, key, title in zip(axes, ('relative_update', 'capacity_used', 'subspace_k10'),
                                 ('Relative Frobenius update', 'Effective rank / nominal rank', 'Leading-subspace distance, k=10')):
            for (lr, kind), group in spectral.groupby(['lr', 'type']):
                summary = group.groupby('rank')[key].agg(['mean', 'std']).sort_index()
                x, y, error = summary.index.to_numpy(), summary['mean'].to_numpy(), summary['std'].to_numpy()
                ax.plot(x, y, marker='o', label=f'{kind}, LR {lr:.2e}')
                ax.fill_between(x, y - error, y + error, where=np.isfinite(error), alpha=.15)
            ax.set_title(title)
            ax.set_xlabel('LoRA rank')
            ax.set_ylabel('Mean across matrices, then mean ± seed SD')
            ax.legend(fontsize=6)
        fig.suptitle(f'{model}: final-boundary spectral diagnostics (run-level replication)')
        names.append(save(fig, directory, f'update_capacity_subspace_{model}.png'))
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for (lr, kind), group in spectral.groupby(['lr', 'type']):
            for k, style in ((10, '-'), (20, '--')):
                summary = group.groupby('rank')[f'subspace_k{k}'].mean().sort_index()
                ax.plot(summary.index, summary.values, style, marker='o', label=f'{kind}, k={k}, LR {lr:.2e}')
        ax.set_xlabel('LoRA rank')
        ax.set_ylabel('Leading-subspace distance (mean across runs/matrices)')
        ax.set_title(f'{model}: k=10/20 sensitivity at the final boundary')
        ax.legend(fontsize=6)
        names.append(save(fig, directory, f'subspace_sensitivity_{model}.png'))
    return names
