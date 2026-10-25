"""GLUE data for frw-v4.

Selection is identical to v3: each task's 2,490 training examples are drawn
once with seed 42 (RTE uses its whole training split). LR-search jobs fit on a
stratified 80% of those examples and score the remaining 20% (the same split as
the v3 calibration); reported jobs fit on all 2,490 and score the full official
validation split. Official validation never influences any choice.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from . import config as C


@dataclass
class Split:
    input_ids: list
    labels: np.ndarray
    source_index: list

    def __len__(self):
        return len(self.labels)


@dataclass
class TaskData:
    task: C.Task
    train: Split
    eval: Split
    provenance: dict


def _draw(n_total: int, n: int, seed: int) -> list:
    if n > n_total:
        raise ValueError(f"requested {n} of {n_total}")
    if n == n_total:
        return list(range(n_total))
    return sorted(np.random.RandomState(seed).permutation(n_total)[:n].tolist())


def _encode(rows, task: C.Task, tok) -> list:
    if task.pair:
        out = tok(["Premise: " + t for t in rows[task.fields[0]]],
                  ["\nHypothesis: " + t for t in rows[task.fields[1]]],
                  truncation="longest_first", max_length=C.MAX_LEN)
    else:
        out = tok(["Text: " + t for t in rows[task.fields[0]]], truncation=True, max_length=C.MAX_LEN)
    return out["input_ids"]


def _digest(values) -> str:
    return hashlib.sha256(json.dumps(list(map(int, values))).encode()).hexdigest()


def _raw_from_hub(task: C.Task):
    from datasets import load_dataset
    raw = load_dataset("nyu-mll/glue", task.glue, revision=C.DATA_REVISION)
    train_all = raw["train"]
    idx = _draw(len(train_all), C.TRAIN_SAMPLES, C.DATA_SEED)
    sub = train_all.select(idx).add_column("__src", idx)
    ev = raw[task.eval_split]
    return sub, ev.add_column("__src", list(range(len(ev))))


def export_raw(directory) -> dict:
    """Freeze the exact rows used by every job (train subset + full official eval) as JSON."""
    import os
    from pathlib import Path
    out = {}
    Path(directory).mkdir(parents=True, exist_ok=True)
    for key, task in C.TASKS.items():
        sub, ev = _raw_from_hub(task)
        cols = list(task.fields) + ["label", "__src"]
        payload = {"task": key, "revision": C.DATA_REVISION, "data_seed": C.DATA_SEED,
                   "label_names": sub.features["label"].names,
                   "train_subset": {c: list(sub[c]) for c in cols}, "eval": {c: list(ev[c]) for c in cols}}
        text = json.dumps(payload)
        path = Path(directory) / f"{key}.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
        out[key] = hashlib.sha256(text.encode()).hexdigest()
    return out


def _raw(task: C.Task):
    import os
    from pathlib import Path
    d = os.environ.get("FRW_DATA_DIR")
    path = Path(d) / f"{task.key}.json" if d else None
    if path is None or not path.exists():
        return _raw_from_hub(task)
    from datasets import ClassLabel, Dataset
    payload = json.loads(path.read_text())
    labels = ClassLabel(names=payload["label_names"])
    sub = Dataset.from_dict(payload["train_subset"])
    sub = sub.cast_column("label", labels)
    ev = Dataset.from_dict(payload["eval"]).cast_column("label", labels)
    return sub, ev


def load_task(key: str, tok, split: str) -> TaskData:
    """split='holdout' (LR search) or 'official' (reported results)."""
    task = C.TASKS[key]
    sub, official = _raw(task)
    if split == "holdout":
        parts = sub.train_test_split(test_size=C.HOLDOUT_FRACTION, seed=C.DATA_SEED,
                                     stratify_by_column="label")
        fit, ev = parts["train"], parts["test"]
    elif split == "official":
        fit, ev = sub, official
    else:
        raise ValueError(split)
    tr = Split(_encode(fit, task, tok), np.asarray(fit["label"], dtype=np.int64), list(fit["__src"]))
    va = Split(_encode(ev, task, tok), np.asarray(ev["label"], dtype=np.int64), list(ev["__src"]))
    counts = {int(k): int(v) for k, v in zip(*np.unique(va.labels, return_counts=True))}
    lens = np.asarray([len(x) for x in tr.input_ids])
    prov = {"dataset": "nyu-mll/glue", "config": task.glue, "revision": C.DATA_REVISION,
            "split": split, "n_train": len(tr), "n_eval": len(va),
            "train_source_sha256": _digest(tr.source_index), "eval_source_sha256": _digest(va.source_index),
            "eval_label_counts": counts, "majority_eval_accuracy": max(counts.values()) / len(va),
            "train_len_mean": float(lens.mean()), "train_frac_truncated": float((lens >= C.MAX_LEN).mean())}
    return TaskData(task, tr, va, prov)


def batches(split: Split, batch: int, pad_id: int, shuffle_seed=None, sort=False):
    """Yield (ids[B,L], mask[B,L], labels[B], positions) with right padding to a multiple of 8."""
    import torch
    n = len(split)
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        order = torch.randperm(n, generator=g).tolist()
    elif sort:
        order = sorted(range(n), key=lambda i: len(split.input_ids[i]))
    else:
        order = list(range(n))
    for s in range(0, n, batch):
        pos = order[s:s + batch]
        seqs = [split.input_ids[i] for i in pos]
        L = max(len(x) for x in seqs)
        L = (L + 7) // 8 * 8
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, :len(x)] = torch.tensor(x, dtype=torch.long)
            mask[r, :len(x)] = 1
        yield ids, mask, torch.as_tensor(split.labels[pos]), pos
