"""GLUE data pipeline (methodology Sec. 5.1).

- Source: nyu-mll/glue via Hugging Face datasets.
- Each task's 2,490-sample training subset is drawn once with DATA_SEED (42)
  and shared by every run of every track and seed, so data selection can
  never differ between configurations. The drawn indices are persisted to the
  results directory for the reproducibility record.
- RTE / MNLI use the tokenizer's pair encoding; SST-2 uses single-sentence
  encoding. Max length 256, right truncation, dynamic right padding.
- Evaluation uses the official validation sets in full (no subsampling).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .utils import get_logger, save_json
from .protocol import digest, immutable_json

logger = get_logger("rank_suite.data")


@dataclass
class TaskData:
    spec: "config.TaskSpec"
    train_dataset: object
    eval_dataset: object
    train_indices: list
    label_counts_train: dict
    fingerprint: str = ""
    provenance: dict | None = None

    @property
    def n_train(self) -> int:
        return len(self.train_dataset)

    @property
    def n_eval(self) -> int:
        return len(self.eval_dataset)


class PadCollator:
    """Dynamic right padding; labels collected into a LongTensor."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch

        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        ids = [f["input_ids"] for f in features]
        mask = [f["attention_mask"] for f in features]
        batch = self.tokenizer.pad(
            {"input_ids": ids, "attention_mask": mask},
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


def _draw_indices(n_total: int, n_draw: int, data_seed: int) -> list:
    """Deterministic subset indices; recorded so runs can be reproduced."""
    if n_draw > n_total:
        raise ValueError(f"Requested {n_draw} samples but only {n_total} exist")
    if n_total == n_draw:
        return list(range(n_total))
    rng = np.random.RandomState(data_seed)  # legacy RNG: stable across versions
    perm = rng.permutation(n_total).tolist()
    return sorted(perm[:n_draw])


def _tokenize_split(ds, spec, tokenizer, max_len):
    def fn(ex):
        if spec.pair:
            out = tokenizer(
                ["Premise: " + t for t in ex[spec.text_fields[0]]],
                ["\nHypothesis: " + t for t in ex[spec.text_fields[1]]],
                truncation="longest_first",
                max_length=max_len,
            )
        else:
            out = tokenizer(
                ["Text: " + t for t in ex[spec.text_fields[0]]],
                truncation=True,
                max_length=max_len,
            )
        # Decoder base tokenizers need not insert any separator for text pairs.
        # Explicit field markers prevent the premise and hypothesis running together.
        out.pop("token_type_ids", None)
        out["labels"] = ex["label"]
        return out

    return ds.map(fn, batched=True, remove_columns=ds.column_names, desc=f"tokenize {spec.key}")


def prepare_task_data(
    tokenizer,
    task_keys=None,
    data_seed: int = config.DATA_SEED,
    n_train: int = config.TRAIN_SAMPLES,
    max_len: int = config.MAX_LEN,
    eval_limit: int | None = None,
    selection_dir: Path | None = None,
    calibration: bool = False,
) -> dict:
    """Load, subset and tokenize all requested tasks for one tokenizer."""
    from datasets import load_dataset

    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    task_keys = list(task_keys or config.TASK_KEYS)
    out = {}
    for key in task_keys:
        spec = config.TASK_BY_KEY[key]
        raw = load_dataset("nyu-mll/glue", spec.glue_name, revision=config.DATA_REVISION)
        train_split, eval_split = raw["train"], raw[spec.eval_split]

        indices = _draw_indices(len(train_split), n_train, data_seed)
        train_raw = train_split.select(indices)

        # Sensitivity runs never inspect the official validation set. The split is
        # stratified inside the fixed training subset, independently of run seeds.
        if calibration:
            holdout = train_raw.train_test_split(test_size=0.2, seed=config.DATA_SEED,
                                                stratify_by_column="label")
            train_raw, eval_split = holdout["train"], holdout["test"]

        label_counts = {}
        for lab in train_raw["label"]:
            label_counts[int(lab)] = label_counts.get(int(lab), 0) + 1

        train_ds = _tokenize_split(train_raw, spec, tokenizer, max_len)
        eval_raw = eval_split
        if eval_limit is not None and len(eval_raw) > eval_limit:
            eval_raw = eval_raw.select(range(eval_limit))
        eval_ds = _tokenize_split(eval_raw, spec, tokenizer, max_len)
        eval_labels = np.asarray(eval_raw["label"])
        eval_counts = {int(label): int((eval_labels == label).sum())
                       for label in np.unique(eval_labels)}
        def lengths(ds):
            values = np.asarray([len(ids) for ids in ds["input_ids"]])
            return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
                    "fraction_at_max_length": float((values == max_len).mean())}

        td = TaskData(
            spec=spec,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            train_indices=indices,
            label_counts_train=label_counts,
            fingerprint=getattr(train_ds, "_fingerprint", ""),
            provenance={"dataset": "nyu-mll/glue", "revision": config.DATA_REVISION,
                        "selected_indices_sha256": digest(indices),
                        "encoding": config.ENCODING_VERSION,
                        "train_fingerprint": getattr(train_ds, "_fingerprint", ""),
                        "eval_fingerprint": getattr(eval_ds, "_fingerprint", ""),
                        "n_train": len(train_ds), "n_eval": len(eval_ds),
                        "calibration": calibration, "label_counts": label_counts,
                        "eval_label_counts": eval_counts,
                        "majority_eval_accuracy": max(eval_counts.values()) / len(eval_labels),
                        "train_lengths": lengths(train_ds), "eval_lengths": lengths(eval_ds)},
        )
        out[key] = td
        logger.info(
            "%s: train=%d (of %d, seed %d), eval=%d, labels=%s",
            key, td.n_train, len(train_split), data_seed, td.n_eval, label_counts,
        )

        if selection_dir is not None:
            immutable_json(
                Path(selection_dir) / f"data_selection_{key}.json",
                {
                    "task": key,
                    "glue_config": spec.glue_name,
                    "source_split": "train",
                    "source_size": len(train_split),
                    "data_seed": data_seed,
                    "n_selected": len(indices),
                    "selected_indices": indices,
                    "label_counts": label_counts,
                    "eval_split": "train_holdout_20_percent" if calibration else spec.eval_split,
                    "eval_size": td.n_eval,
                    "source_fingerprint": getattr(train_split, "_fingerprint", ""),
                    "revision": config.DATA_REVISION,
                },
            )
            # Tokenizer-dependent fingerprints must not overwrite the shared selection.
            tok_key = digest({"tokenizer": tokenizer.name_or_path,
                              "revision": tokenizer.init_kwargs.get("_commit_hash"),
                              "encoding": config.ENCODING_VERSION, "max_len": max_len})[:16]
            save_json(Path(selection_dir) / f"tokens_{key}_{tok_key}.json", td.provenance)
    return out


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    tokenizer,
    pin_memory: bool = True,
):
    """DataLoader with a seeded generator so batch order follows the run seed
    (methodology Sec. 8: seeds govern mini-batch shuffling order)."""
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=PadCollator(tokenizer),
        num_workers=0,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
