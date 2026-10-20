# LoRA checkpoint storage and reconstruction

New LoRA runs save the standard `adapter_model.safetensors`,
`adapter_config.json`, `heads.pt`, and a `projection-storage.json` manifest at
every task boundary. The adapter contains the exact FP32 q/v LoRA A/B factors.
The manifest records file checksums, a serialization-independent factor
checksum, scaling and run-identity provenance. CPU spectral analysis reads the
adapter directly; it does not create `projections.pt` or require a GPU, a model
download, a Hugging Face token, or retraining.

Full fine-tuning still needs `projections.pt`: its q/v updates are not LoRA
factors. Calibration continues to omit permanent weight exports. Training
hyperparameters, task order, evaluation, heads and recovery are unchanged.

## Compact existing downloaded LoRA runs

Use the CPU environment created by `./setup.sh --cpu`:

```bash
# Verify all 180 main/extension sequences, without deleting snapshots.
./run.sh checkpoints verify --output Finetune-result Finetune-result/large-ranks

# Reverify the complete selected set, then remove only projections.pt.
./run.sh checkpoints prune --output Finetune-result Finetune-result/large-ranks
```

The two cohort directories must be listed explicitly; this command does not
search unrelated studies, calibration or benchmark folders. Only completed
LoRA runs in the `main` or `smoke` mode are eligible. It takes each run's
existing lock, verifies the original projection SHA-256, and compares **all
factor bytes, tensor names, dtypes, shapes, scaling and run identity**. One
mismatched pair aborts the verification phase before any selected projection
is removed. Heads must also exist. Source checksums are rechecked before the
first deletion. The command never removes adapters, heads, pretrained base
projections, predictions, training logs, spectral results or reports.

Every checkpoint gets an atomic, flushed storage manifest. Each cohort also
gets timestamped `execution/checkpoint-storage/<operation>.jsonl` and `.json`
records, including verification, deletion intent, deleted bytes, completion
and failures. Repeating `prune` accepts already compacted, verified snapshots
and continues safely. After an interruption, rerun the same command. A missing
projection **without** a valid manifest remains an error.

Original `result.json`, protocol, source records, transport manifests and
historical file-size/checksum inventories are **not rewritten**. Updated local
audits recognize the attested replacement; an old version of the suite that
requires the original exported files will report them missing. Historical
export byte counts describe the original two-file storage, not current disk
usage. Keep the storage manifests alongside the retained adapters.

## Restore a physical projections.pt

Analysis already reconstructs factors in memory. For another program that
requires a `.pt` file, restore a selected checkpoint:

```bash
./run.sh checkpoints restore --output Finetune-result \
  --run-id YOUR_COMPLETE_RUN_ID --tasks rte
```

This uses CPU and disk only, checks the retained files, writes the `.pt`
atomically, reloads it and verifies its bit-exact factor content. The restored
file has the same logical payload: `track`, `identity`, and `matrices` with
`A`, `B`, and `scale`. Serialization bytes and SHA-256 need not match the
historical `.pt`; the new materialization checksum is recorded separately.
The historical checksum and existing spectral cache identities remain intact.
Older audit code that knows only the historical file checksum may reject the
restored file; use the updated readers.

Omitting filters restores **every** completed LoRA boundary in the listed
cohorts and consumes substantial disk space. Restore can be followed by
`checkpoints prune` to remove the duplicates again.

| Argument | Default | Meaning |
|---|---|---|
| `verify`, `prune`, or `restore` | Required positional action | Verify without deleting; verify then remove redundant projections; or materialize compatible projections. |
| `--output DIR [DIR ...]` | `results` | One or more separate cohort roots, each containing `runs/`. Duplicate roots are rejected. |
| `--run-id ID [ID ...]` | All eligible runs | Exact completed LoRA run IDs. Unknown/ineligible IDs cause an error. |
| `--tasks rte mnli sst2` | All task boundaries | One or more task names to select. |
| `-h`, `--help` | — | Show command help. |

## Future dual exports and downloads

`run` and `plan` accept `--save-lora-projections` (default **off**) to retain an
additional `.pt` export during future LoRA training. It adds disk/I/O cost but
does not change training. Full-FT projection exports are always retained.
Changed scientific source code still requires a fresh output cohort for new
training; storage maintenance works on existing cohorts without changing their
recorded experimental controls.

The archive packager and collectors accept both formats. The downloader also
recognizes a locally compacted legacy projection and checks its retained source
hashes instead of downloading that redundant file again. A corrupt source is
reported as an error. This local cleanup does not delete or alter remote
archives; their original transport checksums remain valid.

Keep **all three task-boundary adapters/configs/heads**, their manifests, and the
pretrained `cache/projections/.../base.pt` files. A final adapter alone cannot
recover earlier task-boundary weights. The base projections are still required
for spectral comparisons, although not for reconstructing a LoRA factor `.pt`.
