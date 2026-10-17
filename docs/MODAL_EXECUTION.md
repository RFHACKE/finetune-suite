# Requested Modal execution

The requested cohort fixes the four methodology checkpoints and expands the LoRA ranks to **4, 8, 16, 32, 64, 128, 256**. Five seeds (42, 123, 456, 789, 1024) give **140 complete RTE → MNLI → SST-2 sequences**. Each main task retains 2,490 training examples, three epochs and its complete official validation split. There is no full-FT baseline in this requested execution; baseline-dependent feasible windows cannot be certified from this cohort alone.

The [Modal runner](../tools/modal_experiments.py) uses the pinned package environment with CUDA 12.8, SDPA and bf16 computation. Trainable adapter/head parameters and AdamW moments remain fp32. A micro-batch of 16 and accumulation of one preserve effective batch 16, with the partition recorded consistently in calibration and main results. No quantization, reduced main validation, reduced task data, shortened epochs or substituted checkpoints are used to save credit.

The training-only LR/head calibration is the procedure in [PROTOCOL_V3.md](PROTOCOL_V3.md), restricted to LoRA: at most 48 cases and 30,000 optimizer updates. Its endpoint checks follow the supplied rank range, here 4 and 256. The default CLI's original range still checks 4 and 64. The selected rate is the best eligible tested policy under the declared selection objective, rather than a claim of a universal optimum for every rank or task.

CPU preparation downloads all pinned weights and tokenizes the training/validation data before GPU allocation. A separate performance benchmark uses 192 training examples and one epoch at ranks 4 and 256; it is excluded from LR selection and main scientific results. GPU type and time allowances are chosen after measuring that benchmark and checking current Modal rates. The authorization is $30 total, including preparation; worker timeouts and whole-run admission limits bound spending. Incomplete work is retained and reported explicitly if that envelope cannot cover the requested design.

Only `rank_suite/`, `requirements.txt` and the Modal runner enter the image. `hf.env` stays local; its token is supplied through an ephemeral Modal secret. `notify.env` is not used. No messaging service is invoked. Model cache and scientific results live on separate durable Modal volumes. Every worker has a private local directory for Linux locks; incremental copies preserve task-boundary recovery checkpoints and completed results on the output volume.

Staged commands, run from the repository root with Modal already authenticated:

```bash
modal run --detach tools/modal_experiments.py --stage prefetch
modal run --detach tools/modal_experiments.py --stage benchmark --seconds 600
# Supply measured, budgeted seconds per model; four workers run concurrently.
modal run --detach tools/modal_experiments.py --stage calibration --seconds SECONDS
modal run --detach tools/modal_experiments.py --stage merge-defaults
modal run --detach tools/modal_experiments.py --stage main --seconds SECONDS
modal run --detach tools/modal_experiments.py --stage consolidate
```

`--models` accepts a comma-separated subset of fixed registry keys; `--gpu` defaults to H100. `FYP_MODAL_RUN` selects the remote run directory and defaults to `frw-20261018`. Stages do not automatically advance; their completed evidence and measured costs must be reviewed before admitting the next stage. Local call identifiers are saved under `Finetune-result/execution/`. The Modal SDK `FunctionCall.from_id(id).get(timeout=0)` can check them after a client interruption.

Merged defaults retain the original activated calibration reports for all four models. All main workers load that same policy before freezing their protocol. Consolidation verifies common experimental controls and writes a separate `requested-completeness.json` for the 140-sequence requested scope; the original suite audit still reports the absent full-FT design honestly. Retrieve the run volume into `Finetune-result`, retaining raw predictions, step logs, adapters, heads, projection factors, calibration JSON and charts. CPU spectral analysis and aggregate figures can then use the saved tensors without renting a GPU.
