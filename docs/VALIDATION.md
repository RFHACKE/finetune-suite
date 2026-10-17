# Validation record

## Modal preservation update — 18 October 2026

Real LoRA-only LR/head calibration has now completed all 48 planned cases for the four pinned models. The expanded 140-sequence main cohort is running on four L40S workers; it is not yet a completed scientific result. The scientific package remains frozen at SHA-256 `ff3d682687b8ca5db927a383539cf816691c06a647c96c7b2750048f14d3fb16`. Production full-FT calibration/baselines are outside this requested cloud cohort.

The private CPU-only preservation observer has been deployed and started from a client that exits immediately. Its first durable remote inventory, failure records, retry candidates and source manifest have been retrieved successfully. All four pretrained projection caches were checked for remote presence. The preservation checks also passed against 15 actual completed local run exports. The cloud training coordinator and archive packager continued after their local CLI clients were disconnected. Local collection, spectral analysis and finalization are paused until the user returns; the temporary download endpoint is stopped.

The new preservation tests exercise all 140 distinct identities, pending versus failed work, partial task checkpoint reporting, budget-deferred runs, retained tracebacks, archive availability and missing export detection. These tests and the existing protocol/parallel tests passed **20 cases**; Ruff and whitespace checks passed. The earlier authenticated streaming checks verified access rejection, ranged resumption and a complete real archive checksum. These transport and preservation checks do not replace the final requested-cohort numerical/spectral audit. See [OVERNIGHT.md](OVERNIGHT.md) for durable locations and recovery commands.

## Development review — 17 October 2026

The following historical review records software and numerical checks at the development stage, before the Modal execution.

The v3 full CPU regression run and final targeted checks cover **58 passing CPU cases**; **six CUDA integration cases were skipped** in the validated CPU environment. The full pass completed with 56 passing tests; subsequent targeted runs covered the two added guards and revalidated the changed calibration/measurement paths. The parallel launcher also completed a real two-worker planning dry run during the original suite validation.

## V3 implementation checks

- Executed the complete 88-case selection workflow with explicitly synthetic scoring fixtures for all four fixed model keys and both tracks. Verified different winning rates per model/track, automatic head/rate defaults, 32 saved charts, no retraining on repeat, training-only selection, failure/budget behavior and matching evidence checks. Synthetic fixture outputs remain in temporary test directories.
- Verified forced recalibration preserves the existing active report and activates a separate replacement report only after success. Main/pilot controls prevent recalibration in an already-started cohort.
- Executed actual tiny Qwen calibration training, recording NLL and scheduled head/backbone LR ratios without permanent model/projection exports. Accumulation equivalence covers head ratios 1 and 5; real bitsandbytes parameter-group configuration keeps matrix moments at 8-bit and head/vector moments at fp32.
- Compared QR-derived LoRA update spectra against dense full-update SVDs and reciprocal factor rescaling; checked exact zero updates. Verified subspace invariance under basis rotation and sensitivity to a changed leading span.
- Executed real tiny-model task-boundary export → deferred spectra → long JSON/CSV → all three new measurement-chart paths. A LoRA-only partial-cohort plotting failure was found and fixed; final targeted analysis checks passed.
- Checked all-task gates against hidden cancellation, underlearned references, fixed tolerances and calibrated baseline anchors. Verified explicit disjoint calibration split indices.
- Checked all **30 CLI flag spellings** against command-specific documentation, parsed 36 runnable usage examples, and confirmed invalid calibration combinations fail before GPU work. Ruff, shell syntax, Word structure/citations and whitespace checks passed.

At that development-review point, production calibration had not been performed. The development GPU had insufficient VRAM for the full fixed-model/full-FT scope, and the validated Python environment used CPU PyTorch. That review made no claims about winning production LRs, GPU throughput, scientific accuracy or rental costs.

## Environment and checks

The final CPU stack was installed and checked in an isolated `/tmp` environment: Python 3.10.12, torch 2.8.0+cpu, transformers 4.57.6, PEFT 0.17.1, bitsandbytes 0.48.2, datasets 4.4.1, NumPy 2.2.6, SciPy 1.15.3 and the pinned requirements. `setup.sh --cpu` completed in an isolated project copy, including the environment doctor and `pip check`. Shell syntax, Ruff and whitespace checks passed.

The tests cover full grid accounting, pilot reuse, stable seed-block shards, immutable manifests, data-selection reuse, holdout isolation, signed BWT versus peak forgetting, disconnected windows, missing baselines/seeds, fixed baseline LR, seed pairing, LR-specific spectral correlations, unchanged/permuted/rotated singular-vector cases, dense-SVD/eigensolver agreement, corrupt base-cache recovery, correct plot units and model-specific grids, partial reports, actual optimizer configuration, and resumed adapter tensor equality.

Real tiny Llama, Qwen2 and Granite architectures each execute all three task boundaries with both LoRA and non-embedding full FT on CPU. Synthetic test accuracies and figures remain in temporary test directories and are not bundled as study results. GPU versions of the six architecture/track integration cases are included and skip when CUDA is unavailable.

The local CUDA package installation was attempted but not completed because large wheel downloads were too slow on the development connection. **CUDA kernels and production-size 4090/5090 memory/throughput remained unverified in this development environment.** The installer runs a real bf16/optimizer check, and the full-size pilot is the required acceptance test on a rented instance. No cloud rental was created during that development validation.

## Real data-pipeline checks

The production pinned Qwen2.5-1.5B tokenizer and pinned `nyu-mll/glue` revision were downloaded and exercised without downloading model weights. Observed main selections were:

| Task | Training count | Full validation count | Training label counts |
|---|---:|---:|---|
| RTE | 2,490 | 277 | 0: 1,249; 1: 1,241 |
| MNLI matched | 2,490 | 9,815 | 0: 793; 1: 829; 2: 868 |
| SST-2 | 2,490 | 872 | 0: 1,071; 1: 1,419 |

Repeated RTE preparation reused identical selection metadata successfully. The training-only calibration path produced 1,992 training and 498 held-out examples, with no official validation examples used for calibration. Token lengths and majority-class references are recorded in data provenance.

## CPU spectral microbenchmark (v2 numerical path)

Four BLAS threads, seeded random float32 matrices at the actual projection dimensions; full fp32 SVD versus float64 Gram + top-ten subset eigensolve. These are **synthetic matrices**, not timing measurements on pretrained checkpoints or estimates of paid GPU throughput.

| Matrix shape | Full SVD, seconds | Top-vector eigensolve, seconds |
|---|---:|---:|
| 256 × 2048 | 0.076 | 0.020 |
| 512 × 2048 | 0.195 | 0.030 |
| 1536 × 1536 | 0.759 | 0.293 |
| 2048 × 2048 | 2.235 | 0.736 |

The v3 analyzer additionally computes top-21 tuned vectors, update spectra and k=10/20 subspace diagnostics; the table is not a measured v3 end-to-end runtime.

Absolute leading-vector alignment exceeded 0.9999997 in every case, apart from floating-point rounding at unity. At 2048 square the subset method was about three times faster on this CPU. End-to-end spectral time also includes reference decompositions, matrix reconstruction, checksums and I/O, and varies across hardware and spectra.

Full-study accuracy, optimal LRs, feasible ranks, intruder correlations, GPU-hour estimates and total rental cost require actual main experiments. No values for those outcomes are asserted by this validation record.
