"""Command-line entry point. Planning needs only the Python standard library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import config, utils


def _models(values):
    reverse = {m.hf_id: k for k, m in config.MODELS.items()}
    keys = [reverse.get(v, v) for v in (values or config.DEFAULT_MODEL_KEYS)]
    if any(k not in config.MODELS for k in keys):
        raise ValueError("Model selection is fixed by fyp_methodology.docx. Choose: "
                         + ", ".join(config.MODELS))
    return list(dict.fromkeys(keys))


def _rates(values, models):
    if not values:
        return None
    scoped, flat = {}, []
    for token in values:
        if "=" in token:
            key, raw = token.split("=", 1)
            if key not in models:
                raise ValueError(f"LR supplied for unknown/unselected model: {key}")
            scoped[key] = utils.parse_number_list([raw])
            if not scoped[key]:
                raise ValueError(f"Empty learning-rate list for {key}")
        else:
            flat.extend(utils.parse_number_list([token]))
    if flat:
        for m in models:
            scoped.setdefault(m, flat)
    return scoped


def _add_models(p):
    p.add_argument("--models", nargs="+", help="fixed model keys or their exact Hugging Face IDs")
    p.add_argument("--output", default="results", help="experiment directory (default: results)")


def _add_run(p):
    _add_models(p)
    p.add_argument("--test-lr", action="store_true",
                   help="calibrate LR/head rates on training holdouts, save plots and defaults, then exit")
    p.add_argument("--head-lr-multiplier", type=float,
                   help="explicit head/backbone LR ratio (otherwise cohort or project track default); fixes head ratio during LR tests")
    p.add_argument("--track", choices=("lora", "full", "both"), default="both")
    p.add_argument("--mode", choices=("main", "pilot", "sanity", "smoke"), default="main",
                   help="pilot runs reusable main configurations; smoke/sanity are isolated")
    p.add_argument("--ranks", nargs="+", help="default: 4 8 16 32 64")
    p.add_argument("--seeds", nargs="+", help="default: 42 123 456 789 1024")
    p.add_argument("--lora-lrs", nargs="+", help="optional sensitivity rates, globally or key=rate,rate")
    p.add_argument("--full-lrs", nargs="+", help="optional sensitivity rates; fixed default baseline anchors the window")
    p.add_argument("--optimizer", choices=("standard", "adamw"), default="standard",
                   help="standard: LoRA fp32 moments, full FT 8-bit matrix moments; adamw: fp32 states in both tracks")
    p.add_argument("--micro-batch", type=int, default=4, help="positive divisor of 16 (default: 4)")
    p.add_argument("--accum", type=int, help="inferred so effective batch is exactly 16")
    p.add_argument("--eval-batch", type=int, default=config.EVAL_BATCH)
    p.add_argument("--deterministic", action="store_true", help="strict deterministic algorithms; slower")
    p.add_argument("--save-full-weights", action="store_true", help="also retain every full model boundary (large)")
    p.add_argument("--force", action="store_true", help="explicitly replace completed runs")
    p.add_argument("--keep-going", action="store_true", help="continue after run failure (default: fail fast)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0, help="assign complete model/seed blocks to this GPU worker")
    p.add_argument("--hourly-rate", type=float, help="actual USD/hour for the instance serving this worker")
    p.add_argument("--budget", type=float, help="optional per-invocation scheduling allowance in USD")
    p.add_argument("--reserve", type=float, default=0, help="unspent allowance for setup/storage/uncertainty")
    p.add_argument("--max-hours", type=float, help="finish current run, then stop admitting runs at this limit")


def _make_spec(args):
    models = _models(args.models)
    return config.ExperimentSpec(
        models=models, track=args.track, mode=args.mode,
        ranks=utils.parse_int_list(args.ranks) or list(config.DEFAULT_RANKS),
        seeds=utils.parse_int_list(args.seeds) or list(config.DEFAULT_SEEDS),
        lora_lrs=_rates(args.lora_lrs, models), full_lrs=_rates(args.full_lrs, models),
        output_dir=args.output, force=args.force, deterministic=args.deterministic,
        optimizer=args.optimizer, micro_batch=args.micro_batch, accum=args.accum,
        eval_batch=args.eval_batch, save_full_weights=args.save_full_weights,
        num_shards=args.num_shards, shard_index=args.shard_index,
        hourly_rate=args.hourly_rate, budget=args.budget, reserve=args.reserve,
        max_hours=args.max_hours, fail_fast=not args.keep_going,
        head_lr_multiplier=args.head_lr_multiplier,
        calibrated_defaults={} if args.test_lr else None,
    )


def _print_plan(spec):
    from .runner import build_grid, estimate_workload
    from .cost import cost_estimate
    root = Path(spec.output_dir)
    if spec.mode in ("sanity", "smoke"):
        root /= spec.mode
    grid = build_grid(spec)
    print(json.dumps({"workload": estimate_workload(grid),
                      "cost": cost_estimate(grid, root, spec.hourly_rate, spec.force),
                      "results": str(root), "shard": [spec.shard_index, spec.num_shards],
                      "fixed_models": {m: {"hf_id": config.MODELS[m].hf_id,
                                            "revision": config.MODELS[m].revision,
                                            "lora_lrs": spec.resolved_lora_lrs(m),
                                            "full_lrs": spec.resolved_full_lrs(m),
                                            "head_lr_multipliers": {t: spec.resolved_head_multiplier(m, t)
                                                                    for t in ("lora", "full")}}
                                       for m in spec.models},
                      "declared_defaults": spec.default_policy()}, indent=2))
    if spec.mode == "main":
        print("\nUse --mode pilot to measure full-length r=16 and full-FT runs before committing the grid.")
    if spec.budget or spec.max_hours:
        print("Scheduling limits finish complete runs and can overrun by one unmeasured run. "
              "They do not stop or destroy your rented instance.")


def _build_parser():
    p = argparse.ArgumentParser(description="Feasible Rank Window: fixed models, 120 main runs, deferred CPU spectra")
    sub = p.add_subparsers(dest="command")
    for name in ("run", "plan"):
        s = sub.add_parser(name)
        _add_run(s)
        if name == "run":
            s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("prepare", help="prefetch pinned models/tokenized GLUE; CPU supported")
    _add_models(s)
    s = sub.add_parser("intruders", help="resumable spectral analysis; no GPU needed")
    _add_models(s)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--solver", choices=("eigh", "svd"), default="eigh")
    s = sub.add_parser("analyze", help="generate tables, figures and completeness audit on CPU")
    s.add_argument("--output", default="results")
    s.add_argument("--mode", choices=("main", "smoke"), default="main")
    s = sub.add_parser("doctor", help="test imports and real bf16/optimizer CUDA kernels")
    s.add_argument("--cpu", action="store_true", help="validate an analysis/preparation environment")
    s.add_argument("--optimizer", choices=("adamw8bit", "adamw"), default="adamw8bit")
    sub.add_parser("list-models")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--test-lr":
        argv.insert(0, "run")
    parser = _build_parser()
    args = parser.parse_args(argv)
    utils.get_logger("rank_suite")
    try:
        if args.command in ("run", "plan"):
            spec = _make_spec(args)
            if args.test_lr:
                from .calibration import make_plan, run_calibration
                if args.ranks:
                    raise ValueError("--test-lr selects at rank 16 and diagnoses ranks 4/64; --ranks is for main runs")
                seeds = utils.parse_int_list(args.seeds) if args.seeds else (42, 123)
                if args.command == "plan" or args.dry_run:
                    print(json.dumps(make_plan(spec, seeds), indent=2))
                    return 0
                summary = run_calibration(spec, seeds)
                print(json.dumps(summary, indent=2))
                print("Calibration finished. Review its JSON/charts, then run main with the same --output.")
                return 0 if summary["status"] == "complete" else (3 if summary["status"] == "deferred" else 1)
            if args.command == "plan" or args.dry_run:
                _print_plan(spec)
                return 0
            from .runner import run_all
            summary = run_all(spec)
            print(json.dumps(summary, indent=2))
            print("GPU work finished. Run intruders and analyze on CPU; manage your rental in Vast.ai.")
            return 1 if summary["failed"] else (3 if summary["deferred"] else 0)
        if args.command == "prepare":
            from .runner import prepare
            print(prepare(config.ExperimentSpec(models=_models(args.models), output_dir=args.output)))
        elif args.command == "intruders":
            from .intruder import analyze_intruders
            print(analyze_intruders(args.output, _models(args.models), args.threads, args.solver))
        elif args.command == "analyze":
            from .analysis import run_analysis
            print(run_analysis(Path(args.output), mode=args.mode))
        elif args.command == "doctor":
            from .hardware import check_environment
            print(json.dumps(utils.capture_environment(), indent=2))
            print(check_environment(require_gpu=not args.cpu, optimizer=args.optimizer))
        elif args.command == "list-models":
            for key, m in config.MODELS.items():
                print(f"{key:16} {m.hf_id:42} {m.params_total}B  LoRA={m.default_lora_lr:g} "
                      f"LoRA_head={config.DEFAULT_LORA_HEAD_MULTIPLIER[key]:g}x "
                      f"full={m.default_full_lr:g}  {m.license}")
            print("Default ranks:", config.DEFAULT_RANKS, " Seeds:", config.DEFAULT_SEEDS)
        else:
            parser.print_help()
        return 0
    except KeyboardInterrupt:
        print("Interrupted. The next launch resumes from completed task boundaries.")
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
