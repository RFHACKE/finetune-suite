"""Job specifications, the experiment grid and the pre-registered LR selection rule."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

from . import config as C

# Rank interleaving so a partially completed stage still spans the whole range.
COVERAGE = (16, 1, 1024, 4, 256, 64, 2, 512, 8, 128, 32)


@dataclass(frozen=True)
class Job:
    kind: str            # tune | main | order | full_tune | full_main
    model: str
    track: str           # lora | full
    rank: Optional[int]
    lr: float
    seed: int
    order: str           # O1/O2/O3
    split: str           # holdout (LR search) | official (reported)
    priority: float = 0.0

    @property
    def spec(self) -> dict:
        d = asdict(self)
        d.pop("priority")
        d["protocol"] = C.PROTOCOL
        return d

    @property
    def job_id(self) -> str:
        r = f"r{self.rank}" if self.rank is not None else "full"
        return (f"{self.model}__{r}__lr{C.fmt_lr(self.lr)}__s{self.seed}__{self.order}__"
                f"{self.split}__{C.stable_hash(self.spec, 8)}")

    @staticmethod
    def from_spec(d: dict, priority: float = 0.0) -> "Job":
        keys = ("kind", "model", "track", "rank", "lr", "seed", "order", "split")
        return Job(**{k: d[k] for k in keys}, priority=priority)


def _rank_pos(rank) -> int:
    return COVERAGE.index(rank) if rank in COVERAGE else len(COVERAGE)


def tune_jobs(model: str, ranks=C.RANKS, lrs=C.LORA_LR_GRID, seeds=C.TUNE_SEEDS) -> list:
    jobs = []
    for si, seed in enumerate(seeds):
        for rank in ranks:
            for lr in lrs:
                jobs.append(Job("tune", model, "lora", rank, lr, seed, "O1", "holdout",
                                priority=100 + si * 20 + _rank_pos(rank)))
    return jobs


def main_jobs(model: str, rank: int, lr: float, seeds=C.MAIN_SEEDS) -> list:
    # First three seeds before the last two so partial completion keeps n=3 everywhere.
    return [Job("main", model, "lora", rank, lr, s, "O1", "official",
                priority=300 + (0 if i < 3 else 30) + _rank_pos(rank))
            for i, s in enumerate(seeds)]


def order_jobs(model: str, rank: int, lr: float) -> list:
    if rank not in C.ORDER_RANKS:
        return []
    return [Job("order", model, "lora", rank, lr, s, o, "official",
                priority=500 + 10 * oi + _rank_pos(rank))
            for oi, o in enumerate(("O2", "O3")) for s in C.ORDER_SEEDS]


def full_tune_jobs(model: str, lrs=C.FULL_LR_GRID, seeds=C.TUNE_SEEDS[:1]) -> list:
    return [Job("full_tune", model, "full", None, lr, s, "O1", "holdout", priority=400 + i)
            for i, (lr, s) in enumerate((lr, s) for s in seeds for lr in lrs)]


def full_main_jobs(model: str, lr: float) -> list:
    return [Job("full_main", model, "full", None, lr, s, "O1", "official", priority=450 + i)
            for i, s in enumerate(C.MAIN_SEEDS)]


# ---------------------------------------------------------------------------
# Pre-registered selection: maximise mean final-row holdout accuracy (ACC) over
# tuning seeds; ties -> lower LR. If the winner sits on an edge of the tested
# grid, one more LR beyond that edge is tested (factor 2, within bounds) before
# the choice is frozen. Failed/diverged runs score 0.
# ---------------------------------------------------------------------------
def acc_of(result: Optional[dict]) -> float:
    if not result or result.get("status") != "completed":
        return 0.0
    A = result["A"]
    final = A[-1]
    return float(sum(final) / len(final))


def select_lr(results_by_lr: dict, n_seeds: int, lo: float, hi: float):
    """results_by_lr: {lr: [result dicts]} -> (status, lr, extra_lr, table).

    status: 'pending' (not all seeds done), 'extend' (test extra_lr first) or 'selected'.
    """
    table = {}
    for lr, rs in results_by_lr.items():
        if len(rs) < n_seeds:
            return "pending", None, None, None
        table[lr] = sum(acc_of(r) for r in rs) / len(rs)
    lrs = sorted(table)
    best = max(lrs, key=lambda x: (round(table[x], 10), -x))
    if best == lrs[0] and best / 2 >= lo * 0.999 and not _has(lrs, best / 2):
        return "extend", None, best / 2, table
    if best == lrs[-1] and best * 2 <= hi * 1.001 and not _has(lrs, best * 2):
        return "extend", None, best * 2, table
    return "selected", best, None, table


def _has(values, x) -> bool:
    return any(math.isclose(v, x, rel_tol=1e-6) for v in values)
