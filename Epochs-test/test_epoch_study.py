"""Scientific preflight checks; fixtures stay in pytest temporary directories."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epoch_study import expected_grid, observation_rng, protocol  # noqa: E402
from build_reports import analyze, verdict  # noqa: E402
from rank_suite import utils  # noqa: E402
from rank_suite.protocol import digest  # noqa: E402


def test_exactly_twelve_continuous_runs():
    grid = expected_grid("sst2")
    assert len(grid) == 12
    assert len({r["run_id"] for r in grid}) == 12
    assert all(r["training_epochs"] == 6 and r["rank"] == 16 for r in grid)
    assert protocol("sst2")["expected_epoch_observations"] == 72
    with pytest.raises(ValueError):
        protocol("full")


def test_observations_do_not_change_training(tmp_path, monkeypatch):
    from rank_suite import train

    monkeypatch.setattr(train, "forward_logits", lambda model, head, batch: head(model(batch["x"])))
    outputs = []
    metrics = []
    for observe in (False, True):
        utils.seed_all(42)
        model, head = torch.nn.Linear(4, 3), torch.nn.Linear(3, 2)
        data = [
            {"x": torch.arange(4, dtype=torch.float32) / (i + 1), "labels": torch.tensor(i % 2)}
            for i in range(16)
        ]
        generator = torch.Generator().manual_seed(19)
        loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True, generator=generator)

        def callback(epoch, step, loss):
            with observation_rng("cpu"):
                unshuffled = torch.utils.data.DataLoader(data, batch_size=4)
                metrics.append(train.evaluate(model, head, unshuffled, "cpu", return_details=True))
                torch.rand(7)

        train._train_task(
            model,
            head,
            list(model.parameters()) + list(head.parameters()),
            loader,
            6,
            1e-3,
            "adamw",
            "cpu",
            tmp_path / f"steps-{observe}.jsonl",
            0,
            "sst2",
            epoch_callback=callback if observe else None,
        )
        outputs.append([p.detach().clone() for p in list(model.parameters()) + list(head.parameters())])
    assert len(metrics) == 6
    assert all(torch.equal(a, b) for a, b in zip(*outputs))


def fixture_cohort(root, best_epoch):
    control = protocol("sst2")
    utils.save_json(root / "protocol.json", control)
    grid = expected_grid("sst2")
    utils.save_json(root / "expected_grid.json", grid)
    for identity in grid:
        run = root / "models" / identity["model"] / "runs" / identity["run_id"]
        attempt = run / "attempts" / "fixture"
        attempt.mkdir(parents=True)
        observations = []
        for epoch in range(1, 7):
            row = {**identity, "epoch": epoch, "task": "sst2", "online_train_loss": 0.4, "step": 156 * epoch}
            for split, count in (("train", 2490), ("validation", 872)):
                labels = np.arange(count, dtype=np.int64) % 2
                probability = (
                    0.7 + 0.04 * epoch
                    if split == "train"
                    else 0.7 + 0.04 * min(epoch, best_epoch) - 0.08 * max(0, epoch - best_epoch)
                )
                true_prob = np.full(count, probability, dtype=np.float32)
                true_prob[::10] = 0.25
                prob = np.empty((count, 2), dtype=np.float32)
                prob[np.arange(count), labels] = true_prob
                prob[np.arange(count), 1 - labels] = 1 - true_prob
                nll = -np.log(true_prob)
                pred = prob.argmax(axis=1)
                path = attempt / f"epoch{epoch}_{split}.npz"
                np.savez_compressed(
                    path, labels=labels, predictions=pred, probabilities=prob, example_nll=nll
                )
                row[split] = {
                    "count": count,
                    "nll": float(nll.mean()),
                    "accuracy": float((labels == pred).mean()),
                    "predictions": str(path.relative_to(run)),
                }
            row["nll_gap"] = row["validation"]["nll"] - row["train"]["nll"]
            row["accuracy_gap"] = row["train"]["accuracy"] - row["validation"]["accuracy"]
            observations.append(row)
        (attempt / "epoch_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in observations))
        steps = [{"step": s, "epoch": (s - 1) // 156 + 1, "lr": 1e-4} for s in range(1, 937)]
        (attempt / "train_steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in steps))
        utils.save_json(
            run / "result.json",
            {
                **identity,
                "status": "completed",
                "protocol_sha256": digest(control),
                "attempt": "attempts/fixture",
                "epoch_metrics": "attempts/fixture/epoch_metrics.jsonl",
                "elapsed_s": 1.0,
                "peak_vram_gb": 1.0,
            },
        )
    return grid


@pytest.mark.parametrize("best_epoch", [3, 4])
def test_conclusions_follow_data_not_desired_hypothesis(tmp_path, best_epoch):
    fixture_cohort(tmp_path, best_epoch)
    summary, rows, times, _ = analyze(tmp_path)
    assert summary["complete"]
    assert len(rows) == 72 and len(times) == 12
    assert all(m["best_observed_nll_epoch"] == best_epoch for m in summary["models"].values())
    assert summary["epoch3_hypothesis_supported_descriptively"] is (best_epoch == 3)
    if best_epoch == 4:
        assert "not supported" in verdict(summary)


def test_missing_epoch_blocks_conclusion(tmp_path):
    grid = fixture_cohort(tmp_path, 3)
    first = grid[0]
    path = (
        tmp_path
        / "models"
        / first["model"]
        / "runs"
        / first["run_id"]
        / "attempts/fixture/epoch_metrics.jsonl"
    )
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    summary, *_ = analyze(tmp_path)
    assert not summary["complete"]
    assert summary["epoch3_hypothesis_supported_descriptively"] is None
    assert summary["rejected"][0]["run_id"] == first["run_id"]


def test_generated_documents_open_and_preserve_contrary_findings(tmp_path):
    from build_reports import build
    from docx import Document
    from pptx import Presentation

    fixture_cohort(tmp_path, 4)
    output = build(tmp_path)
    assert output["complete"] and output["epoch_observations"] == 72
    document = Document(tmp_path / "reports/epoch-study.docx")
    body = "\n".join(p.text for p in document.paragraphs)
    assert "not supported" in body
    assert "not 72 independent replications" in body
    slides = Presentation(tmp_path / "reports/epoch-study.pptx")
    assert len(slides.slides) == 16
    assert len(document.inline_shapes) == 8
