import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rank_suite import figures
from test_metrics import window_data


def test_dashboard_preserves_intruder_count_units(monkeypatch, tmp_path):
    df = window_data()
    df["intruder_total_final"] = 16
    other = df.copy()
    other["model"] = "qwen2.5-3b"
    other.loc[other.track == "lora", "lr"] = .00015
    other.loc[other.track == "full", "lr"] = .000015
    df = pd.concat([df, other], ignore_index=True)
    monkeypatch.setattr(figures, "save", lambda fig, *args: fig)
    fig = figures.fig_dashboard(plt, df, tmp_path, "qwen2.5-1.5b", "test", [])
    counts = fig.axes[2].lines[0].get_ydata()
    assert all(value == 16 for value in counts)
    assert fig.axes[0].get_legend_handles_labels()[1] == ["lr=1.85e-04"]
    plt.close(fig)
