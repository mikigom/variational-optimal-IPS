"""Figures 1-11 from <results>/summary.csv."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter

from experiments.uci.common import DATASETS, METHODS

LABELS = ["IPS", "bIPS", "wIPS", "oIPS", "DR-bIPS", "DR-oIPS"]
COLORS = ["#CC3333", "#2379B5", "#8A60B8", "#229641", "#95614F", "#E67E12"]
MARKERS = ["o", "x", "s", "D", "^", "P"]
TITLES = ["Letter", "OptDigits", "PenDigits", "SatImage"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def style(ax, ratio_axis=True):
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    if ratio_axis:
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1, 10], ["$10^{-1}$", "$1$", "$10$"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("$n_1/n_2$")


def ratio_figure(summary, setting, estimated, methods=METHODS):
    """Relative-RMSE against n1/n2 with bootstrap bands (Figures 1-3, 8-11)."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    sub = summary[(summary.setting == setting) & (summary.estimate_pi_b == estimated)].copy()
    sub["ratio"] = sub.ratio_label.astype(float)
    for ax, dataset, title in zip(axes, DATASETS, TITLES):
        for m in methods:
            i = METHODS.index(m)
            rows = sub[(sub.dataset == dataset) & (sub.method == m)].sort_values("ratio")
            ax.plot(rows.ratio, rows.rel_rmse, color=COLORS[i], marker=MARKERS[i], markersize=3, linewidth=1,
                    label=LABELS[i])
            ax.fill_between(rows.ratio, rows.ci_low, rows.ci_high, color=COLORS[i], alpha=0.16, linewidth=0)
        style(ax)
        ax.set_title(title)
    axes[0].set_ylabel("Relative-RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False, columnspacing=1.2,
               handletextpad=0.4, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.08, right=0.985, wspace=0.35)
    return fig


def logger_figure(summary, k, estimated):
    """Relative-RMSE with K > 2 loggers (Figures 4-7)."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5))
    sub = summary[(summary.num_loggers == k) & (summary.estimate_pi_b == estimated)]
    for ax, dataset, title in zip(axes.flat, DATASETS, TITLES):
        for i, m in enumerate(METHODS):
            row = sub[(sub.dataset == dataset) & (sub.method == m)].iloc[0]
            ax.errorbar(i, row.rel_rmse, yerr=[[row.rel_rmse - row.ci_low], [row.ci_high - row.rel_rmse]],
                        fmt=MARKERS[i], color=COLORS[i], markersize=5, capsize=3, linewidth=1.2)
        style(ax, ratio_axis=False)
        ax.set_xticks(range(len(METHODS)), LABELS, rotation=20, ha="right")
        ax.set_xlim(-0.5, len(METHODS) - 0.5)
        ax.set_title(title)
        ax.set_ylabel("Relative-RMSE")
    fig.subplots_adjust(top=0.94, bottom=0.11, left=0.10, right=0.98, hspace=0.60, wspace=0.33)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/uci"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--format", default="pdf")
    args = parser.parse_args()

    summary = pd.read_csv(args.results / "summary.csv", dtype={"ratio_label": str})
    figures = {
        1: ratio_figure(summary, "main_eta_0.95", False),
        2: ratio_figure(summary, "main_eta_0.95", True),
        3: ratio_figure(summary, "main_eta_0.95", True, ["dr_bips", "dr_oips"]),
        4: logger_figure(summary, 5, False),
        5: logger_figure(summary, 5, True),
        6: logger_figure(summary, 3, False),
        7: logger_figure(summary, 3, True),
        8: ratio_figure(summary, "sensitivity_eta_0.80", False),
        9: ratio_figure(summary, "sensitivity_eta_0.80", True),
        10: ratio_figure(summary, "sensitivity_eta_0.65", False),
        11: ratio_figure(summary, "sensitivity_eta_0.65", True),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for number, fig in figures.items():
        fig.savefig(args.out / f"fig{number}.{args.format}", bbox_inches="tight", pad_inches=0.04, dpi=220)
        plt.close(fig)
    print(f"wrote {len(figures)} figures to {args.out}")


if __name__ == "__main__":
    main()
