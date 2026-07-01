from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw the reward-preference trade-off figure from an existing summary CSV."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Path to reward_pref_summary.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path. A PDF with the same stem will also be written.",
    )
    parser.add_argument(
        "--left-column",
        type=str,
        default="best_test_data_satis_ratio_mean",
        help="CSV column used for the left y-axis.",
    )
    parser.add_argument(
        "--right-column",
        type=str,
        default="best_test_aoi_satis_ratio_mean",
        help="CSV column used for the right y-axis.",
    )
    parser.add_argument(
        "--left-label",
        type=str,
        default="Throughput satisfaction ratio",
        help="Left y-axis label.",
    )
    parser.add_argument(
        "--right-label",
        type=str,
        default="AoI satisfaction ratio",
        help="Right y-axis label.",
    )
    parser.add_argument(
        "--legend-x",
        type=float,
        default=0.50,
        help="Legend x anchor in axes coordinates.",
    )
    parser.add_argument(
        "--legend-y",
        type=float,
        default=0.97,
        help="Legend y anchor in axes coordinates.",
    )
    return parser.parse_args()


def to_float(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("锛?", ".", regex=False)
        .str.replace("�", ".", regex=False)
        .str.replace("．", ".", regex=False)
        .str.replace("。", ".", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned = cleaned.str.extract(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce")


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv, dtype=str)

    required_columns = {"reward_pref", args.left_column, args.right_column}
    missing = sorted(required_columns.difference(summary.columns))
    if missing:
        raise KeyError(f"Missing required columns in {args.summary_csv}: {missing}")

    summary["reward_pref"] = to_float(summary["reward_pref"])
    summary[args.left_column] = to_float(summary[args.left_column])
    summary[args.right_column] = to_float(summary[args.right_column])
    summary = summary.dropna(subset=["reward_pref", args.left_column, args.right_column])
    summary = summary.sort_values("reward_pref")
    if summary.empty:
        raise ValueError(f"No valid rows found in {args.summary_csv}")

    x = summary["reward_pref"].to_numpy(dtype=float)
    left = summary[args.left_column].to_numpy(dtype=float)
    right = summary[args.right_column].to_numpy(dtype=float)

    fig, ax_left = plt.subplots(figsize=(6.8, 4.8))
    ax_right = ax_left.twinx()

    left_color = "#1f4fff"
    right_color = "#ff2b2b"

    left_line = ax_left.plot(
        x,
        left,
        color=left_color,
        marker="o",
        markersize=6.2,
        markerfacecolor="none",
        markeredgewidth=1.5,
        linewidth=1.8,
        label=args.left_label,
    )
    right_line = ax_right.plot(
        x,
        right,
        color=right_color,
        marker="x",
        markersize=6.0,
        linewidth=1.6,
        label=args.right_label,
    )

    ax_left.set_xlabel(r"Reward preference $\lambda$", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax_left.set_ylabel(args.left_label, fontsize=AXIS_LABEL_SIZE, color=left_color, labelpad=4)
    ax_right.set_ylabel(args.right_label, fontsize=AXIS_LABEL_SIZE, color=right_color, labelpad=4)
    ax_left.set_xticks(x)
    ax_left.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, direction="in", top=True)
    ax_left.tick_params(axis="y", colors=left_color, labelsize=TICK_LABEL_SIZE, direction="in")
    ax_right.tick_params(axis="y", colors=right_color, labelsize=TICK_LABEL_SIZE, direction="in")
    ax_left.spines["left"].set_color(left_color)
    ax_right.spines["right"].set_color(right_color)
    ax_left.grid(True, color="#b0b0b0", linewidth=0.8, alpha=0.6)
    ax_left.set_axisbelow(True)

    handles = left_line + right_line
    labels = [handle.get_label() for handle in handles]
    ax_left.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(args.legend_x, args.legend_y),
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        facecolor="white",
        framealpha=0.95,
        fontsize=10,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=260, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(args.output)


if __name__ == "__main__":
    main()
