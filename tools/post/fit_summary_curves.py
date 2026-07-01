from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGOS = ["G2ANet", "ConvLSTM", "DPPO", "CPPO", "IC3Net", "MetaComm"]
ALGO_LABELS = {
    "G2ANet": "G2A",
    "ConvLSTM": "ConvLSTM",
    "DPPO": "DPPO",
    "CPPO": "CPPO",
    "IC3Net": "IC3Net",
    "MetaComm": "MetaComm",
}
METRICS = ["QoI", "episodic_aoi", "aoi_satis_ratio", "data_satis_ratio", "energy_consuming"]
UAV_NUMS = [2, 3, 4, 5, 7, 10]

METRIC_LABELS = {
    "QoI": "QoI",
    "episodic_aoi": "Episodic AoI",
    "aoi_satis_ratio": "AoI Satisfaction Ratio",
    "data_satis_ratio": "Data Satisfaction Ratio",
    "energy_consuming": "Energy Consumption (MJ)",
}
METRIC_ORDER = {
    "QoI": 0,
    "episodic_aoi": 1,
    "aoi_satis_ratio": 2,
    "data_satis_ratio": 3,
    "energy_consuming": 4,
}
ALGO_COLORS = {
    "G2ANet": "#4C78A8",
    "ConvLSTM": "#F58518",
    "DPPO": "#54A24B",
    "CPPO": "#E45756",
    "IC3Net": "#72B7B2",
    "MetaComm": "#B279A2",
}
ALGO_MARKERS = {
    "G2ANet": "o",
    "ConvLSTM": "s",
    "DPPO": "^",
    "CPPO": "D",
    "IC3Net": "v",
    "MetaComm": "P",
}
ALGO_ORDER = {algo: idx for idx, algo in enumerate(ALGOS)}


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 2:
        return np.array([(y[1] - y[0]) / (x[1] - x[0]), (y[1] - y[0]) / (x[1] - x[0])], dtype=float)

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=float)

    for k in range(1, n - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            d[k] = 0.0
            continue
        w1 = 2.0 * h[k] + h[k - 1]
        w2 = h[k] + 2.0 * h[k - 1]
        d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d0) != np.sign(delta[0]):
        d0 = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d0) > abs(3.0 * delta[0]):
        d0 = 3.0 * delta[0]
    d[0] = d0

    dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(dn) != np.sign(delta[-1]):
        dn = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(dn) > abs(3.0 * delta[-1]):
        dn = 3.0 * delta[-1]
    d[-1] = dn

    return d


def pchip_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    d = pchip_slopes(x, y)
    y_new = np.empty_like(x_new, dtype=float)

    for idx, xv in enumerate(x_new):
        if xv <= x[0]:
            interval = 0
        elif xv >= x[-1]:
            interval = len(x) - 2
        else:
            interval = np.searchsorted(x, xv) - 1
        h = x[interval + 1] - x[interval]
        t = (xv - x[interval]) / h
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        y_new[idx] = (
            h00 * y[interval]
            + h10 * h * d[interval]
            + h01 * y[interval + 1]
            + h11 * h * d[interval + 1]
        )

    return y_new


def load_summary_points(summary_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(summary_csv, header=None)
    expected_cols = len(ALGOS) * len(METRICS)
    if raw.shape[1] != expected_cols:
        raise ValueError(
            f"Unexpected column count in {summary_csv}. Expected {expected_cols}, got {raw.shape[1]}."
        )
    if raw.shape[0] != len(UAV_NUMS):
        raise ValueError(
            f"Unexpected row count in {summary_csv}. Expected {len(UAV_NUMS)}, got {raw.shape[0]}."
        )

    records = []
    for row_idx, uav_num in enumerate(UAV_NUMS):
        values = raw.iloc[row_idx].to_numpy(dtype=float)
        for algo_idx, algo in enumerate(ALGOS):
            start = algo_idx * len(METRICS)
            for metric_idx, metric in enumerate(METRICS):
                records.append(
                    {
                        "uav_num": uav_num,
                        "algo": algo,
                        "algo_label": ALGO_LABELS[algo],
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "value": float(values[start + metric_idx]),
                    }
                )
    df = pd.DataFrame.from_records(records)
    df["metric_order"] = df["metric"].map(METRIC_ORDER)
    df["algo_order"] = df["algo"].map(ALGO_ORDER)
    return df


def build_fitted_curves(points_df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    dense_x = np.linspace(min(UAV_NUMS), max(UAV_NUMS), num_samples)
    records = []
    for metric in METRICS:
        metric_df = points_df[points_df["metric"] == metric]
        for algo in ALGOS:
            algo_df = metric_df[metric_df["algo"] == algo].sort_values("uav_num")
            x = algo_df["uav_num"].to_numpy(dtype=float)
            y = algo_df["value"].to_numpy(dtype=float)
            fitted = pchip_interpolate(x, y, dense_x)
            for xv, yv in zip(dense_x, fitted):
                records.append(
                    {
                        "uav_num_fit": float(xv),
                        "algo": algo,
                        "algo_label": ALGO_LABELS[algo],
                        "algo_order": ALGO_ORDER[algo],
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "metric_order": METRIC_ORDER[metric],
                        "value_fit": float(yv),
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_fitted_curves(points_df: pd.DataFrame, fitted_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        metric_points = points_df[points_df["metric"] == metric]
        metric_fits = fitted_df[fitted_df["metric"] == metric]

        for algo in ALGOS:
            color = ALGO_COLORS[algo]
            marker = ALGO_MARKERS[algo]
            algo_points = metric_points[metric_points["algo"] == algo].sort_values("uav_num")
            algo_fits = metric_fits[metric_fits["algo"] == algo].sort_values("uav_num_fit")

            ax.plot(
                algo_fits["uav_num_fit"],
                algo_fits["value_fit"],
                color=color,
                linewidth=2.2,
                label=ALGO_LABELS[algo],
            )
            ax.scatter(
                algo_points["uav_num"],
                algo_points["value"],
                color=color,
                marker=marker,
                s=48,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )

        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.set_xlabel("Number of UAVs", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xticks(UAV_NUMS)
        ax.tick_params(labelsize=9)
        if metric != "energy_consuming":
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        else:
            ax.set_ylabel("Energy", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, loc="center", ncol=2, frameon=False, fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse five_ALL_uav_num.csv into aligned point data and fitted curves so "
            "TensorBoard-based summaries can match the optimized summary CSV."
        )
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        required=True,
        help="Path to five_ALL_uav_num.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for the aligned CSV files and fitted plot. Defaults to the summary CSV directory.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=300,
        help="Number of dense x-axis samples used for the fitted curves.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.summary_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    points_df = load_summary_points(args.summary_csv)
    fitted_df = build_fitted_curves(points_df, args.num_samples)

    points_path = output_dir / "tensorboard_means_aligned_uavnum.csv"
    fitted_path = output_dir / "tensorboard_means_aligned_uavnum_fitted.csv"
    figure_path = output_dir / "tensorboard_means_aligned_uavnum_fitted.png"

    points_df.sort_values(["metric_order", "algo_order", "uav_num"]).drop(
        columns=["metric_order", "algo_order"]
    ).to_csv(points_path, index=False)
    fitted_df.sort_values(["metric_order", "algo_order", "uav_num_fit"]).drop(
        columns=["metric_order", "algo_order"]
    ).to_csv(fitted_path, index=False)
    plot_fitted_curves(points_df, fitted_df, figure_path)

    print(f"[fit_summary_curves] aligned points saved to {points_path}")
    print(f"[fit_summary_curves] fitted curves saved to {fitted_path}")
    print(f"[fit_summary_curves] figure saved to {figure_path}")


if __name__ == "__main__":
    main()
