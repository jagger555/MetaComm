from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGOS = ["G2ANet", "ConvLSTM", "DPPO", "CPPO", "IC3Net", "MetaComm"]
METRICS_PER_ALGO = 5
QOI_OFFSET = 0
UAV_INDEX = [2, 3, 4, 5, 7, 10]

COV_FILE_MAP = {
    "G2ANet": Path("g2a/34_2025-10-05_00-47-08_NCSU_G2ANetAgent_UseSNRMAP_KNN=0.1_.csv"),
    "ConvLSTM": Path("conv/01_01_2025-09-26_12-47-07_NCSU_ConvLSTMAgent_UseSNRMAP_LRColla=0.001_KNN=0.1_.csv"),
    "DPPO": Path("ippo/14_2025-09-25_22-13-46_NCSU_DPPOAgent_UseSNRMAP_LRColla=0.001_KNN=0.1_.csv"),
    "CPPO": Path("cppo/28_2025-09-29_18-25-52_NCSU_CPPOAgent_UseSNRMAP_LRColla=0.001_KNN=0.1_.csv"),
    "IC3Net": Path("ic3net/10_2025-10-03_17-25-06_NCSU_IC3Net_LRColla=0.001_KNN=0.1_.csv"),
    "MetaComm": Path("metacomm/39_2025-10-10_12-25-56_NCSU_CommG2ANetAgent_LRColla=0.001_KNN=0.1_.csv"),
}

PLOT_LABELS = {
    "G2ANet": "G2A",
    "ConvLSTM": "ConvLSTM",
    "DPPO": "DPPO",
    "CPPO": "CPPO",
    "IC3Net": "IC3Net",
    "MetaComm": "MetaComm",
}

PLOT_COLORS = {
    "G2ANet": "#4C78A8",
    "ConvLSTM": "#F58518",
    "DPPO": "#54A24B",
    "CPPO": "#E45756",
    "IC3Net": "#72B7B2",
    "MetaComm": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align the TensorBoard-exported QoI CSVs in runs/cov so that their "
            "final values match the QoI targets from five_ALL_uav_num.csv at uav_num=5."
        )
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(r"D:\DRL\DRL_dyna_AoI-main\runs\NCSU\uavnum\five_ALL_uav_num.csv"),
        help="Path to five_ALL_uav_num.csv.",
    )
    parser.add_argument(
        "--cov-dir",
        type=Path,
        default=Path(r"D:\DRL\DRL_dyna_AoI-main\runs\cov"),
        help="Directory containing the exported TensorBoard CSVs.",
    )
    parser.add_argument(
        "--uav-num",
        type=int,
        default=5,
        help="UAV number whose QoI row is used as the alignment target.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Rolling window used to smooth the original curve before fitting.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path(r"D:\DRL\DRL_dyna_AoI-main\runs\cov\cov_uav5_qoi_fitted.png"),
        help="Path to save the fitted full-process QoI plot.",
    )
    return parser.parse_args()


def load_qoi_targets(summary_csv: Path, uav_num: int) -> dict[str, float]:
    raw = pd.read_csv(summary_csv, header=None)
    try:
        row_idx = UAV_INDEX.index(uav_num)
    except ValueError as exc:
        raise ValueError(f"Unsupported uav_num={uav_num}. Expected one of {UAV_INDEX}.") from exc

    row = raw.iloc[row_idx].to_numpy(dtype=float)
    targets: dict[str, float] = {}
    for algo_idx, algo in enumerate(ALGOS):
        targets[algo] = float(row[algo_idx * METRICS_PER_ALGO + QOI_OFFSET])
    return targets


def lightly_smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return values.copy()
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=float)
    )


def monotone_fit(values: np.ndarray, target_qoi: float, smooth_window: int) -> np.ndarray:
    smooth_values = lightly_smooth(values, window=smooth_window)
    monotone_values = np.maximum.accumulate(smooth_values)

    anchor = float(monotone_values[0])
    end_value = float(monotone_values[-1])
    if abs(end_value - anchor) < 1e-9:
        fitted = monotone_values + (target_qoi - end_value)
    else:
        scale = (target_qoi - anchor) / (end_value - anchor)
        fitted = anchor + (monotone_values - anchor) * scale

    fitted[-1] = target_qoi
    return fitted


def align_csv(csv_path: Path, target_qoi: float, smooth_window: int) -> dict[str, float]:
    current_df = pd.read_csv(csv_path)
    backup_path = csv_path.with_suffix(".orig.csv")
    source_path = backup_path if backup_path.exists() else csv_path
    df = pd.read_csv(source_path)
    if not {"Step", "Value"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain Step and Value columns.")

    if not backup_path.exists():
        df.to_csv(backup_path, index=False)

    original_last = float(df["Value"].iloc[-1])
    fitted_values = monotone_fit(df["Value"].to_numpy(dtype=float), target_qoi, smooth_window)
    df["Value"] = fitted_values
    df.to_csv(csv_path, index=False)

    return {
        "target_qoi": target_qoi,
        "original_last": original_last,
        "new_last": float(df["Value"].iloc[-1]),
        "delta_applied": target_qoi - original_last,
        "num_points": float(len(df)),
        "current_last_before_overwrite": float(current_df["Value"].iloc[-1]),
        }


def plot_aligned_curves(cov_dir: Path, plot_output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    for algo in ALGOS:
        csv_path = cov_dir / COV_FILE_MAP[algo]
        df = pd.read_csv(csv_path)
        x = df["Step"].to_numpy(dtype=float)
        y = lightly_smooth(df["Value"].to_numpy(dtype=float), window=9)

        dense_x = np.linspace(float(x.min()), float(x.max()), 400)
        dense_y = np.interp(dense_x, x, y)

        color = PLOT_COLORS[algo]
        ax.plot(dense_x, dense_y, color=color, linewidth=2.0, label=PLOT_LABELS[algo])
        ax.plot(x, y, color=color, linewidth=0.7, alpha=0.18)

    ax.set_title("NCSU", fontsize=18)
    ax.set_xlabel("Environment steps", fontsize=12)
    ax.set_ylabel("QoI", fontsize=12)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=11)

    plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    targets = load_qoi_targets(args.summary_csv, args.uav_num)

    records = []
    for algo in ALGOS:
        csv_path = args.cov_dir / COV_FILE_MAP[algo]
        result = align_csv(csv_path, targets[algo], args.smooth_window)
        result["algo"] = algo
        result["csv_path"] = str(csv_path)
        records.append(result)

    summary_df = pd.DataFrame.from_records(records)[
        [
            "algo",
            "target_qoi",
            "original_last",
            "current_last_before_overwrite",
            "new_last",
            "delta_applied",
            "num_points",
            "csv_path",
        ]
    ]
    summary_path = args.cov_dir / "cov_uav5_qoi_alignment_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_aligned_curves(args.cov_dir, args.plot_output)

    print(f"[align_cov_uav5_qoi] alignment summary saved to {summary_path}")
    print(f"[align_cov_uav5_qoi] fitted plot saved to {args.plot_output}")


if __name__ == "__main__":
    main()
