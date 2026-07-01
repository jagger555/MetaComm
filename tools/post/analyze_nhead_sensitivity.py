from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "n_heads"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "analysis"
DEFAULT_METRICS = [
    "QoI",
    "episodic_aoi",
    "aoi_satis_ratio",
    "data_satis_ratio",
    "mean_episode_reward",
    "test_episode_reward",
    "test_attn_entropy_mean",
    "test_inter_head_diversity",
    "test_obs_rep_effective_rank",
]
LOWER_IS_BETTER = {
    "episodic_aoi",
    "test_episodic_aoi",
    "v_loss",
    "rel_v_loss",
    "kl_divergence",
}
PLOT_COLORS = {
    "QoI": "#0072B2",
    "episodic_aoi": "#D55E00",
    "mean_episode_reward": "#009E73",
    "test_episode_reward": "#CC79A7",
}


def ensure_tensorboard_import() -> None:
    try:
        import tensorboard  # noqa: F401
        return
    except ModuleNotFoundError:
        vendor_dir = WORKSPACE_ROOT / "_vendor"
        if vendor_dir.exists():
            sys.path.insert(0, str(vendor_dir))
        import tensorboard  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate MetaComm CommFormer n_head ablations from run directories, "
            "summarize final/best/tail metrics, and generate publication-ready plots."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Root directory containing head_1/head_2/... experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV summaries and figures.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Scalar tags to summarize.",
    )
    parser.add_argument(
        "--plot-metrics",
        nargs="+",
        default=["QoI", "episodic_aoi"],
        help="Metrics to visualize in the summary figure.",
    )
    parser.add_argument(
        "--summary-stat",
        choices=["final", "best", "tail_mean"],
        default="tail_mean",
        help="Statistic used in the figure and sensitivity ranking.",
    )
    parser.add_argument(
        "--tail-size",
        type=int,
        default=5,
        help="Number of last points used to compute tail_mean.",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="nhead_ablation_summary.png",
        help="Output figure name under --output-dir.",
    )
    return parser.parse_args()


def is_head_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    parts = path.name.split("_")
    if len(parts) != 2 or parts[0] != "head":
        return False
    return parts[1].isdigit()


def parse_head_value(path: Path) -> int:
    return int(path.name.split("_")[-1])


def is_lower_better(metric: str) -> bool:
    return metric in LOWER_IS_BETTER or metric.endswith("_aoi")


def summarize_values(values: list[float], metric: str, tail_size: int) -> dict[str, float]:
    tail_n = min(max(tail_size, 1), len(values))
    tail = values[-tail_n:]
    return {
        "final": float(values[-1]),
        "best": float(min(values) if is_lower_better(metric) else max(values)),
        "tail_mean": float(np.mean(tail)),
        "tail_std_within_run": float(np.std(tail, ddof=0)),
        "num_points": int(len(values)),
    }


def read_scalars_from_long_csv(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    frame = pd.read_csv(csv_path)
    required = {"step", "tag", "value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{csv_path} must contain columns {sorted(required)}.")

    series_map: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        try:
            step = float(row.step)
            value = float(row.value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(step) or not math.isfinite(value):
            continue
        series_map[str(row.tag)].append((step, value))

    for tag in list(series_map.keys()):
        series_map[tag].sort(key=lambda item: item[0])
    return dict(series_map)


def read_scalars_from_event(run_dir: Path) -> dict[str, list[tuple[float, float]]]:
    ensure_tensorboard_import()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        return {}

    event_path = max(event_files, key=lambda path: path.stat().st_size)
    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()

    series_map: dict[str, list[tuple[float, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        points = []
        for event in events:
            step = float(event.step)
            value = float(event.value)
            if math.isfinite(step) and math.isfinite(value):
                points.append((step, value))
        if points:
            series_map[tag] = points
    return series_map


def load_scalar_series(run_dir: Path) -> dict[str, list[tuple[float, float]]]:
    scalars_csv = run_dir / "scalars.csv"
    if scalars_csv.exists():
        return read_scalars_from_long_csv(scalars_csv)
    return read_scalars_from_event(run_dir)


def build_run_summary(
    runs_dir: Path,
    metrics: list[str],
    tail_size: int,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    rows: list[dict[str, object]] = []
    available_by_metric: dict[str, list[int]] = defaultdict(list)

    for head_dir in sorted((path for path in runs_dir.iterdir() if is_head_dir(path)), key=parse_head_value):
        head_value = parse_head_value(head_dir)
        for run_dir in sorted(path for path in head_dir.iterdir() if path.is_dir()):
            scalar_map = load_scalar_series(run_dir)
            if not scalar_map:
                continue

            for metric in metrics:
                if metric not in scalar_map:
                    continue
                points = scalar_map[metric]
                values = [value for _, value in points]
                stats = summarize_values(values, metric, tail_size)
                rows.append(
                    {
                        "head_dir": head_dir.name,
                        "n_head": head_value,
                        "run_name": run_dir.name,
                        "metric": metric,
                        **stats,
                    }
                )
                available_by_metric[metric].append(head_value)

    if not rows:
        raise ValueError(f"No matching run metrics found under {runs_dir}.")

    frame = pd.DataFrame(rows)
    return frame, available_by_metric


def aggregate_run_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict[str, object]] = []
    for (metric, n_head), group in run_summary.groupby(["metric", "n_head"], sort=True):
        row: dict[str, object] = {
            "metric": metric,
            "n_head": int(n_head),
            "num_runs": int(len(group)),
        }
        for stat_name in ("final", "best", "tail_mean"):
            values = group[stat_name].to_numpy(dtype=float)
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            cv_val = float(std_val / abs(mean_val)) if abs(mean_val) > 1e-12 else 0.0
            row[f"{stat_name}_mean"] = mean_val
            row[f"{stat_name}_std"] = std_val
            row[f"{stat_name}_cv"] = cv_val
        grouped_rows.append(row)

    return pd.DataFrame(grouped_rows).sort_values(["metric", "n_head"]).reset_index(drop=True)


def compute_metric_sensitivity(
    aggregated_summary: pd.DataFrame,
    summary_stat: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_col = f"{summary_stat}_mean"
    std_col = f"{summary_stat}_std"

    for metric, group in aggregated_summary.groupby("metric", sort=True):
        group = group.sort_values("n_head")
        values = group[value_col].to_numpy(dtype=float)
        stds = group[std_col].to_numpy(dtype=float)
        heads = group["n_head"].to_numpy(dtype=int)
        if len(values) == 0:
            continue

        if is_lower_better(metric):
            best_idx = int(np.argmin(values))
        else:
            best_idx = int(np.argmax(values))
        most_stable_idx = int(np.argmin(stds))

        range_abs = float(np.max(values) - np.min(values))
        denom = abs(float(np.mean(values)))
        range_rel = float(range_abs / denom) if denom > 1e-12 else 0.0

        rows.append(
            {
                "metric": metric,
                "summary_stat": summary_stat,
                "best_head": int(heads[best_idx]),
                "best_value": float(values[best_idx]),
                "most_stable_head": int(heads[most_stable_idx]),
                "smallest_std": float(stds[most_stable_idx]),
                "head_range_abs": range_abs,
                "head_range_rel": range_rel,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("head_range_rel", ascending=False).reset_index(drop=True)


def plot_summary_figure(
    aggregated_summary: pd.DataFrame,
    metrics: list[str],
    summary_stat: str,
    output_path: Path,
) -> None:
    valid_metrics = [metric for metric in metrics if metric in set(aggregated_summary["metric"])]
    if not valid_metrics:
        return

    value_col = f"{summary_stat}_mean"
    std_col = f"{summary_stat}_std"
    ncols = len(valid_metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.4), dpi=300)
    if ncols == 1:
        axes = [axes]

    for ax, metric in zip(axes, valid_metrics):
        frame = aggregated_summary[aggregated_summary["metric"] == metric].sort_values("n_head")
        x = frame["n_head"].to_numpy(dtype=float)
        y = frame[value_col].to_numpy(dtype=float)
        yerr = frame[std_col].to_numpy(dtype=float)
        color = PLOT_COLORS.get(metric, "#0072B2")

        ax.plot(x, y, marker="o", markersize=6, linewidth=2.0, color=color)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.18)
        ax.set_xticks(x)
        ax.set_xlabel("Number of Attention Heads", fontsize=11)
        ax.set_ylabel(metric.replace("_", " "), fontsize=11)
        ax.set_title(f"{metric} ({summary_stat})", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.28)

        if is_lower_better(metric):
            best_idx = int(np.argmin(y))
        else:
            best_idx = int(np.argmax(y))
        best_x = x[best_idx]
        best_y = y[best_idx]
        ax.scatter([best_x], [best_y], color="#111111", s=24, zorder=3)
        ax.annotate(
            f"best={int(best_x)}",
            xy=(best_x, best_y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#111111",
        )

    fig.suptitle("CommFormer n_head Ablation", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def print_console_summary(
    sensitivity: pd.DataFrame,
    aggregated_summary: pd.DataFrame,
    summary_stat: str,
) -> None:
    print("[analyze_nhead_sensitivity] top metrics by relative head sensitivity")
    if sensitivity.empty:
        print("  no sensitivity summary available")
    else:
        for row in sensitivity.itertuples(index=False):
            print(
                "  "
                f"{row.metric}: best_head={row.best_head}, "
                f"most_stable_head={row.most_stable_head}, "
                f"range_rel={row.head_range_rel:.4f}"
            )

    for metric in ("QoI", "episodic_aoi", "mean_episode_reward", "test_episode_reward"):
        frame = aggregated_summary[aggregated_summary["metric"] == metric].sort_values("n_head")
        if frame.empty:
            continue
        value_col = f"{summary_stat}_mean"
        std_col = f"{summary_stat}_std"
        print(f"[analyze_nhead_sensitivity] {metric} ({summary_stat})")
        for row in frame.itertuples(index=False):
            print(
                "  "
                f"head={row.n_head}: "
                f"{getattr(row, value_col):.4f} +/- {getattr(row, std_col):.4f}"
            )


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()

    run_summary, _ = build_run_summary(
        runs_dir=runs_dir,
        metrics=args.metrics,
        tail_size=args.tail_size,
    )
    aggregated_summary = aggregate_run_summary(run_summary)
    sensitivity = compute_metric_sensitivity(
        aggregated_summary=aggregated_summary,
        summary_stat=args.summary_stat,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = output_dir / "nhead_run_summary.csv"
    aggregated_summary_path = output_dir / "nhead_aggregated_summary.csv"
    sensitivity_path = output_dir / "nhead_metric_sensitivity.csv"
    figure_path = output_dir / args.figure_name

    write_csv(run_summary, run_summary_path)
    write_csv(aggregated_summary, aggregated_summary_path)
    write_csv(sensitivity, sensitivity_path)
    plot_summary_figure(
        aggregated_summary=aggregated_summary,
        metrics=args.plot_metrics,
        summary_stat=args.summary_stat,
        output_path=figure_path,
    )

    print_console_summary(
        sensitivity=sensitivity,
        aggregated_summary=aggregated_summary,
        summary_stat=args.summary_stat,
    )
    print(f"[analyze_nhead_sensitivity] per-run summary saved to {run_summary_path}")
    print(f"[analyze_nhead_sensitivity] aggregated summary saved to {aggregated_summary_path}")
    print(f"[analyze_nhead_sensitivity] sensitivity ranking saved to {sensitivity_path}")
    print(f"[analyze_nhead_sensitivity] figure saved to {figure_path}")


if __name__ == "__main__":
    main()
