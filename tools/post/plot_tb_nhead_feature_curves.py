from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import get_scaled_sizes, add_font_scale_arg


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "head2"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "analysis_feature_curves"
DEFAULT_METRICS = [
    "test_attn_entropy_mean",
    "test_inter_head_diversity",
    "test_obs_rep_effective_rank",
]
TITLE_MAP = {
    "test_attn_entropy_mean": "Attention Entropy",
    "test_inter_head_diversity": "Inter-Head Diversity",
    "test_obs_rep_effective_rank": "Observation Effective Rank",
}
Y_LABEL_MAP = {
    "test_attn_entropy_mean": "Entropy",
    "test_inter_head_diversity": "Diversity",
    "test_obs_rep_effective_rank": "Effective Rank",
}
_PLOT_SIZES = get_scaled_sizes(1.0)
COLOR_MAP = {
    1: "#4E79A7",
    2: "#E15759",
    4: "#76B7B2",
    8: "#59A14F",
    16: "#F28E2B",
    32: "#B07AA1",
}
MARKER_MAP = {
    1: "x",
    2: "o",
    4: "^",
    8: "D",
    16: "s",
    32: "P",
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
            "Read TensorBoard event files from flat n_head run directories, "
            "rank newly added feature-stability metrics by head-count separation, "
            "and generate publication-ready curves."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing run folders with params.json and TensorBoard event files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Candidate TensorBoard scalar tags to analyze.",
    )
    parser.add_argument(
        "--select-top-k",
        type=int,
        default=2,
        help="Number of top-separated metrics to plot.",
    )
    parser.add_argument(
        "--tail-size",
        type=int,
        default=20,
        help="Number of final points used to summarize head-count separation.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.18,
        help="EMA smoothing factor applied to each run before plotting.",
    )
    parser.add_argument(
        "--marker-every",
        type=int,
        default=48,
        help="Show one marker every N evaluation rounds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures and summary CSV files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="tb_nhead_feature_curves.png",
        help="Output figure filename.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    add_font_scale_arg(parser, default=1.0)
    return parser.parse_args()


def iter_run_dirs(runs_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "params.json").exists()
    )


def load_n_head(run_dir: Path) -> int:
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    return int(params["input_args"]["n_head"])


def load_metric_series(run_dir: Path, metric: str) -> tuple[np.ndarray, np.ndarray] | None:
    ensure_tensorboard_import()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        return None

    event_path = max(event_files, key=lambda path: path.stat().st_size)
    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    if metric not in tags:
        return None

    values = []
    for event in accumulator.Scalars(metric):
        value = float(event.value)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None

    x = np.arange(1, len(values) + 1, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return x, y


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0 or len(values) == 0:
        return values.copy()
    out = np.empty_like(values)
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def aggregate_metric(
    runs_dir: Path,
    metric: str,
    smooth_alpha: float,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[int, list[np.ndarray]] = {}
    for run_dir in iter_run_dirs(runs_dir):
        n_head = load_n_head(run_dir)
        series = load_metric_series(run_dir, metric)
        if series is None:
            continue
        _, y = series
        grouped.setdefault(n_head, []).append(ema_smooth(y, smooth_alpha))

    aggregated = {}
    for n_head, curves in grouped.items():
        min_len = min(len(curve) for curve in curves)
        if min_len == 0:
            continue
        grid = np.arange(1, min_len + 1, dtype=np.float64)
        clipped = np.stack([curve[:min_len] for curve in curves], axis=0)
        aggregated[n_head] = (
            grid,
            clipped.mean(axis=0),
            clipped.std(axis=0, ddof=0),
        )
    return dict(sorted(aggregated.items()))


def build_metric_ranking(
    aggregated_by_metric: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    tail_size: int,
) -> pd.DataFrame:
    rows = []
    for metric, aggregated in aggregated_by_metric.items():
        if len(aggregated) < 2:
            continue
        head_tail_means = []
        for n_head, (_, mean, _) in aggregated.items():
            tail_n = min(max(tail_size, 1), len(mean))
            tail_val = float(np.mean(mean[-tail_n:]))
            head_tail_means.append((n_head, tail_val))

        ordered = sorted(head_tail_means)
        tail_values = np.asarray([value for _, value in ordered], dtype=np.float64)
        adjacent_gaps = np.abs(np.diff(tail_values))
        rows.append(
            {
                "metric": metric,
                "num_heads": int(len(ordered)),
                "tail_range": float(tail_values.max() - tail_values.min()),
                "mean_adjacent_gap": float(adjacent_gaps.mean()) if adjacent_gaps.size else 0.0,
                "max_adjacent_gap": float(adjacent_gaps.max()) if adjacent_gaps.size else 0.0,
                "tail_head_values": "; ".join(f"{n_head}:{value:.6f}" for n_head, value in ordered),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["tail_range", "mean_adjacent_gap", "max_adjacent_gap"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_tail_summary(
    aggregated_by_metric: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    tail_size: int,
) -> pd.DataFrame:
    rows = []
    for metric, aggregated in aggregated_by_metric.items():
        for n_head, (_, mean, std) in aggregated.items():
            tail_n = min(max(tail_size, 1), len(mean))
            rows.append(
                {
                    "metric": metric,
                    "n_head": int(n_head),
                    "final_value": float(mean[-1]),
                    "tail_mean": float(np.mean(mean[-tail_n:])),
                    "tail_std_within_curve": float(np.std(mean[-tail_n:], ddof=0)),
                    "avg_curve_std": float(np.mean(std)),
                    "num_points": int(len(mean)),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "n_head"]).reset_index(drop=True)


def plot_metric_panel(
    ax: plt.Axes,
    metric: str,
    aggregated: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    marker_every: int,
) -> None:
    for n_head, (grid, mean, std) in aggregated.items():
        color = COLOR_MAP.get(n_head, "#4E79A7")
        marker = MARKER_MAP.get(n_head, "o")
        ax.plot(
            grid,
            mean,
            color=color,
            linewidth=2.2,
            marker=marker,
            markevery=max(marker_every, 1),
            markersize=4.8,
            label=f"{n_head} heads",
            zorder=3,
        )
        if np.any(std > 0.0):
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.12, linewidth=0, zorder=2)

    ax.set_title(TITLE_MAP.get(metric, metric), fontsize=_PLOT_SIZES["title"])
    ax.set_xlabel("Evaluation Round", fontsize=_PLOT_SIZES["label"])
    ax.set_ylabel(Y_LABEL_MAP.get(metric, metric), fontsize=_PLOT_SIZES["label"])
    ax.tick_params(labelsize=_PLOT_SIZES["tick"])
    ax.grid(True, linestyle="--", alpha=0.24)
    ax.legend(loc="best", fontsize=_PLOT_SIZES["legend"], frameon=True, framealpha=0.92)


def build_figure(
    selected_metrics: list[str],
    aggregated_by_metric: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    output_path: Path,
    marker_every: int,
    dpi: int,
    font_scale: float = 1.0,
) -> None:
    global _PLOT_SIZES
    _PLOT_SIZES = get_scaled_sizes(font_scale)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    fig, axes = plt.subplots(1, len(selected_metrics), figsize=(6.8 * len(selected_metrics), 4.9), dpi=dpi)
    if len(selected_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, selected_metrics):
        aggregated = aggregated_by_metric.get(metric, {})
        if not aggregated:
            ax.axis("off")
            continue
        plot_metric_panel(ax=ax, metric=metric, aggregated=aggregated, marker_every=marker_every)

    for idx, ax in enumerate(axes):
        ax.text(0.5, -0.24, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=_PLOT_SIZES["suptitle"])

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_tail_figure(
    selected_metrics: list[str],
    tail_summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
    font_scale: float = 1.0,
) -> None:
    sizes = get_scaled_sizes(font_scale)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    fig, axes = plt.subplots(1, len(selected_metrics), figsize=(5.8 * len(selected_metrics), 4.7), dpi=dpi)
    if len(selected_metrics) == 1:
        axes = [axes]

    for idx, (ax, metric) in enumerate(zip(axes, selected_metrics)):
        sub = tail_summary[tail_summary["metric"] == metric].sort_values("n_head")
        if sub.empty:
            ax.axis("off")
            continue

        xs = sub["n_head"].to_numpy(dtype=float)
        ys = sub["tail_mean"].to_numpy(dtype=float)
        for n_head, x, y in zip(sub["n_head"], xs, ys):
            color = COLOR_MAP.get(int(n_head), "#4E79A7")
            marker = MARKER_MAP.get(int(n_head), "o")
            ax.plot([x], [y], color=color, marker=marker, markersize=9, linestyle="None", zorder=4)

        ax.plot(xs, ys, color="#3B3B3B", linewidth=1.8, alpha=0.85, zorder=3)
        ax.set_xticks(xs)
        ax.set_xlabel("Number of Attention Heads", fontsize=sizes["label"])
        ax.set_ylabel(f"Tail Mean {Y_LABEL_MAP.get(metric, metric)}", fontsize=sizes["label"])
        ax.set_title(f"{TITLE_MAP.get(metric, metric)} (Tail Mean)", fontsize=sizes["title"])
        ax.grid(True, linestyle="--", alpha=0.24)
        ax.tick_params(labelsize=sizes["tick"])
        ax.text(0.5, -0.24, f"({chr(97 + idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=sizes["suptitle"])

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    aggregated_by_metric = {
        metric: aggregate_metric(args.runs_dir.resolve(), metric, args.smooth_alpha) for metric in args.metrics
    }
    aggregated_by_metric = {metric: value for metric, value in aggregated_by_metric.items() if value}
    if not aggregated_by_metric:
        raise ValueError(f"No TensorBoard metrics found under {args.runs_dir.resolve()}")

    ranking = build_metric_ranking(aggregated_by_metric, tail_size=args.tail_size)
    if ranking.empty:
        raise ValueError("Unable to rank metrics because fewer than two head-count curves were available.")

    selected_metrics = ranking["metric"].head(max(args.select_top_k, 1)).tolist()
    tail_summary = build_tail_summary(aggregated_by_metric, tail_size=args.tail_size)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "tb_nhead_metric_ranking.csv", index=False)
    tail_summary.to_csv(output_dir / "tb_nhead_tail_summary.csv", index=False)

    build_figure(
        selected_metrics=selected_metrics,
        aggregated_by_metric=aggregated_by_metric,
        output_path=output_dir / args.output_name,
        marker_every=args.marker_every,
        dpi=args.dpi,
        font_scale=args.font_scale,
    )
    build_tail_figure(
        selected_metrics=selected_metrics,
        tail_summary=tail_summary,
        output_path=output_dir / "tb_nhead_feature_tail_vs_heads.png",
        dpi=args.dpi,
        font_scale=args.font_scale,
    )

    print(f"[plot_tb_nhead_feature_curves] selected metrics: {selected_metrics}")
    print(f"[plot_tb_nhead_feature_curves] ranking saved to {output_dir / 'tb_nhead_metric_ranking.csv'}")
    print(f"[plot_tb_nhead_feature_curves] tail summary saved to {output_dir / 'tb_nhead_tail_summary.csv'}")
    print(f"[plot_tb_nhead_feature_curves] figure saved to {output_dir / args.output_name}")
    print(f"[plot_tb_nhead_feature_curves] tail figure saved to {output_dir / 'tb_nhead_feature_tail_vs_heads.png'}")


if __name__ == "__main__":
    main()
