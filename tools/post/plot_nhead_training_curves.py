from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_style import get_scaled_sizes, add_font_scale_arg


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "n_heads"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "analysis_figures"
DEFAULT_METRICS = ["mean_episode_reward", "episodic_aoi"]
COLOR_MAP = {
    1: "#4E79A7",
    2: "#E15759",
    4: "#76B7B2",
    8: "#59A14F",
    16: "#F28E2B",
}
MARKER_MAP = {
    1: "x",
    2: "o",
    4: "^",
    8: "D",
    16: "s",
}
TITLE_MAP = {
    "mean_episode_reward": "Mean Episode Reward",
    "episodic_aoi": "Episodic AoI",
    "QoI": "QoI",
    "test_episode_reward": "Test Episode Reward",
}
LOWER_IS_BETTER = {"episodic_aoi", "test_episodic_aoi"}
_PLOT_SIZES = get_scaled_sizes(1.0)


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
        description="Plot publication-style n_head ablation training curves from TensorBoard event files."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Root directory containing head_1/head_2/... experiment folders.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Scalar tags to plot. Recommended: mean_episode_reward episodic_aoi.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the output figure.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="nhead_clear_gap_curves.png",
        help="Output figure filename.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=240,
        help="Interpolation grid size for run aggregation.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.18,
        help="EMA smoothing factor applied after interpolation.",
    )
    parser.add_argument(
        "--marker-every",
        type=int,
        default=12,
        help="Show one marker every N interpolated points.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    add_font_scale_arg(parser, default=1.0)
    return parser.parse_args()


def is_head_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("head_") and path.name.split("_")[-1].isdigit()


def parse_head(path: Path) -> int:
    return int(path.name.split("_")[-1])


def load_run_series(run_dir: Path, metric: str) -> tuple[np.ndarray, np.ndarray] | None:
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

    events = accumulator.Scalars(metric)
    y = np.asarray([float(event.value) for event in events if math.isfinite(float(event.value))], dtype=np.float64)
    if y.size == 0:
        return None
    x = np.arange(1, y.size + 1, dtype=np.float64)
    return x, y


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return values.copy()
    out = np.empty_like(values)
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def aggregate_metric_runs(
    runs_dir: Path,
    metric: str,
    num_points: int,
    smooth_alpha: float,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped = {}
    for head_dir in sorted((path for path in runs_dir.iterdir() if is_head_dir(path)), key=parse_head):
        head = parse_head(head_dir)
        runs = []
        for run_dir in sorted(path for path in head_dir.iterdir() if path.is_dir()):
            series = load_run_series(run_dir, metric)
            if series is not None:
                runs.append(series)
        if not runs:
            continue

        min_x = min(run[0][0] for run in runs)
        max_x = min(run[0][-1] for run in runs)
        if max_x <= min_x:
            continue

        grid = np.linspace(min_x, max_x, num_points)
        curves = []
        for x, y in runs:
            interp = np.interp(grid, x, y)
            interp = ema_smooth(interp, smooth_alpha)
            curves.append(interp)
        stacked = np.stack(curves, axis=0)
        grouped[head] = (grid, stacked.mean(axis=0), stacked.std(axis=0, ddof=0))
    return grouped


def choose_anchor_point(y: np.ndarray, lower_is_better: bool) -> int:
    return int(np.argmin(y) if lower_is_better else np.argmax(y))


def plot_metric_panel(
    ax: plt.Axes,
    aggregated: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    metric: str,
    marker_every: int,
) -> None:
    lower_is_better = metric in LOWER_IS_BETTER
    for head, (grid, mean, std) in aggregated.items():
        color = COLOR_MAP.get(head, "#4E79A7")
        marker = MARKER_MAP.get(head, "o")
        label = f"{head} heads"
        linewidth = 2.5 if head == 2 else 2.0
        markersize = 5.2 if head == 2 else 4.8
        zorder = 4 if head == 2 else 3

        ax.plot(
            grid,
            mean,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markevery=marker_every,
            markersize=markersize,
            label=label,
            zorder=zorder,
        )
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)

        if head == 2:
            best_idx = choose_anchor_point(mean, lower_is_better)
            ax.scatter([grid[best_idx]], [mean[best_idx]], color=color, s=36, zorder=5)

    sizes = _PLOT_SIZES
    ax.set_title(TITLE_MAP.get(metric, metric), fontsize=sizes["title"])
    ax.set_xlabel("Episode", fontsize=sizes["label"])
    ax.set_ylabel(TITLE_MAP.get(metric, metric), fontsize=sizes["label"])
    ax.tick_params(labelsize=sizes["tick"])
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best", fontsize=sizes["legend"], frameon=True, framealpha=0.92)


def build_figure(
    runs_dir: Path,
    metrics: list[str],
    output_path: Path,
    num_points: int,
    smooth_alpha: float,
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

    fig, axes = plt.subplots(1, len(metrics), figsize=(6.8 * len(metrics), 4.9), dpi=dpi)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        aggregated = aggregate_metric_runs(
            runs_dir=runs_dir,
            metric=metric,
            num_points=num_points,
            smooth_alpha=smooth_alpha,
        )
        if not aggregated:
            ax.axis("off")
            continue
        plot_metric_panel(
            ax=ax,
            aggregated=aggregated,
            metric=metric,
            marker_every=marker_every,
        )

    fig.suptitle(
        "Impact of Attention Head Number on MetaComm Training Dynamics",
        fontsize=_PLOT_SIZES["suptitle"],
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = args.output_dir.resolve() / args.output_name
    build_figure(
        runs_dir=args.runs_dir.resolve(),
        metrics=args.metrics,
        output_path=output_path,
        num_points=args.num_points,
        smooth_alpha=args.smooth_alpha,
        marker_every=args.marker_every,
        dpi=args.dpi,
        font_scale=args.font_scale,
    )
    print(f"[plot_nhead_training_curves] figure saved to {output_path}")


if __name__ == "__main__":
    main()
