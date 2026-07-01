from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def discover_repo_root() -> Path:
    candidates: list[Path] = []
    file_path = Path(__file__).resolve()
    candidates.extend(file_path.parents)
    cwd = Path.cwd().resolve()
    candidates.extend(cwd.parents)
    candidates.append(cwd)
    candidates.append(Path(r"D:\DRL\DRL_dyna_AoI-main"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "runs").exists() and (candidate / "source_code").exists():
            return candidate
    return cwd


REPO_ROOT = discover_repo_root()
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "newdata1"
METRIC_CHOICES = ("QoI", "episodic_aoi")
SOURCE_CHOICES = ("comm", "tensorboard", "csv")
NORMALIZE_CHOICES = ("comm", "none", "global_minmax", "series_minmax")
BAND_STAT_CHOICES = ("std", "var")
PLOT_STYLE_CHOICES = ("band", "raw_smooth")
THEME_CHOICES = ("comm", "light", "dark")
STYLE_PRESET_CHOICES = ("default", "qoi_reference_raw")

COLOR_MAP = {
    1: "#1f77b4",
    2: "#ff7f0e",
    4: "#2ca02c",
    8: "#d62728",
    16: "#9467bd",
    32: "#8c564b",
}
TITLE_MAP = {
    "QoI": "QoI",
    "episodic_aoi": "Episodic AoI",
}
LOWER_IS_BETTER = {"episodic_aoi", "test_episodic_aoi"}
DEFAULT_FOCUS_CONFIG = {
    "QoI": {
        "ignore_left_ratio": 0.10,
        "lower_quantile": 0.03,
        "upper_quantile": 0.97,
        "pad_ratio": 0.16,
    },
    "episodic_aoi": {
        "ignore_left_ratio": 0.10,
        "lower_quantile": 0.04,
        "upper_quantile": 0.96,
        "pad_ratio": 0.18,
    },
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
        description="Plot attention-head ablation curves from TensorBoard runs or exported CSV files."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Root directory containing head-specific subdirectories.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="comm",
        choices=SOURCE_CHOICES,
        help="Input source type. Auto prefers TensorBoard runs when present, otherwise direct CSV files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=METRIC_CHOICES,
        help="TensorBoard scalar tag to visualize. Optional when plotting direct CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to runs-dir/figures/<metric-or-source>_head_curves.png.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="NCSU",
        help="Figure title.",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="Optional custom y-axis label.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.8,
        help="EMA smoothing factor in [0, 1).",
    )
    parser.add_argument(
        "--post-smooth-window",
        type=int,
        default=9,
        help="Centered rolling-average window applied after EMA. Use odd numbers; set 1 to disable.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=800,
        help="Interpolation grid size used when aggregating multiple runs per head.",
    )
    parser.add_argument(
        "--band-window",
        type=int,
        default=25,
        help="Centered rolling window for local variance when only one run is available.",
    )
    parser.add_argument(
        "--band-alpha",
        type=float,
        default=0.16,
        help="Transparency of variance shadow.",
    )
    parser.add_argument(
        "--band-stat",
        type=str,
        default="var",
        choices=BAND_STAT_CHOICES,
        help="Statistic used for the shadow band around the mean curve.",
    )
    parser.add_argument(
        "--plot-style",
        type=str,
        default="band",
        choices=PLOT_STYLE_CHOICES,
        help="Visual style for each curve: mean+band or TensorBoard-like raw+smooth.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="light",
        choices=THEME_CHOICES,
        help="Plot theme.",
    )
    parser.add_argument(
        "--style-preset",
        type=str,
        default="default",
        choices=STYLE_PRESET_CHOICES,
        help="Optional plotting preset. qoi_reference_raw matches the sparsity QoI figure style but keeps raw QoI values.",
    )
    parser.add_argument(
        "--raw-alpha",
        type=float,
        default=0.20,
        help="Alpha used for the raw noisy line in raw_smooth mode.",
    )
    parser.add_argument(
        "--raw-line-width",
        type=float,
        default=1.1,
        help="Line width used for the raw noisy line in raw_smooth mode.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Main curve width.",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="comm",
        choices=NORMALIZE_CHOICES,
        help="Normalization mode. Auto keeps the original y-axis scale.",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help="Optional maximum environment step for plotting.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Optional manual y-axis lower bound.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional manual y-axis upper bound.",
    )
    parser.add_argument(
        "--disable-comm-ylim",
        action="store_true",
        help="Disable the automatic late-stage y-axis focus range.",
    )
    parser.add_argument(
        "--focus-after-ratio",
        type=float,
        default=None,
        help="Override the fraction of the left-side x-range ignored when comm-focusing the y-axis.",
    )
    parser.add_argument(
        "--focus-pad-ratio",
        type=float,
        default=None,
        help="Override the y-axis padding ratio used by the automatic focus range.",
    )
    parser.add_argument(
        "--legend-outside",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Place legend outside the plotting area instead of comm-placing it inside.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )
    return parser.parse_args()


def parse_head_value(path: Path) -> int | None:
    match = re.search(r"head[_-]?(\d+)$", path.name)
    if match is not None:
        return int(match.group(1))
    match = re.search(r"head(\d+)$", path.name)
    if match is not None:
        return int(match.group(1))
    return None


def iter_head_dirs(runs_dir: Path) -> list[tuple[int, Path]]:
    head_dirs: list[tuple[int, Path]] = []
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue
        head = parse_head_value(path)
        if head is None:
            continue
        head_dirs.append((head, path))
    return sorted(head_dirs, key=lambda item: item[0])


def iter_run_dirs(head_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    if list(head_dir.glob("events.out.tfevents*")):
        run_dirs.append(head_dir)

    for child in sorted(head_dir.iterdir()):
        if child.is_dir() and list(child.glob("events.out.tfevents*")):
            run_dirs.append(child)

    return run_dirs


def iter_csv_series_paths(head_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in head_dir.glob("*.csv")
        if path.is_file() and path.name.lower().startswith("head")
    )


def smooth_series(values: np.ndarray, smooth: float) -> np.ndarray:
    if values.size == 0 or smooth <= 0.0:
        return values.copy()
    if not 0.0 <= smooth < 1.0:
        raise ValueError(f"--smooth must be in [0, 1), got {smooth}")

    out = np.empty_like(values, dtype=np.float64)
    accumulator = 0.0
    debias = 0.0
    for idx, point in enumerate(values):
        accumulator = accumulator * smooth + (1.0 - smooth) * float(point)
        debias = debias * smooth + (1.0 - smooth)
        out[idx] = accumulator / debias
    return out


def centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if window <= 1:
        return np.zeros_like(values)
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1, center=True)
        .std(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def rolling_var(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if window <= 1:
        return np.zeros_like(values)
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1, center=True)
        .var(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def finalize_series_rows(
    rows: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not rows:
        return None

    rows.sort(key=lambda item: (item[0], item[1]))
    dedup: dict[float, float] = {}
    for step, _, value in rows:
        dedup[step] = value

    steps = np.asarray(sorted(dedup.keys()), dtype=np.float64)
    values = np.asarray([dedup[step] for step in steps], dtype=np.float64)
    if steps.size == 0:
        return None

    if steps[0] > 0:
        steps = np.insert(steps, 0, 0.0)
        values = np.insert(values, 0, values[0])

    return steps, values


def load_scalar_series(run_dir: Path, metric: str) -> tuple[np.ndarray, np.ndarray] | None:
    ensure_tensorboard_import()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows: list[tuple[float, float, float]] = []
    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    for event_path in event_files:
        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        tags = set(accumulator.Tags().get("scalars", []))
        if metric not in tags:
            continue
        for event in accumulator.Scalars(metric):
            step = float(event.step)
            value = float(event.value)
            wall_time = float(event.wall_time)
            if math.isfinite(step) and math.isfinite(value):
                rows.append((step, wall_time, value))

    return finalize_series_rows(rows)


def load_csv_series(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    frame = pd.read_csv(csv_path)
    normalized_columns = {str(column).strip().lower(): column for column in frame.columns}
    if "step" not in normalized_columns or "value" not in normalized_columns:
        return None

    step_values = pd.to_numeric(
        frame[normalized_columns["step"]],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    value_values = pd.to_numeric(
        frame[normalized_columns["value"]],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    wall_column = normalized_columns.get("wall time")
    if wall_column is not None:
        wall_values = pd.to_numeric(frame[wall_column], errors="coerce").to_numpy(dtype=np.float64)
    else:
        wall_values = np.arange(frame.shape[0], dtype=np.float64)

    rows: list[tuple[float, float, float]] = []
    for idx, (step, value) in enumerate(zip(step_values, value_values)):
        if not math.isfinite(step) or not math.isfinite(value):
            continue
        wall_time = float(wall_values[idx]) if idx < wall_values.size else float(idx)
        if not math.isfinite(wall_time):
            wall_time = float(idx)
        rows.append((float(step), wall_time, float(value)))

    return finalize_series_rows(rows)


def resolve_source_mode(runs_dir: Path, requested_source: str) -> str:
    if requested_source != "comm":
        return requested_source

    for _, head_dir in iter_head_dirs(runs_dir):
        if iter_run_dirs(head_dir):
            return "tensorboard"

    for _, head_dir in iter_head_dirs(runs_dir):
        if iter_csv_series_paths(head_dir):
            return "csv"

    return "tensorboard"


def iter_series_paths(head_dir: Path, source_mode: str) -> list[Path]:
    if source_mode == "csv":
        return iter_csv_series_paths(head_dir)
    return iter_run_dirs(head_dir)


def load_series(
    input_path: Path,
    source_mode: str,
    metric: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if source_mode == "csv":
        return load_csv_series(input_path)
    if metric is None:
        raise ValueError("--metric is required when reading TensorBoard event files.")
    return load_scalar_series(input_path, metric)


def aggregate_head_runs(
    run_paths: list[Path],
    source_mode: str,
    metric: str | None,
    smooth: float,
    post_smooth_window: int,
    grid_points: int,
    band_window: int,
    band_stat: str,
    max_step: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    raw_series_list: list[tuple[np.ndarray, np.ndarray]] = []
    smooth_series_list: list[tuple[np.ndarray, np.ndarray]] = []
    for run_path in run_paths:
        series = load_series(run_path, source_mode, metric)
        if series is None:
            continue
        steps, values = series
        if max_step is not None:
            mask = steps <= max_step
            steps = steps[mask]
            values = values[mask]
            if steps.size == 0:
                continue
        raw_series_list.append((steps, values.copy()))
        smooth_values = smooth_series(values, smooth)
        smooth_values = centered_rolling_mean(smooth_values, post_smooth_window)
        smooth_series_list.append((steps, smooth_values))

    if not smooth_series_list:
        return None

    common_max = min(float(steps[-1]) for steps, _ in smooth_series_list)
    common_min = min(float(steps[0]) for steps, _ in smooth_series_list)
    if common_max <= common_min:
        return None

    grid = np.linspace(common_min, common_max, grid_points, dtype=np.float64)
    aligned_raw = [np.interp(grid, steps, values) for steps, values in raw_series_list]
    raw_curves = np.stack(aligned_raw, axis=0)
    raw_mean = raw_curves.mean(axis=0)

    aligned_smooth = [np.interp(grid, steps, values) for steps, values in smooth_series_list]
    smooth_curves = np.stack(aligned_smooth, axis=0)
    smooth_mean = smooth_curves.mean(axis=0)

    if smooth_curves.shape[0] > 1:
        if band_stat == "var":
            band = smooth_curves.var(axis=0, ddof=0)
        else:
            band = smooth_curves.std(axis=0, ddof=0)
    else:
        band = (
            rolling_var(smooth_curves[0], band_window)
            if band_stat == "var"
            else rolling_std(smooth_curves[0], band_window)
        )

    return grid, raw_mean, smooth_mean, band


def infer_shared_max_step(
    runs_dir: Path,
    source_mode: str,
    metric: str | None,
) -> float | None:
    per_head_limits: list[float] = []
    for _, head_dir in iter_head_dirs(runs_dir):
        run_limits: list[float] = []
        for run_path in iter_series_paths(head_dir, source_mode):
            series = load_series(run_path, source_mode, metric)
            if series is None:
                continue
            steps, _ = series
            if steps.size:
                run_limits.append(float(steps[-1]))
        if run_limits:
            per_head_limits.append(min(run_limits))
    if not per_head_limits:
        return None
    return min(per_head_limits)


def resolve_normalize_mode(source_mode: str, requested_mode: str) -> str:
    if requested_mode != "comm":
        return requested_mode
    return "none"


def normalize_plot_payloads(
    aggregated_payloads: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    normalize_mode: str,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if normalize_mode == "none" or not aggregated_payloads:
        return aggregated_payloads

    normalized: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    if normalize_mode == "global_minmax":
        lower = min(
            float(min(np.min(raw_mean), np.min(smooth_mean - band)))
            for _, _, raw_mean, smooth_mean, band in aggregated_payloads
        )
        upper = max(
            float(max(np.max(raw_mean), np.max(smooth_mean + band)))
            for _, _, raw_mean, smooth_mean, band in aggregated_payloads
        )
        scale = max(upper - lower, 1e-9)
        for head, grid, raw_mean, smooth_mean, band in aggregated_payloads:
            normalized.append(
                (
                    head,
                    grid,
                    (raw_mean - lower) / scale,
                    (smooth_mean - lower) / scale,
                    band / scale,
                )
            )
        return normalized

    if normalize_mode == "series_minmax":
        for head, grid, raw_mean, smooth_mean, band in aggregated_payloads:
            lower = float(min(np.min(raw_mean), np.min(smooth_mean - band)))
            upper = float(max(np.max(raw_mean), np.max(smooth_mean + band)))
            scale = max(upper - lower, 1e-9)
            normalized.append(
                (
                    head,
                    grid,
                    (raw_mean - lower) / scale,
                    (smooth_mean - lower) / scale,
                    band / scale,
                )
            )
        return normalized

    raise ValueError(f"Unsupported normalize mode: {normalize_mode}")


def resolve_focus_config(
    metric: str,
    focus_after_ratio: float | None,
    focus_pad_ratio: float | None,
) -> dict[str, float]:
    config = dict(
        DEFAULT_FOCUS_CONFIG.get(
            metric,
            {
                "ignore_left_ratio": 0.10,
                "lower_quantile": 0.04,
                "upper_quantile": 0.96,
                "pad_ratio": 0.18,
            },
        )
    )
    if focus_after_ratio is not None:
        config["ignore_left_ratio"] = float(focus_after_ratio)
    if focus_pad_ratio is not None:
        config["pad_ratio"] = float(focus_pad_ratio)
    return config


def compute_auto_focus_ylim(
    metric: str,
    aggregated_payloads: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    focus_after_ratio: float | None,
    focus_pad_ratio: float | None,
) -> tuple[float | None, float | None]:
    if not aggregated_payloads:
        return None, None

    config = resolve_focus_config(metric, focus_after_ratio, focus_pad_ratio)
    ignore_left_ratio = min(max(config["ignore_left_ratio"], 0.0), 0.95)
    lower_quantile = min(max(config["lower_quantile"], 0.0), 1.0)
    upper_quantile = min(max(config["upper_quantile"], 0.0), 1.0)
    pad_ratio = max(config["pad_ratio"], 0.0)

    global_start = min(float(grid[0]) for _, grid, _, _ in aggregated_payloads)
    global_end = max(float(grid[-1]) for _, grid, _, _ in aggregated_payloads)
    focus_start = global_start + ignore_left_ratio * (global_end - global_start)

    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for _, grid, mean, band in aggregated_payloads:
        mask = grid >= focus_start
        if not np.any(mask):
            mask = np.ones_like(grid, dtype=bool)
        lows.append((mean - band)[mask])
        highs.append((mean + band)[mask])

    low_values = np.concatenate(lows)
    high_values = np.concatenate(highs)
    if low_values.size == 0 or high_values.size == 0:
        return None, None

    lower = float(np.quantile(low_values, lower_quantile))
    upper = float(np.quantile(high_values, upper_quantile))
    span = max(upper - lower, 1e-6)
    pad = pad_ratio * span

    lower -= pad
    upper += pad
    if metric in LOWER_IS_BETTER:
        lower = max(0.0, lower)

    return lower, upper


def compute_full_range_ylim(
    metric: str,
    aggregated_payloads: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float | None, float | None]:
    if not aggregated_payloads:
        return None, None

    lows = np.concatenate([(mean - band) for _, _, mean, band in aggregated_payloads])
    highs = np.concatenate([(mean + band) for _, _, mean, band in aggregated_payloads])
    if lows.size == 0 or highs.size == 0:
        return None, None

    lower = float(np.min(lows))
    upper = float(np.max(highs))
    span = max(upper - lower, 1e-6)
    pad_ratio = 0.04 if metric == "QoI" else 0.05
    pad = pad_ratio * span

    lower -= pad
    upper += pad
    if metric in LOWER_IS_BETTER:
        lower = max(0.0, lower)

    return lower, upper


def choose_inside_legend_location(
    aggregated_payloads: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> str:
    x0, x1 = xlim
    y0, y1 = ylim
    x_span = max(x1 - x0, 1e-9)
    y_span = max(y1 - y0, 1e-9)

    boxes = {
        "upper left": (x0, x0 + 0.38 * x_span, y0 + 0.68 * y_span, y1),
        "upper right": (x1 - 0.38 * x_span, x1, y0 + 0.68 * y_span, y1),
        "lower left": (x0, x0 + 0.38 * x_span, y0, y0 + 0.34 * y_span),
        "lower right": (x1 - 0.38 * x_span, x1, y0, y0 + 0.34 * y_span),
    }

    scores: dict[str, float] = {}
    for loc, (xa, xb, ya, yb) in boxes.items():
        score = 0.0
        for _, grid, mean, band in aggregated_payloads:
            mask = (grid >= xa) & (grid <= xb)
            if not np.any(mask):
                continue
            local_upper = (mean + band)[mask]
            local_lower = (mean - band)[mask]
            overlap = np.maximum(np.minimum(local_upper, yb) - np.maximum(local_lower, ya), 0.0)
            score += float(overlap.sum() / y_span)
        scores[loc] = score

    return min(scores, key=scores.get)


def resolve_ylabel(
    metric: str | None,
    source_mode: str,
    normalize_mode: str,
    custom_ylabel: str | None,
) -> str:
    if custom_ylabel:
        return custom_ylabel

    if metric is not None:
        base_label = TITLE_MAP.get(metric, metric)
    elif source_mode == "csv":
        base_label = "Value"
    else:
        base_label = "Metric"

    if normalize_mode != "none":
        return f"Normalized {base_label}"
    return base_label


def build_default_output_name(
    metric: str | None,
    source_mode: str,
    normalize_mode: str,
) -> str:
    token = metric if metric is not None else source_mode
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token).strip("_") or "plot"
    suffix = "_normalized" if normalize_mode != "none" else ""
    return f"{token}_head_curves{suffix}.png"


def format_head_label(head: int, style_preset: str) -> str:
    if style_preset == "qoi_reference_raw":
        return f"head={head}"
    return f"Head {head}"


def resolve_theme(theme: str, plot_style: str) -> dict[str, str]:
    resolved_theme = theme
    if resolved_theme == "comm":
        resolved_theme = "dark" if plot_style == "raw_smooth" else "light"

    if resolved_theme == "dark":
        return {
            "figure_facecolor": "#303030",
            "axes_facecolor": "#303030",
            "grid_color": "#cfcfcf",
            "spine_color": "#bfbfbf",
            "text_color": "#f2f2f2",
            "legend_facecolor": "#303030",
            "legend_edgecolor": "#b5b5b5",
        }
    return {
        "figure_facecolor": "#ffffff",
        "axes_facecolor": "#ffffff",
        "grid_color": "#b8b8b8",
        "spine_color": "#444444",
        "text_color": "#111111",
        "legend_facecolor": "#ffffff",
        "legend_edgecolor": "#d0d0d0",
    }


def build_plot(
    runs_dir: Path,
    source_mode: str,
    metric: str | None,
    output_path: Path,
    title: str,
    ylabel: str | None,
    smooth: float,
    post_smooth_window: int,
    grid_points: int,
    band_window: int,
    band_alpha: float,
    band_stat: str,
    plot_style: str,
    style_preset: str,
    theme: str,
    raw_alpha: float,
    raw_line_width: float,
    line_width: float,
    normalize_mode: str,
    max_step: float | None,
    ymin: float | None,
    ymax: float | None,
    disable_auto_ylim: bool,
    focus_after_ratio: float | None,
    focus_pad_ratio: float | None,
    legend_outside: bool,
    dpi: int,
) -> None:
    is_reference_raw_style = style_preset == "qoi_reference_raw"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    figure_size = (8.8, 5.9) if is_reference_raw_style else (5.1, 4.2)
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    theme_config = resolve_theme(theme, plot_style)
    fig.patch.set_facecolor(theme_config["figure_facecolor"])
    ax.set_facecolor(theme_config["axes_facecolor"])
    for spine in ax.spines.values():
        spine.set_color(theme_config["spine_color"])

    if max_step is None:
        max_step = infer_shared_max_step(runs_dir, source_mode, metric)

    plot_payloads: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for head, head_dir in iter_head_dirs(runs_dir):
        aggregated = aggregate_head_runs(
            run_paths=iter_series_paths(head_dir, source_mode),
            source_mode=source_mode,
            metric=metric,
            smooth=smooth,
            post_smooth_window=post_smooth_window,
            grid_points=grid_points,
            band_window=band_window,
            band_stat=band_stat,
            max_step=max_step,
        )
        if aggregated is None:
            continue
        grid, raw_mean, smooth_mean, band = aggregated
        plot_payloads.append((head, grid, raw_mean, smooth_mean, band))

    plot_payloads = normalize_plot_payloads(plot_payloads, normalize_mode)

    if not plot_payloads:
        label_token = metric if metric is not None else source_mode
        raise ValueError(f"No valid {label_token} series found under {runs_dir}")

    aggregated_payloads = [
        (head, grid, smooth_mean, band) for head, grid, _, smooth_mean, band in plot_payloads
    ]

    for head, grid, raw_mean, smooth_mean, band in plot_payloads:
        color = COLOR_MAP.get(head, "#7f7f7f")
        if plot_style == "raw_smooth":
            ax.plot(
                grid,
                raw_mean,
                linewidth=raw_line_width,
                color=color,
                alpha=raw_alpha,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                grid,
                smooth_mean,
                linewidth=line_width,
                color=color,
                label=format_head_label(head, style_preset),
                solid_capstyle="round",
                zorder=3,
            )
        else:
            ax.plot(
                grid,
                smooth_mean,
                linewidth=line_width,
                color=color,
                label=format_head_label(head, style_preset),
                solid_capstyle="round",
                zorder=3,
            )
            ax.fill_between(
                grid,
                smooth_mean - band,
                smooth_mean + band,
                color=color,
                alpha=band_alpha,
                linewidth=0.0,
                zorder=2,
            )

    title_fontsize = 18 if is_reference_raw_style else 14
    axis_fontsize = 12 if is_reference_raw_style else 10
    ax.set_title(title, fontsize=title_fontsize, pad=6, color=theme_config["text_color"])
    ax.set_xlabel("Environment steps", fontsize=axis_fontsize, color=theme_config["text_color"])
    ax.set_ylabel(
        resolve_ylabel(
            metric=metric,
            source_mode=source_mode,
            normalize_mode=normalize_mode,
            custom_ylabel=ylabel,
        ),
        fontsize=axis_fontsize,
        color=theme_config["text_color"],
    )
    ax.set_axisbelow(True)
    if is_reference_raw_style:
        ax.grid(False)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=6,
            width=1.4,
            labelsize=10,
            colors=theme_config["text_color"],
        )
        ax.minorticks_off()
        for spine in ax.spines.values():
            spine.set_linewidth(1.7)
    else:
        ax.grid(True, color=theme_config["grid_color"], alpha=0.65, linewidth=1.0)
        ax.tick_params(labelsize=9, colors=theme_config["text_color"])
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.margins(x=0.01)
    ax.set_xlim(left=0)
    ax.xaxis.get_offset_text().set_color(theme_config["text_color"])
    ax.yaxis.get_offset_text().set_color(theme_config["text_color"])

    if ymin is not None or ymax is not None:
        resolved_ymin, resolved_ymax = ymin, ymax
    elif is_reference_raw_style or disable_auto_ylim or (focus_after_ratio is None and focus_pad_ratio is None):
        resolved_ymin, resolved_ymax = compute_full_range_ylim(
            metric=metric,
            aggregated_payloads=aggregated_payloads,
        )
    else:
        resolved_ymin, resolved_ymax = compute_auto_focus_ylim(
            metric=metric,
            aggregated_payloads=aggregated_payloads,
            focus_after_ratio=focus_after_ratio,
            focus_pad_ratio=focus_pad_ratio,
        )

    if resolved_ymin is not None or resolved_ymax is not None:
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(
            bottom=resolved_ymin if resolved_ymin is not None else current_ymin,
            top=resolved_ymax if resolved_ymax is not None else current_ymax,
        )
    elif metric in LOWER_IS_BETTER:
        current_ymin, _ = ax.get_ylim()
        ax.set_ylim(bottom=max(0.0, current_ymin))

    if legend_outside:
        legend = ax.legend(
            loc="lower left",
            bbox_to_anchor=(1.01, 0.02),
            frameon=True,
            fontsize=13 if is_reference_raw_style else 10,
            title=None,
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    else:
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            fontsize=13 if is_reference_raw_style else 8.5,
            title=None,
        )
        fig.tight_layout()

    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_edgecolor(theme_config["legend_edgecolor"])
    legend.get_frame().set_facecolor(theme_config["legend_facecolor"])
    for text in legend.get_texts():
        text.set_color(theme_config["text_color"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    source_mode = resolve_source_mode(runs_dir, args.source)
    normalize_mode = resolve_normalize_mode(source_mode, args.normalize)
    resolved_ylabel = args.ylabel
    resolved_smooth = args.smooth
    resolved_post_smooth_window = args.post_smooth_window
    resolved_band_alpha = args.band_alpha
    resolved_band_stat = args.band_stat
    resolved_plot_style = args.plot_style
    resolved_theme = args.theme
    resolved_line_width = args.line_width

    if args.style_preset == "qoi_reference_raw":
        normalize_mode = "none"
        resolved_ylabel = args.ylabel or "QoI"
        resolved_smooth = 0.35
        resolved_post_smooth_window = 5
        resolved_band_alpha = 0.22
        resolved_band_stat = "std"
        resolved_plot_style = "band"
        resolved_theme = "light"
        resolved_line_width = 1.9

    if source_mode == "tensorboard" and args.metric is None:
        raise ValueError("--metric is required when plotting TensorBoard runs.")

    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            runs_dir
            / "figures"
            / build_default_output_name(
                metric=args.metric,
                source_mode=source_mode,
                normalize_mode=normalize_mode,
            )
        ).resolve()
    )
    build_plot(
        runs_dir=runs_dir,
        source_mode=source_mode,
        metric=args.metric,
        output_path=output_path,
        title=args.title,
        ylabel=resolved_ylabel,
        smooth=resolved_smooth,
        post_smooth_window=resolved_post_smooth_window,
        grid_points=args.grid_points,
        band_window=args.band_window,
        band_alpha=resolved_band_alpha,
        band_stat=resolved_band_stat,
        plot_style=resolved_plot_style,
        style_preset=args.style_preset,
        theme=resolved_theme,
        raw_alpha=args.raw_alpha,
        raw_line_width=args.raw_line_width,
        line_width=resolved_line_width,
        normalize_mode=normalize_mode,
        max_step=args.max_step,
        ymin=args.ymin,
        ymax=args.ymax,
        disable_auto_ylim=args.disable_auto_ylim,
        focus_after_ratio=args.focus_after_ratio,
        focus_pad_ratio=args.focus_pad_ratio,
        legend_outside=args.legend_outside,
        dpi=args.dpi,
    )
    print(f"[plot_nhead_tensorboard_curves_optimized] figure saved to {output_path}")


if __name__ == "__main__":
    main()
