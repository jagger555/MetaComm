from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import get_scaled_sizes, add_font_scale_arg


DEFAULT_COLORS = {
    "Head 1": "#1f77b4",
    "Head 2": "#ff7f0e",
    "Head 4": "#2ca02c",
    "Head 8": "#d62728",
    "Head 16": "#9467bd",
}


def load_csv_series(csv_path):
    df = pd.read_csv(csv_path)
    columns = {c.lower(): c for c in df.columns}

    step_col = None
    value_col = None
    for candidate in ("step", "steps", "global_step"):
        if candidate in columns:
            step_col = columns[candidate]
            break
    for candidate in ("value", "values", "scalar"):
        if candidate in columns:
            value_col = columns[candidate]
            break

    if step_col is None or value_col is None:
        raise ValueError(
            f"CSV {csv_path} must contain Step/Value columns. Found: {list(df.columns)}"
        )

    x = df[step_col].to_numpy(dtype=np.float64)
    y = df[value_col].to_numpy(dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    return unique_x, y


def ema_smooth(y, alpha):
    if alpha <= 0:
        return y.copy()
    out = np.empty_like(y, dtype=np.float64)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def aggregate_runs(csv_paths, num_points=300, smooth_alpha=0.0):
    runs = [load_csv_series(path) for path in csv_paths]
    min_x = min(x[0] for x, _ in runs)
    max_x = min(x[-1] for x, _ in runs)
    if max_x <= min_x:
        raise ValueError("Run step ranges do not overlap enough to aggregate.")

    grid = np.linspace(min_x, max_x, num_points)
    ys = []
    for x, y in runs:
        interp = np.interp(grid, x, y)
        if smooth_alpha > 0:
            interp = ema_smooth(interp, smooth_alpha)
        ys.append(interp)
    ys = np.stack(ys, axis=0)
    mean = ys.mean(axis=0)
    std = ys.std(axis=0)
    return grid, mean, std


def plot_subplot(ax, subplot_cfg, num_points, smooth_alpha, show_legend, legend_fontsize, font_scale=1.0):
    sizes = get_scaled_sizes(font_scale)
    methods = subplot_cfg["methods"]
    x_label = subplot_cfg.get("x_label", "Steps")
    y_label = subplot_cfg.get("y_label", "Mean Episode Reward")
    title = subplot_cfg.get("title", "")
    ylim = subplot_cfg.get("ylim")
    xlim = subplot_cfg.get("xlim")

    for method_name, method_cfg in methods.items():
        if isinstance(method_cfg, dict):
            csv_paths = method_cfg["csvs"]
            color = method_cfg.get("color", DEFAULT_COLORS.get(method_name))
            label = method_cfg.get("label", method_name)
        else:
            csv_paths = method_cfg
            color = DEFAULT_COLORS.get(method_name)
            label = method_name

        grid, mean, std = aggregate_runs(
            csv_paths=csv_paths,
            num_points=num_points,
            smooth_alpha=smooth_alpha,
        )
        ax.plot(grid, mean, linewidth=1.8, label=label, color=color)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_title(title, fontsize=sizes["title"])
    ax.set_xlabel(x_label, fontsize=sizes["label"])
    if y_label:
        ax.set_ylabel(y_label, fontsize=sizes["label"])
    ax.tick_params(labelsize=sizes["tick"])
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.grid(True, linestyle="--", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if show_legend:
        ax.legend(fontsize=legend_fontsize, frameon=True)


def build_figure(cfg, output_path, font_scale=1.0):
    sizes = get_scaled_sizes(font_scale)
    figure_cfg = cfg.get("figure", {})
    subplots = cfg["subplots"]

    nrows = figure_cfg.get("nrows", 1)
    ncols = figure_cfg.get("ncols", len(subplots))
    figsize = figure_cfg.get("figsize", [6 * ncols, 4.5 * nrows])
    sharey = figure_cfg.get("sharey", False)
    dpi = figure_cfg.get("dpi", 300)
    num_points = figure_cfg.get("num_points", 300)
    smooth_alpha = figure_cfg.get("smooth_alpha", 0.0)
    legend_mode = figure_cfg.get("legend", "each")
    legend_fontsize = figure_cfg.get("legend_fontsize", sizes["legend"])

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)

    for idx, subplot_cfg in enumerate(subplots):
        show_legend = legend_mode == "each"
        plot_subplot(
            ax=axes[idx],
            subplot_cfg=subplot_cfg,
            num_points=num_points,
            smooth_alpha=smooth_alpha,
            show_legend=show_legend,
            legend_fontsize=legend_fontsize,
            font_scale=font_scale,
        )

    for idx in range(len(subplots), len(axes)):
        axes[idx].axis("off")

    if legend_mode == "global" and len(subplots) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=True,
            fontsize=legend_fontsize,
        )

    if "suptitle" in figure_cfg:
        fig.suptitle(figure_cfg["suptitle"], fontsize=sizes["suptitle"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard-exported CSV runs with mean curve and std shading."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config describing subplots and CSV files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tb_csv_compare.png",
        help="Output image path.",
    )
    add_font_scale_arg(parser, default=1.0)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_figure(cfg, output_path, font_scale=args.font_scale)
    print(f"[plot_tb_csv_runs] saved figure to {output_path}")


if __name__ == "__main__":
    main()
