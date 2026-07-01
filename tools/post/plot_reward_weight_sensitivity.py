from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "reward_pref_analysis"

AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 11
BEST_METRIC_COLUMNS = (
    "test_QoI",
    "test_episodic_aoi",
    "test_aoi_satis_ratio",
    "test_tx_satis_ratio",
    "test_data_satis_ratio",
    "test_tx_reward",
    "test_good_reward",
    "test_aoi_penalty_reward",
    "test_effective_aoi_task_reward",
    "test_effective_tx_task_reward",
    "test_effective_joint_task_reward",
)
CURVE_METRIC_COLUMNS = (
    "test_episodic_aoi",
    "test_aoi_satis_ratio",
    "test_tx_satis_ratio",
    "test_QoI",
)


@dataclass
class RunSelection:
    run_dir: Path
    dataset: str
    algo: str
    seed: int | None
    reward_pref: float
    effective_aVPS: float
    effective_tVPS: float
    total_weight: float
    best_step: int
    best_metrics: dict[str, float]
    curve_frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate reward preference sweeps from scalars.csv, select the best "
            "test_QoI checkpoint per run, and generate trade-off figures."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Root runs directory or a specific sweep directory to scan.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Optional group subdirectory under runs/, for example weight_pref.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NCSU",
        help="Dataset filter.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="MetaComm",
        help="Algorithm filter.",
    )
    parser.add_argument(
        "--total-weight",
        type=float,
        default=0.4,
        help="Expected sum of effective aVPS and tVPS when reward_pref must be derived from legacy runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV summaries and figures will be written.",
    )
    parser.add_argument(
        "--representative-prefs",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.8],
        help="Preference values used for the training-curve figure.",
    )
    return parser.parse_args()


def approx_equal(left: float, right: float, atol: float = 1e-8) -> bool:
    return abs(left - right) <= atol


def resolve_scan_root(runs_dir: Path, group: str | None) -> Path:
    if group is None:
        return runs_dir
    grouped = runs_dir / group
    return grouped if grouped.exists() else runs_dir


def load_params(params_path: Path) -> dict | None:
    try:
        return json.loads(params_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_reward_pref(input_args: dict, total_weight: float) -> tuple[float | None, float | None, float | None, float | None]:
    effective_a = input_args.get("effective_aVPS", input_args.get("aVPS"))
    effective_t = input_args.get("effective_tVPS", input_args.get("tVPS"))
    if effective_a is None or effective_t is None:
        return None, None, None, None

    effective_a = float(effective_a)
    effective_t = float(effective_t)
    derived_total = effective_a + effective_t
    reward_pref = input_args.get("reward_pref")
    if reward_pref is not None:
        return float(reward_pref), effective_a, effective_t, derived_total

    if derived_total <= 0.0:
        return None, effective_a, effective_t, derived_total
    if total_weight is not None and not approx_equal(derived_total, total_weight, atol=1e-6):
        return None, effective_a, effective_t, derived_total
    return effective_t / derived_total, effective_a, effective_t, derived_total


def load_scalar_frame(csv_path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return None
    required_cols = {"step", "tag", "value"}
    if not required_cols.issubset(frame.columns):
        return None
    frame = frame.dropna(subset=["step", "tag", "value"]).copy()
    if frame.empty:
        return None
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["step", "value"])
    if frame.empty:
        return None
    frame["step"] = frame["step"].astype(int)
    return frame


def build_eval_frame(scalar_frame: pd.DataFrame) -> pd.DataFrame:
    subset = scalar_frame[scalar_frame["tag"].str.startswith("test_")].copy()
    if subset.empty:
        return pd.DataFrame()
    pivot = (
        subset.pivot_table(index="step", columns="tag", values="value", aggfunc="last")
        .sort_index()
        .reset_index()
    )
    return pivot


def select_best_row(eval_frame: pd.DataFrame) -> pd.Series | None:
    if eval_frame.empty or "test_QoI" not in eval_frame.columns:
        return None
    ranked = eval_frame.copy()
    ranked["__qoi_sort"] = ranked["test_QoI"].fillna(-np.inf)
    ranked["__aoi_sort"] = ranked.get("test_episodic_aoi", pd.Series(np.inf, index=ranked.index)).fillna(np.inf)
    ranked["__tx_sort"] = ranked.get("test_tx_satis_ratio", pd.Series(-np.inf, index=ranked.index)).fillna(-np.inf)
    ranked = ranked.sort_values(
        by=["__qoi_sort", "__aoi_sort", "__tx_sort", "step"],
        ascending=[False, True, False, True],
    )
    return ranked.iloc[0]


def collect_runs(
    scan_root: Path,
    dataset_filter: str,
    algo_filter: str,
    total_weight: float,
) -> list[RunSelection]:
    selections: list[RunSelection] = []

    for params_path in scan_root.rglob("params.json"):
        if params_path.is_relative_to(DEFAULT_RUNS_DIR / "R"):
            continue

        payload = load_params(params_path)
        if payload is None:
            continue
        input_args = payload.get("input_args", {})
        dataset = str(input_args.get("dataset", ""))
        algo = str(input_args.get("algo", ""))
        if dataset_filter and dataset != dataset_filter:
            continue
        if algo_filter and algo != algo_filter:
            continue

        reward_pref, effective_a, effective_t, derived_total = extract_reward_pref(input_args, total_weight)
        if reward_pref is None or effective_a is None or effective_t is None or derived_total is None:
            continue

        scalar_path = params_path.parent / "scalars.csv"
        scalar_frame = load_scalar_frame(scalar_path)
        if scalar_frame is None:
            continue
        eval_frame = build_eval_frame(scalar_frame)
        best_row = select_best_row(eval_frame)
        if best_row is None:
            continue

        best_metrics = {}
        for column in BEST_METRIC_COLUMNS:
            if column in best_row and pd.notna(best_row[column]):
                best_metrics[column] = float(best_row[column])

        curve_columns = ["step", *[col for col in CURVE_METRIC_COLUMNS if col in eval_frame.columns]]
        curve_frame = eval_frame[curve_columns].copy()

        selections.append(
            RunSelection(
                run_dir=params_path.parent,
                dataset=dataset,
                algo=algo,
                seed=int(input_args["seed"]) if "seed" in input_args else None,
                reward_pref=float(reward_pref),
                effective_aVPS=float(effective_a),
                effective_tVPS=float(effective_t),
                total_weight=float(derived_total),
                best_step=int(best_row["step"]),
                best_metrics=best_metrics,
                curve_frame=curve_frame,
            )
        )

    return sorted(selections, key=lambda item: (item.reward_pref, item.seed if item.seed is not None else -1, item.run_dir.name))


def build_best_run_frame(selections: Iterable[RunSelection]) -> pd.DataFrame:
    rows = []
    for item in selections:
        row = {
            "run_dir": str(item.run_dir),
            "dataset": item.dataset,
            "algo": item.algo,
            "seed": item.seed,
            "reward_pref": item.reward_pref,
            "effective_aVPS": item.effective_aVPS,
            "effective_tVPS": item.effective_tVPS,
            "total_weight": item.total_weight,
            "best_step": item.best_step,
        }
        for key, value in item.best_metrics.items():
            row[f"best_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["reward_pref", "seed", "run_dir"]).reset_index(drop=True)


def build_summary_frame(best_run_frame: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {
        "num_runs": ("run_dir", "count"),
        "effective_aVPS_mean": ("effective_aVPS", "mean"),
        "effective_tVPS_mean": ("effective_tVPS", "mean"),
        "total_weight_mean": ("total_weight", "mean"),
    }
    metric_columns = (
        "best_test_episodic_aoi",
        "best_test_aoi_satis_ratio",
        "best_test_tx_satis_ratio",
        "best_test_QoI",
        "best_test_data_satis_ratio",
        "best_test_tx_reward",
        "best_test_effective_aoi_task_reward",
        "best_test_effective_tx_task_reward",
        "best_test_effective_joint_task_reward",
    )
    for column in metric_columns:
        if column in best_run_frame.columns:
            agg_spec[f"{column}_mean"] = (column, "mean")
            agg_spec[f"{column}_std"] = (column, "std")

    summary = best_run_frame.groupby("reward_pref", as_index=False).agg(**agg_spec)
    expected_summary_columns = (
        "best_test_episodic_aoi_mean",
        "best_test_episodic_aoi_std",
        "best_test_aoi_satis_ratio_mean",
        "best_test_aoi_satis_ratio_std",
        "best_test_tx_satis_ratio_mean",
        "best_test_tx_satis_ratio_std",
        "best_test_QoI_mean",
        "best_test_QoI_std",
        "best_test_data_satis_ratio_mean",
        "best_test_data_satis_ratio_std",
        "best_test_tx_reward_mean",
        "best_test_tx_reward_std",
        "best_test_effective_aoi_task_reward_mean",
        "best_test_effective_aoi_task_reward_std",
        "best_test_effective_tx_task_reward_mean",
        "best_test_effective_tx_task_reward_std",
        "best_test_effective_joint_task_reward_mean",
        "best_test_effective_joint_task_reward_std",
    )
    for column in expected_summary_columns:
        if column not in summary.columns:
            summary[column] = np.nan
    return summary.fillna(0.0).sort_values("reward_pref").reset_index(drop=True)


def build_curve_frame(selections: Iterable[RunSelection]) -> pd.DataFrame:
    rows = []
    for item in selections:
        for _, curve_row in item.curve_frame.iterrows():
            row = {
                "run_dir": str(item.run_dir),
                "seed": item.seed,
                "reward_pref": item.reward_pref,
                "step": int(curve_row["step"]),
            }
            for metric in CURVE_METRIC_COLUMNS:
                if metric in curve_row and pd.notna(curve_row[metric]):
                    row[metric] = float(curve_row[metric])
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["reward_pref", "seed", "step"]).reset_index(drop=True)


def choose_representative_prefs(targets: Iterable[float], available: Iterable[float]) -> list[float]:
    available_values = sorted({float(value) for value in available})
    if not available_values:
        return []
    chosen: list[float] = []
    for target in targets:
        best_match = min(available_values, key=lambda candidate: (abs(candidate - target), candidate))
        if best_match not in chosen:
            chosen.append(best_match)
    return chosen


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary_frame: pd.DataFrame, output_path: Path, dataset: str, algo: str) -> None:
    x = summary_frame["reward_pref"].to_numpy(dtype=float)
    tx_mean = summary_frame["best_test_tx_satis_ratio_mean"].to_numpy(dtype=float)
    aoi_mean = summary_frame["best_test_aoi_satis_ratio_mean"].to_numpy(dtype=float)

    fig, ax_left = plt.subplots(figsize=(8.0, 5.0))
    ax_right = ax_left.twinx()

    left_color = "#0b43ff"
    right_color = "#ff2020"

    left_line = ax_left.plot(
        x,
        tx_mean,
        color=left_color,
        marker="o",
        markersize=6.5,
        markerfacecolor="none",
        markeredgewidth=1.5,
        linewidth=1.8,
        label="Throughput satisfaction ratio",
    )
    right_line = ax_right.plot(
        x,
        aoi_mean,
        color=right_color,
        marker="x",
        markersize=6.0,
        linewidth=1.6,
        label="AoI satisfaction ratio",
    )

    ax_left.set_xlabel("Reward preference λ", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax_left.set_ylabel("Throughput satisfaction ratio", fontsize=AXIS_LABEL_SIZE, color=left_color, labelpad=4)
    ax_right.set_ylabel("AoI satisfaction ratio", fontsize=AXIS_LABEL_SIZE, color=right_color, labelpad=4)
    ax_left.set_xticks(x)
    ax_left.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, direction="in", top=True)
    ax_left.tick_params(axis="y", colors=left_color, labelsize=TICK_LABEL_SIZE, direction="in")
    ax_right.tick_params(axis="y", colors=right_color, labelsize=TICK_LABEL_SIZE, direction="in")
    ax_left.spines["left"].set_color(left_color)
    ax_right.spines["right"].set_color(right_color)
    ax_left.grid(False)

    handles = left_line + right_line
    labels = [handle.get_label() for handle in handles]
    ax_left.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.58, 0.48),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=10,
    )

    save_figure(fig, output_path)


def plot_pareto(summary_frame: pd.DataFrame, output_path: Path, dataset: str, algo: str) -> None:
    x = summary_frame["best_test_episodic_aoi_mean"].to_numpy(dtype=float)
    xerr = summary_frame["best_test_episodic_aoi_std"].to_numpy(dtype=float)
    y = summary_frame["best_test_tx_satis_ratio_mean"].to_numpy(dtype=float)
    yerr = summary_frame["best_test_tx_satis_ratio_std"].to_numpy(dtype=float)
    prefs = summary_frame["reward_pref"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(prefs.min()), vmax=float(prefs.max()))

    ax.plot(x, y, color="0.7", linewidth=1.5, zorder=1)
    for idx, reward_pref in enumerate(prefs):
        color = cmap(norm(reward_pref))
        ax.errorbar(
            x[idx],
            y[idx],
            xerr=xerr[idx],
            yerr=yerr[idx],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            markersize=7,
            zorder=2,
        )
        ax.annotate(f"λ={reward_pref:g}", (x[idx], y[idx]), textcoords="offset points", xytext=(5, 6), fontsize=9)

    ax.set_xlabel("Best test episodic AoI", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Best test throughput satisfaction ratio", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.set_title(f"{dataset} | {algo} | AoI-throughput Pareto front", fontsize=AXIS_LABEL_SIZE, pad=10)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.35)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Reward preference λ")

    save_figure(fig, output_path)


def plot_qoi(summary_frame: pd.DataFrame, output_path: Path, dataset: str, algo: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.errorbar(
        summary_frame["reward_pref"].to_numpy(dtype=float),
        summary_frame["best_test_QoI_mean"].to_numpy(dtype=float),
        yerr=summary_frame["best_test_QoI_std"].to_numpy(dtype=float),
        marker="o",
        linewidth=2.0,
        capsize=3,
        color="#2f7d4e",
    )
    ax.set_xlabel("Reward preference λ (0=AoI, 1=throughput)", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Best test QoI", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.set_title(f"{dataset} | {algo} | QoI under reward preference sweep", fontsize=AXIS_LABEL_SIZE, pad=10)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.35)
    save_figure(fig, output_path)


def plot_training_curves(
    curve_frame: pd.DataFrame,
    prefs: list[float],
    output_path: Path,
    dataset: str,
    algo: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(prefs), 1)))
    metric_specs = (
        ("test_episodic_aoi", "Test episodic AoI"),
        ("test_tx_satis_ratio", "Test throughput satisfaction ratio"),
    )

    for color, reward_pref in zip(colors, prefs):
        pref_frame = curve_frame[curve_frame["reward_pref"] == reward_pref]
        if pref_frame.empty:
            continue
        grouped = pref_frame.groupby("step", as_index=False).agg(
            test_episodic_aoi_mean=("test_episodic_aoi", "mean"),
            test_episodic_aoi_std=("test_episodic_aoi", "std"),
            test_tx_satis_ratio_mean=("test_tx_satis_ratio", "mean"),
            test_tx_satis_ratio_std=("test_tx_satis_ratio", "std"),
        ).fillna(0.0)
        label = f"λ={reward_pref:g}"
        for axis, (metric, ylabel) in zip(axes, metric_specs):
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            axis.plot(grouped["step"], grouped[mean_key], color=color, linewidth=2.0, label=label)
            axis.fill_between(
                grouped["step"],
                grouped[mean_key] - grouped[std_key],
                grouped[mean_key] + grouped[std_key],
                color=color,
                alpha=0.18,
            )
            axis.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, labelpad=4)
            axis.tick_params(labelsize=TICK_LABEL_SIZE)
            axis.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_title(f"{dataset} | {algo} | training-time reward preference trajectories", fontsize=AXIS_LABEL_SIZE, pad=10)
    axes[1].set_xlabel("Environment steps", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    axes[0].legend(loc="best", frameon=True)
    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    scan_root = resolve_scan_root(args.runs_dir.resolve(), args.group)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = collect_runs(
        scan_root=scan_root,
        dataset_filter=args.dataset,
        algo_filter=args.algo,
        total_weight=args.total_weight,
    )
    if not selections:
        raise FileNotFoundError(
            f"No matching runs found under {scan_root} for dataset={args.dataset}, algo={args.algo}, total_weight={args.total_weight:g}."
        )

    best_run_frame = build_best_run_frame(selections)
    summary_frame = build_summary_frame(best_run_frame)
    curve_frame = build_curve_frame(selections)

    best_csv_path = output_dir / "reward_pref_best_runs.csv"
    summary_csv_path = output_dir / "reward_pref_summary.csv"
    best_run_frame.to_csv(best_csv_path, index=False)
    summary_frame.to_csv(summary_csv_path, index=False)

    plot_tradeoff(summary_frame, output_dir / "reward_pref_tradeoff.png", args.dataset, args.algo)
    plot_pareto(summary_frame, output_dir / "reward_pref_pareto.png", args.dataset, args.algo)
    plot_qoi(summary_frame, output_dir / "reward_pref_qoi.png", args.dataset, args.algo)

    representative_prefs = choose_representative_prefs(args.representative_prefs, summary_frame["reward_pref"].tolist())
    if representative_prefs and not curve_frame.empty:
        plot_training_curves(
            curve_frame=curve_frame,
            prefs=representative_prefs,
            output_path=output_dir / "reward_pref_training_curves.png",
            dataset=args.dataset,
            algo=args.algo,
        )

    print(f"[plot_reward_weight_sensitivity] best-run summary saved to {best_csv_path}")
    print(f"[plot_reward_weight_sensitivity] aggregated summary saved to {summary_csv_path}")
    print(f"[plot_reward_weight_sensitivity] figures saved under {output_dir}")


if __name__ == "__main__":
    main()
