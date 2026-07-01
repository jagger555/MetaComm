from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fmt_float(value: float) -> str:
    return f"{float(value):g}"


def _setting_label(input_args: dict) -> str:
    schedule = str(input_args.get("tau_schedule", "fixed"))
    if schedule == "linear":
        tau_start = float(input_args.get("tau_start", input_args.get("tau", 1.0)))
        tau_end = float(input_args.get("tau_end", tau_start))
        return f"annealed {_fmt_float(tau_start)}->{_fmt_float(tau_end)}"
    tau = float(input_args.get("tau", 1.0))
    return f"tau={_fmt_float(tau)}"


def _setting_sort_key(input_args: dict) -> tuple[int, float, float]:
    schedule = str(input_args.get("tau_schedule", "fixed"))
    if schedule == "linear":
        return (
            1,
            float(input_args.get("tau_start", input_args.get("tau", 1.0))),
            float(input_args.get("tau_end", input_args.get("tau", 1.0))),
        )
    tau = float(input_args.get("tau", 1.0))
    return (0, tau, tau)


def _load_params(run_dir: Path) -> dict:
    params_path = run_dir / "params.json"
    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_run_dirs(base_dir: Path, algo: str, dataset: str | None) -> list[Path]:
    run_dirs: list[Path] = []
    for root, _, files in os.walk(base_dir):
        if "params.json" not in files:
            continue
        run_dir = Path(root)
        snapshot_dir = run_dir / "graph_snapshots"
        if not snapshot_dir.exists():
            continue
        params = _load_params(run_dir)
        input_args = params.get("input_args", {})
        if input_args.get("algo") != algo:
            continue
        if dataset is not None and input_args.get("dataset") != dataset:
            continue
        if not any(snapshot_dir.glob("iter_*.npz")):
            continue
        run_dirs.append(run_dir)
    return sorted(run_dirs)


def _load_eval_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "graph_snapshots" / "eval_metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


def _reconstruct_exact_adj(logits: np.ndarray, topk: int) -> np.ndarray:
    n_agent = logits.shape[0]
    topk = max(1, min(int(topk), n_agent))
    indices = np.argsort(-logits, axis=-1)[:, :topk]
    exact = np.zeros_like(logits, dtype=np.int8)
    rows = np.arange(n_agent)[:, None]
    exact[rows, indices] = 1
    return exact


def _has_topk_tie(logits: np.ndarray, topk: int, tol: float = 1e-8) -> bool:
    if topk <= 0 or topk >= logits.shape[1]:
        return False
    sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
    kth = sorted_logits[:, topk - 1]
    next_val = sorted_logits[:, topk]
    return bool(np.any(np.abs(kth - next_val) <= tol))


def _soft_entropy(logits: np.ndarray, tau: float) -> float:
    tau = max(float(tau), 1e-8)
    scaled = logits / tau
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(axis=-1, keepdims=True)
    return float((-(probs * np.log(probs + 1e-12)).sum(axis=-1)).mean())


def _margin(logits: np.ndarray, topk: int) -> float:
    sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
    if topk >= logits.shape[1]:
        return float((sorted_logits[:, 0] - sorted_logits[:, -1]).mean())
    return float((sorted_logits[:, topk - 1] - sorted_logits[:, topk]).mean())


def _normalized_hamming(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    denom = max(a.shape[0] * max(a.shape[1] - 1, 1), 1)
    return float((a != b).sum() / denom)


def _compute_t_conv(exacts: list[np.ndarray], iters: list[int], stability_window: int, fallback_iter: int) -> int:
    if not exacts:
        return int(fallback_iter)
    for idx in range(len(exacts)):
        end = idx + stability_window
        if end >= len(exacts):
            break
        anchor = exacts[idx]
        if all(np.array_equal(exacts[j], anchor) for j in range(idx + 1, end + 1)):
            return int(iters[idx])
    return int(fallback_iter)


def _build_checkpoint_records(
    run_dir: Path,
    verify_exact: bool,
    fallback_iter: int,
) -> tuple[pd.DataFrame, dict]:
    params = _load_params(run_dir)
    input_args = params.get("input_args", {})
    dataset = str(input_args.get("dataset", "unknown"))
    seed = int(input_args.get("seed", -1))
    setting_label = _setting_label(input_args)
    schedule_order, tau_sort, tau_sort_2 = _setting_sort_key(input_args)
    metrics_csv = run_dir / "graph_snapshots" / "eval_metrics.csv"
    eval_metrics = _load_eval_metrics(run_dir).set_index("iter", drop=False) if metrics_csv.exists() else pd.DataFrame()

    snapshot_files = sorted((run_dir / "graph_snapshots").glob("iter_*.npz"))
    records: list[dict] = []
    exacts: list[np.ndarray] = []
    iters: list[int] = []
    prev_exact = None

    for snapshot_file in snapshot_files:
        with np.load(snapshot_file) as data:
            iter_idx = int(np.asarray(data["iter"]).reshape(-1)[0])
            tau = float(np.asarray(data["tau"]).reshape(-1)[0])
            topk = int(np.asarray(data["topk"]).reshape(-1)[0])
            global_step = int(np.asarray(data["global_step"]).reshape(-1)[0])
            logits = np.asarray(data["edge_logits"], dtype=np.float32)
            exact_adj = np.asarray(data["exact_adj"], dtype=np.int8)

        reconstructed = _reconstruct_exact_adj(logits, topk)
        if verify_exact and not np.array_equal(reconstructed, exact_adj):
            if not _has_topk_tie(logits, topk):
                raise ValueError(
                    f"Exact graph mismatch without top-k tie in {snapshot_file}. "
                    f"Saved exact graph no longer matches the logits ordering."
                )

        offdiag = exact_adj.copy()
        np.fill_diagonal(offdiag, 0)
        entropy = _soft_entropy(logits, tau)
        margin = _margin(logits, topk)
        edge_count = float(offdiag.sum())
        density = float(edge_count / max(logits.shape[0] * max(logits.shape[0] - 1, 1), 1))
        delta_exact = 0.0 if prev_exact is None else _normalized_hamming(offdiag, prev_exact)
        prev_exact = offdiag

        record = {
            "run_dir": str(run_dir),
            "dataset": dataset,
            "seed": seed,
            "iter": iter_idx,
            "global_step": global_step,
            "setting_label": setting_label,
            "schedule_order": schedule_order,
            "tau_sort": tau_sort,
            "tau_sort_2": tau_sort_2,
            "graph_tau": tau,
            "graph_margin": margin,
            "graph_entropy_soft": entropy,
            "graph_exact_edge_count": edge_count,
            "graph_exact_density": density,
            "graph_delta_exact": delta_exact,
            "graph_snapshot_file": snapshot_file.name,
            "test_episode_reward": np.nan,
            "test_episode_len": np.nan,
        }
        if not eval_metrics.empty and iter_idx in eval_metrics.index:
            eval_row = eval_metrics.loc[iter_idx]
            if isinstance(eval_row, pd.DataFrame):
                eval_row = eval_row.iloc[-1]
            for key in ("test_episode_reward", "test_episode_len"):
                if key in eval_row:
                    record[key] = float(eval_row[key])
        records.append(record)
        exacts.append(offdiag)
        iters.append(iter_idx)

    if not records:
        raise ValueError(f"No graph snapshots found under {run_dir}")

    checkpoint_df = pd.DataFrame.from_records(records).sort_values("iter").reset_index(drop=True)
    summary = {
        "run_dir": str(run_dir),
        "dataset": dataset,
        "seed": seed,
        "setting_label": setting_label,
        "schedule_order": schedule_order,
        "tau_sort": tau_sort,
        "tau_sort_2": tau_sort_2,
        "t_conv": _compute_t_conv(exacts, iters, stability_window=5, fallback_iter=fallback_iter),
        "tail_volatility": float((checkpoint_df.loc[checkpoint_df["iter"] >= 2500, "graph_delta_exact"] > 0).mean())
        if (checkpoint_df["iter"] >= 2500).any() else np.nan,
        "final_graph_margin": float(checkpoint_df.iloc[-1]["graph_margin"]),
        "final_graph_exact_density": float(checkpoint_df.iloc[-1]["graph_exact_density"]),
        "final_graph_tau": float(checkpoint_df.iloc[-1]["graph_tau"]),
        "final_test_episode_reward": float(checkpoint_df.iloc[-1]["test_episode_reward"])
        if "test_episode_reward" in checkpoint_df.columns and not pd.isna(checkpoint_df.iloc[-1]["test_episode_reward"]) else np.nan,
    }
    return checkpoint_df, summary


def _aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "final_test_episode_reward",
        "t_conv",
        "tail_volatility",
        "final_graph_margin",
        "final_graph_exact_density",
        "final_graph_tau",
    ]
    grouped = summary_df.groupby(
        ["dataset", "setting_label", "schedule_order", "tau_sort", "tau_sort_2"],
        as_index=False,
    )[numeric_cols].agg(["mean", "std"])
    grouped.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped.sort_values(
        ["dataset", "schedule_order", "tau_sort", "tau_sort_2", "setting_label"]
    ).reset_index(drop=True)


def _format_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in agg_df.iterrows():
        rows.append({
            "dataset": row["dataset"],
            "setting": row["setting_label"],
            "final_test_episode_reward": f"{row['final_test_episode_reward_mean']:.3f} +/- {0.0 if pd.isna(row['final_test_episode_reward_std']) else row['final_test_episode_reward_std']:.3f}",
            "t_conv": f"{row['t_conv_mean']:.1f} +/- {0.0 if pd.isna(row['t_conv_std']) else row['t_conv_std']:.1f}",
            "tail_volatility": f"{row['tail_volatility_mean']:.3f} +/- {0.0 if pd.isna(row['tail_volatility_std']) else row['tail_volatility_std']:.3f}",
            "final_graph_margin": f"{row['final_graph_margin_mean']:.4f} +/- {0.0 if pd.isna(row['final_graph_margin_std']) else row['final_graph_margin_std']:.4f}",
            "final_graph_exact_density": f"{row['final_graph_exact_density_mean']:.4f} +/- {0.0 if pd.isna(row['final_graph_exact_density_std']) else row['final_graph_exact_density_std']:.4f}",
        })
    return pd.DataFrame(rows)


def _ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    alpha = min(max(float(alpha), 1e-6), 1.0)
    return pd.Series(values).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float)


def _plot_dataset_curves(
    checkpoint_df: pd.DataFrame,
    dataset: str,
    output_path: Path,
    smooth_alpha: float,
) -> None:
    dataset_df = checkpoint_df[checkpoint_df["dataset"] == dataset].copy()
    if dataset_df.empty:
        return

    metric_specs = [
        ("test_episode_reward", "Test Episode Reward"),
        ("graph_delta_exact", "Communication Graph Change Rate"),
    ]
    order_df = dataset_df[
        ["setting_label", "schedule_order", "tau_sort", "tau_sort_2"]
    ].drop_duplicates().sort_values(["schedule_order", "tau_sort", "tau_sort_2", "setting_label"])
    setting_labels = order_df["setting_label"].tolist()
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(idx % 10) for idx, label in enumerate(setting_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (metric, title) in zip(axes, metric_specs):
        for setting in setting_labels:
            metric_df = dataset_df[dataset_df["setting_label"] == setting]
            grouped = metric_df.groupby("iter")[metric].agg(["mean", "std"]).reset_index()
            mean_values = grouped["mean"].to_numpy(dtype=float)
            std_values = grouped["std"].fillna(0.0).to_numpy(dtype=float)
            x = grouped["iter"].to_numpy(dtype=float)
            smooth_mean = _ema_smooth(mean_values, smooth_alpha)
            smooth_std = _ema_smooth(std_values, smooth_alpha)

            lower = smooth_mean - smooth_std
            upper = smooth_mean + smooth_std
            if metric == "graph_delta_exact":
                lower = np.maximum(lower, 0.0)

            ax.plot(x, smooth_mean, label=setting, color=colors[setting], linewidth=2.2)
            ax.fill_between(
                x,
                lower,
                upper,
                color=colors[setting],
                alpha=0.18,
            )

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        if metric == "test_episode_reward":
            ax.set_ylabel("Reward")
        else:
            ax.set_ylabel("Change Rate")
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].legend(frameon=False, fontsize=10, loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_tau_summary(agg_df: pd.DataFrame, dataset: str, output_path: Path) -> None:
    dataset_df = agg_df[agg_df["dataset"] == dataset].copy()
    if dataset_df.empty:
        return

    fixed_df = dataset_df[dataset_df["schedule_order"] == 0].copy()
    annealed_df = dataset_df[dataset_df["schedule_order"] != 0].copy()
    if fixed_df.empty and annealed_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    metric_specs = [
        ("t_conv", "Tau vs t_conv", "t_conv"),
        ("final_graph_margin", "Tau vs Final Margin", "Final Margin"),
    ]
    cmap = plt.get_cmap("tab10")
    fixed_color = cmap(0)
    annealed_color = cmap(4)

    for ax, (prefix, title, ylabel) in zip(axes, metric_specs):
        mean_col = f"{prefix}_mean"
        std_col = f"{prefix}_std"

        fixed_x: list[float] = []
        if not fixed_df.empty:
            fixed_x = fixed_df["tau_sort"].to_list()
            y_fixed = fixed_df[mean_col].to_numpy(dtype=float)
            yerr_fixed = fixed_df[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(
                fixed_x,
                y_fixed,
                yerr=yerr_fixed,
                fmt="-o",
                linewidth=2.0,
                capsize=4,
                color=fixed_color,
                label="Fixed tau",
            )
            ax.set_xscale("log")

        if not annealed_df.empty:
            x_anchor = (max(fixed_x) * 1.8) if fixed_x else 10.0
            y_annealed = annealed_df[mean_col].to_numpy(dtype=float)
            yerr_annealed = annealed_df[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(
                np.full_like(y_annealed, x_anchor, dtype=float),
                y_annealed,
                yerr=yerr_annealed,
                fmt="D",
                markersize=7,
                capsize=4,
                linestyle="none",
                color=annealed_color,
                label="Annealed",
            )
            for label, y_value in zip(annealed_df["setting_label"], y_annealed):
                ax.annotate(
                    label,
                    xy=(x_anchor, y_value),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                )
            if fixed_x:
                xticks = fixed_x + [x_anchor]
                xticklabels = [_fmt_float(x) for x in fixed_x] + ["annealed"]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)

        ax.set_title(title)
        ax.set_xlabel("Temperature Setting")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MetaComm tau-sensitivity graph snapshots.")
    parser.add_argument("--run_dirs", nargs="*", default=None, help="Explicit run directories to analyze.")
    parser.add_argument("--base_dir", type=Path, default=None, help="Scan this directory recursively for run folders.")
    parser.add_argument("--algo", type=str, default="MetaComm", help="Only analyze runs whose params.json matches this algorithm.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter when scanning base_dir.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for CSV summaries and plots.")
    parser.add_argument("--fallback_iter", type=int, default=3000, help="Fallback convergence iteration if no stable window is found.")
    parser.add_argument("--curve_smooth_alpha", type=float, default=0.25, help="EMA smoothing factor for plotted mean/std curves.")
    parser.add_argument("--skip_verify_exact", action="store_true", help="Skip exact-graph consistency checks against the saved logits.")
    args = parser.parse_args()

    run_dirs = [Path(run_dir) for run_dir in (args.run_dirs or [])]
    if args.base_dir is not None:
        run_dirs.extend(_find_run_dirs(args.base_dir, algo=args.algo, dataset=args.dataset))
    run_dirs = sorted({run_dir.resolve() for run_dir in run_dirs})
    if not run_dirs:
        raise ValueError("No run directories were provided or discovered.")

    checkpoint_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for run_dir in run_dirs:
        checkpoint_df, summary = _build_checkpoint_records(
            run_dir,
            verify_exact=not args.skip_verify_exact,
            fallback_iter=args.fallback_iter,
        )
        checkpoint_frames.append(checkpoint_df)
        summary_rows.append(summary)

    checkpoint_df = pd.concat(checkpoint_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "schedule_order", "tau_sort", "tau_sort_2", "seed"]
    ).reset_index(drop=True)
    agg_df = _aggregate_summary(summary_df)
    table_df = _format_table(agg_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_df.to_csv(args.output_dir / "tau_checkpoint_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "tau_summary_per_run.csv", index=False)
    agg_df.to_csv(args.output_dir / "tau_summary_agg.csv", index=False)
    table_df.to_csv(args.output_dir / "tau_summary_table.csv", index=False)

    for dataset in checkpoint_df["dataset"].drop_duplicates():
        _plot_dataset_curves(
            checkpoint_df,
            dataset=dataset,
            output_path=args.output_dir / f"tau_sensitivity_curves_{dataset}.png",
            smooth_alpha=args.curve_smooth_alpha,
        )
        _plot_tau_summary(
            agg_df,
            dataset=dataset,
            output_path=args.output_dir / f"tau_vs_tconv_margin_{dataset}.png",
        )

    print(f"Saved tau checkpoint metrics to {args.output_dir / 'tau_checkpoint_metrics.csv'}")
    print(f"Saved tau summary table to {args.output_dir / 'tau_summary_table.csv'}")


if __name__ == "__main__":
    main()
