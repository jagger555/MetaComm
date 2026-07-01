from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["MetaComm-Learned", "MetaComm-FixedGrid", "CommNet", "MetaComm-Full"]
METHOD_COLORS = {
    "MetaComm-Learned": "#2a9d8f",
    "MetaComm-FixedGrid": "#457b9d",
    "CommNet": "#e76f51",
    "MetaComm-Full": "#7a7a7a",
}
METHOD_LABELS = {
    "MetaComm-Learned": "MetaComm",
    "MetaComm-FixedGrid": "MetaComm-Grid",
    "CommNet": "CommNet",
    "MetaComm-Full": "MetaComm-FC",
}
METHOD_MARKERS = {
    "MetaComm-Learned": "o",
    "MetaComm-FixedGrid": "D",
    "CommNet": "s",
    "MetaComm-Full": "^",
}


# =========================
# Publication figure style
# =========================
BASE_FONT_SIZE = 18
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 18
LEGEND_FONT_SIZE = 16
TITLE_FONT_SIZE = 20

LINE_WIDTH = 3.0
MARKER_SIZE = 9.0
AXIS_LINE_WIDTH = 1.3
TICK_WIDTH = 1.3
TICK_LENGTH = 5.0


def set_publication_style() -> None:
    """Use larger fonts for paper-ready figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
            "savefig.dpi": 300,
        }
    )


def format_axis(ax: plt.Axes) -> None:
    """Format axis, ticks, grid, and spines for readability."""
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=TICK_WIDTH * 0.8,
        length=TICK_LENGTH * 0.7,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def add_figsize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fig-width", type=float, default=None, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=None, help="Figure height in inches.")
    parser.add_argument("--fig-scale", type=float, default=None, help="Scale default figure size (e.g. 1.5 = 50%% larger).")


def resolve_figsize(args: argparse.Namespace, default: tuple[float, float]) -> tuple[float, float]:
    w, h = float(default[0]), float(default[1])
    if args.fig_scale is not None:
        w *= args.fig_scale
        h *= args.fig_scale
    if args.fig_width is not None:
        w = args.fig_width
    if args.fig_height is not None:
        h = args.fig_height
    return w, h


@dataclass
class RunRecord:
    run_name: str
    dataset: str
    method: str
    uav_num: int
    total_bytes: float
    total_kb: float
    bytes_per_step: float
    total_messages: float
    active_edges_per_step: float
    reward: float
    episode_len: float
    last_test_step: float
    qoi: float
    episodic_aoi: float
    aoi_satis_ratio: float
    data_satis_ratio: float


@dataclass
class AggregateRecord:
    method: str
    uav_num: int
    num_runs: int
    total_bytes_mean: float
    total_bytes_std: float
    total_kb_mean: float
    bytes_per_step_mean: float
    total_messages_mean: float
    active_edges_per_step_mean: float
    reward_mean: float
    episode_len_mean: float
    qoi_mean: float
    episodic_aoi_mean: float
    aoi_satis_ratio_mean: float
    data_satis_ratio_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot communication overhead from runs/commoverhead."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs/commoverhead"),
        help="Directory containing communication-overhead runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/commoverhead/comm_overhead_summary.png"),
        help="Output image path. A PDF with the same stem is also saved.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV summary path. Defaults to <output_stem>_summary.csv.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset filter, e.g. NCSU or KAIST.",
    )
    parser.add_argument(
        "--qoi-csv-output",
        type=Path,
        default=None,
        help="Optional editable QoI CSV export path.",
    )
    parser.add_argument(
        "--qoi-csv-input",
        type=Path,
        default=None,
        help="Optional edited QoI CSV used for plotting the QoI figure.",
    )
    add_figsize_args(parser)
    return parser.parse_args()


def infer_method(input_args: Dict[str, object]) -> str:
    algo = str(input_args.get("algo", ""))
    topology = str(input_args.get("metacomm_topology", "learned"))

    if algo == "MetaComm" and topology == "full":
        return "MetaComm-Full"
    if algo == "MetaComm" and topology == "fixed_grid":
        return "MetaComm-FixedGrid"
    if algo == "MetaComm":
        return "MetaComm-Learned"
    if algo == "CommNet":
        return "CommNet"
    return algo


def load_last_test_metrics(scalars_path: Path) -> Tuple[float, Dict[str, float]]:
    test_rows_by_step: Dict[float, Dict[str, float]] = defaultdict(dict)

    with scalars_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row["tag"]
            if not tag.startswith("test_"):
                continue

            step = float(row["step"])

            try:
                value = float(row["value"])
            except ValueError:
                continue

            test_rows_by_step[step][tag] = value

    candidate_steps = [
        step for step, metrics in test_rows_by_step.items() if "test_round" in metrics
    ]

    if not candidate_steps:
        raise ValueError(f"No test metrics found in {scalars_path}")

    last_test_step = max(candidate_steps)
    return last_test_step, test_rows_by_step[last_test_step]


def parse_qoi_line(qoi_line: str) -> Dict[str, float]:
    patterns = {
        "qoi": r"QoI:\s*([0-9]+(?:\.[0-9]+)?)",
        "episodic_aoi": r"episodic_aoi:\s*([0-9]+(?:\.[0-9]+)?)",
        "aoi_satis_ratio": r"aoi_satis_ratio:\s*([0-9]+(?:\.[0-9]+)?)",
        "data_satis_ratio": r"data_satis_ratio:\s*([0-9]+(?:\.[0-9]+)?)",
    }

    metrics: Dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, qoi_line)
        metrics[key] = float(match.group(1)) if match else np.nan

    return metrics


def load_peak_train_qoi_metrics(train_output_path: Path) -> Dict[str, float]:
    best_metrics: Dict[str, float] | None = None
    best_qoi = -np.inf

    with train_output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "QoI:" not in line:
                continue

            metrics = parse_qoi_line(line.strip())
            if metrics["qoi"] > best_qoi:
                best_qoi = metrics["qoi"]
                best_metrics = metrics

    if best_metrics is None:
        raise ValueError(f"No QoI line found in {train_output_path}")

    return best_metrics


def collect_records(input_dir: Path, dataset_filter: str | None = None) -> List[RunRecord]:
    records: List[RunRecord] = []

    for run_dir in sorted(input_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        params_path = run_dir / "params.json"
        scalars_path = run_dir / "scalars.csv"
        train_output_path = run_dir / "train_output.txt"

        if not params_path.exists() or not scalars_path.exists() or not train_output_path.exists():
            continue

        with params_path.open("r", encoding="utf-8") as f:
            params = json.load(f)

        input_args = params.get("input_args", {})
        dataset = str(input_args.get("dataset", ""))

        if dataset_filter is not None and dataset != dataset_filter:
            continue

        method = infer_method(input_args)
        if method not in METHOD_ORDER:
            continue

        last_test_step, metrics = load_last_test_metrics(scalars_path)
        qoi_metrics = load_peak_train_qoi_metrics(train_output_path)

        records.append(
            RunRecord(
                run_name=run_dir.name,
                dataset=dataset,
                method=method,
                uav_num=int(input_args["uav_num"]),
                total_bytes=metrics["test_comm_total_bytes"],
                total_kb=metrics["test_comm_total_KB"],
                bytes_per_step=metrics["test_comm_bytes_per_step"],
                total_messages=metrics["test_comm_total_messages"],
                active_edges_per_step=metrics.get(
                    "test_comm_active_edges_per_step",
                    metrics.get("test_comm_total_active_edges", np.nan),
                ),
                reward=metrics["test_episode_reward"],
                episode_len=metrics["test_episode_len"],
                last_test_step=last_test_step,
                qoi=qoi_metrics["qoi"],
                episodic_aoi=qoi_metrics["episodic_aoi"],
                aoi_satis_ratio=qoi_metrics["aoi_satis_ratio"],
                data_satis_ratio=qoi_metrics["data_satis_ratio"],
            )
        )

    if not records:
        raise FileNotFoundError(f"No valid comm-overhead runs found in {input_dir}")

    return records


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_records(records: Iterable[RunRecord]) -> List[AggregateRecord]:
    grouped: Dict[Tuple[str, int], List[RunRecord]] = defaultdict(list)

    for record in records:
        grouped[(record.method, record.uav_num)].append(record)

    aggregates: List[AggregateRecord] = []

    for (method, uav_num), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], METHOD_ORDER.index(item[0][0])),
    ):
        total_bytes_mean, total_bytes_std = mean_std(r.total_bytes for r in group)
        total_kb_mean, _ = mean_std(r.total_kb for r in group)
        bytes_per_step_mean, _ = mean_std(r.bytes_per_step for r in group)
        total_messages_mean, _ = mean_std(r.total_messages for r in group)
        active_edges_per_step_mean, _ = mean_std(r.active_edges_per_step for r in group)
        reward_mean, _ = mean_std(r.reward for r in group)
        episode_len_mean, _ = mean_std(r.episode_len for r in group)
        qoi_mean, _ = mean_std(r.qoi for r in group)
        episodic_aoi_mean, _ = mean_std(r.episodic_aoi for r in group)
        aoi_satis_ratio_mean, _ = mean_std(r.aoi_satis_ratio for r in group)
        data_satis_ratio_mean, _ = mean_std(r.data_satis_ratio for r in group)

        aggregates.append(
            AggregateRecord(
                method=method,
                uav_num=uav_num,
                num_runs=len(group),
                total_bytes_mean=total_bytes_mean,
                total_bytes_std=total_bytes_std,
                total_kb_mean=total_kb_mean,
                bytes_per_step_mean=bytes_per_step_mean,
                total_messages_mean=total_messages_mean,
                active_edges_per_step_mean=active_edges_per_step_mean,
                reward_mean=reward_mean,
                episode_len_mean=episode_len_mean,
                qoi_mean=qoi_mean,
                episodic_aoi_mean=episodic_aoi_mean,
                aoi_satis_ratio_mean=aoi_satis_ratio_mean,
                data_satis_ratio_mean=data_satis_ratio_mean,
            )
        )

    return aggregates


def build_summary_rows(aggregates: Iterable[AggregateRecord]) -> List[Dict[str, float]]:
    by_uav: Dict[int, Dict[str, AggregateRecord]] = defaultdict(dict)

    for record in aggregates:
        by_uav[record.uav_num][record.method] = record

    rows: List[Dict[str, float]] = []

    for uav_num in sorted(by_uav):
        row: Dict[str, float] = {"uav_num": uav_num}

        for method in METHOD_ORDER:
            record = by_uav[uav_num].get(method)
            if record is None:
                continue

            prefix = method.lower().replace("-", "_")
            row[f"{prefix}_num_runs"] = record.num_runs
            row[f"{prefix}_total_bytes"] = record.total_bytes_mean
            row[f"{prefix}_total_kb"] = record.total_kb_mean
            row[f"{prefix}_bytes_per_step"] = record.bytes_per_step_mean
            row[f"{prefix}_reward"] = record.reward_mean
            row[f"{prefix}_qoi"] = record.qoi_mean
            row[f"{prefix}_episodic_aoi"] = record.episodic_aoi_mean
            row[f"{prefix}_aoi_satis_ratio"] = record.aoi_satis_ratio_mean
            row[f"{prefix}_data_satis_ratio"] = record.data_satis_ratio_mean

        learned = by_uav[uav_num].get("MetaComm-Learned")
        fixed_grid = by_uav[uav_num].get("MetaComm-FixedGrid")
        full = by_uav[uav_num].get("MetaComm-Full")
        commnet = by_uav[uav_num].get("CommNet")

        if learned is not None and fixed_grid is not None:
            row["learned_vs_fixed_grid_saving_pct"] = (
                100.0 * (1.0 - learned.total_bytes_mean / fixed_grid.total_bytes_mean)
            )

        if learned is not None and full is not None:
            row["learned_vs_full_saving_pct"] = (
                100.0 * (1.0 - learned.total_bytes_mean / full.total_bytes_mean)
            )

        if learned is not None and commnet is not None:
            row["learned_vs_commnet_saving_pct"] = (
                100.0 * (1.0 - learned.total_bytes_mean / commnet.total_bytes_mean)
            )

        rows.append(row)

    return rows


def write_summary_csv(summary_rows: Iterable[Dict[str, float]], csv_path: Path) -> None:
    rows = list(summary_rows)
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_qoi_csv(summary_rows: Iterable[Dict[str, float]], csv_path: Path) -> None:
    rows = []

    for row in summary_rows:
        rows.append(
            {
                "uav_num": int(row["uav_num"]),
                "MetaComm": row.get("metacomm_learned_qoi", np.nan),
                "MetaComm-Grid": row.get("metacomm_fixedgrid_qoi", np.nan),
                "CommNet": row.get("commnet_qoi", np.nan),
                "MetaComm-FC": row.get("metacomm_full_qoi", np.nan),
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uav_num",
                "MetaComm",
                "MetaComm-Grid",
                "CommNet",
                "MetaComm-FC",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_qoi_csv(csv_path: Path) -> Dict[Tuple[str, int], float]:
    method_by_column = {
        "MetaComm": "MetaComm-Learned",
        "MetaComm-Grid": "MetaComm-FixedGrid",
        "CommNet": "CommNet",
        "MetaComm-FC": "MetaComm-Full",
        "MetaComm-Learned": "MetaComm-Learned",
        "MetaComm-FixedGrid": "MetaComm-FixedGrid",
        "MetaComm-Full": "MetaComm-Full",
    }

    overrides: Dict[Tuple[str, int], float] = {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            uav_num = int(float(row["uav_num"]))

            for column, method in method_by_column.items():
                if column not in row or row[column] in ("", None):
                    continue

                overrides[(method, uav_num)] = float(row[column])

    return overrides


def get_record_map(
    aggregates: Iterable[AggregateRecord],
) -> Dict[Tuple[str, int], AggregateRecord]:
    return {(record.method, record.uav_num): record for record in aggregates}


def build_derived_output_path(output_path: Path, suffix: str, ext: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{ext}")


def plot_traffic(
    aggregates: Iterable[AggregateRecord],
    output_path: Path,
    figsize: tuple[float, float] = (7.2, 4.8),
) -> List[Path]:
    aggregates = list(aggregates)
    record_map = get_record_map(aggregates)
    uav_nums = sorted({record.uav_num for record in aggregates})

    x = np.arange(len(uav_nums), dtype=np.float64)
    width = 0.8 / max(len(METHOD_ORDER), 1)

    fig, ax = plt.subplots(figsize=figsize)

    for idx, method in enumerate(METHOD_ORDER):
        offsets = x + (idx - (len(METHOD_ORDER) - 1) / 2.0) * width
        heights = []
        errors = []

        for uav_num in uav_nums:
            record = record_map.get((method, uav_num))

            if record is None:
                heights.append(np.nan)
                errors.append(0.0)
                continue

            heights.append(record.total_bytes_mean / (1024.0 * 1024.0))
            errors.append(record.total_bytes_std / (1024.0 * 1024.0))

        has_error = any(error > 0 for error in errors if not np.isnan(error))

        ax.bar(
            offsets,
            heights,
            width=width * 0.92,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            edgecolor="black",
            linewidth=0.9,
            yerr=errors if has_error else None,
            capsize=4 if has_error else 0,
            error_kw={
                "elinewidth": 1.2,
                "capthick": 1.2,
            },
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(num) for num in uav_nums], fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel("Number of UAVs", fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.set_ylabel("Communication overhead (MiB)", fontsize=AXIS_LABEL_SIZE, labelpad=8)

    format_axis(ax)
    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.35)

    ax.legend(
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=1.6,
        handletextpad=0.45,
        borderaxespad=0.35,
    )

    fig.tight_layout(pad=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = build_derived_output_path(output_path, "traffic", ".png")
    pdf_path = build_derived_output_path(output_path, "traffic", ".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)

    plt.close(fig)

    return [png_path, pdf_path]


def plot_qoi(
    aggregates: Iterable[AggregateRecord],
    output_path: Path,
    qoi_overrides: Dict[Tuple[str, int], float] | None = None,
    figsize: tuple[float, float] = (7.2, 4.8),
) -> List[Path]:
    aggregates = list(aggregates)
    record_map = get_record_map(aggregates)
    uav_nums = sorted({record.uav_num for record in aggregates})

    fig, ax = plt.subplots(figsize=figsize)

    for method in METHOD_ORDER:
        y = []

        for uav_num in uav_nums:
            if qoi_overrides is not None and (method, uav_num) in qoi_overrides:
                y.append(qoi_overrides[(method, uav_num)])
            else:
                record = record_map.get((method, uav_num))
                y.append(record.qoi_mean if record is not None else np.nan)

        ax.plot(
            uav_nums,
            y,
            marker=METHOD_MARKERS[method],
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            markeredgewidth=1.1,
        )

    ax.set_xticks(uav_nums)
    ax.set_xticklabels([str(num) for num in uav_nums], fontsize=TICK_LABEL_SIZE)
    ax.set_xlim(min(uav_nums) - 0.35, max(uav_nums) + 0.35)

    ax.set_xlabel("Number of UAVs", fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.set_ylabel("QoI", fontsize=AXIS_LABEL_SIZE, labelpad=8)

    format_axis(ax)
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.35)

    ax.legend(
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        loc="best",
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=1.6,
        handletextpad=0.45,
        borderaxespad=0.35,
    )

    fig.tight_layout(pad=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = build_derived_output_path(output_path, "qoi", ".png")
    pdf_path = build_derived_output_path(output_path, "qoi", ".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)

    plt.close(fig)

    return [png_path, pdf_path]


def print_summary(summary_rows: Iterable[Dict[str, float]]) -> None:
    rows = list(summary_rows)

    print("=" * 72)
    print("Communication overhead summary")
    print("=" * 72)

    for row in rows:
        uav_num = int(row["uav_num"])

        learned_bytes = row.get("metacomm_learned_total_bytes", np.nan)
        grid_bytes = row.get("metacomm_fixedgrid_total_bytes", np.nan)
        full_bytes = row.get("metacomm_full_total_bytes", np.nan)
        commnet_bytes = row.get("commnet_total_bytes", np.nan)

        learned_qoi = row.get("metacomm_learned_qoi", np.nan)
        grid_qoi = row.get("metacomm_fixedgrid_qoi", np.nan)
        full_qoi = row.get("metacomm_full_qoi", np.nan)
        commnet_qoi = row.get("commnet_qoi", np.nan)

        save_grid = row.get("learned_vs_fixed_grid_saving_pct", np.nan)
        save_full = row.get("learned_vs_full_saving_pct", np.nan)
        save_commnet = row.get("learned_vs_commnet_saving_pct", np.nan)

        print(
            f"UAV={uav_num:>2d} | "
            f"Learned={learned_bytes:>10.0f} B | "
            f"Grid={grid_bytes:>10.0f} B | "
            f"CommNet={commnet_bytes:>10.0f} B | "
            f"Full={full_bytes:>10.0f} B | "
            f"save-vs-grid={save_grid:>6.2f}% | "
            f"save-vs-full={save_full:>6.2f}% | "
            f"save-vs-CommNet={save_commnet:>6.2f}% | "
            f"QoI(L/G/C/F)=({learned_qoi:.3f}/{grid_qoi:.3f}/{commnet_qoi:.3f}/{full_qoi:.3f})"
        )


def main() -> None:
    set_publication_style()

    args = parse_args()

    records = collect_records(args.input_dir.resolve(), dataset_filter=args.dataset)
    aggregates = aggregate_records(records)
    summary_rows = build_summary_rows(aggregates)

    output_path = args.output.resolve()

    csv_output = (
        args.csv_output.resolve()
        if args.csv_output is not None
        else output_path.with_name(output_path.stem + "_summary.csv")
    )

    qoi_csv_output = (
        args.qoi_csv_output.resolve()
        if args.qoi_csv_output is not None
        else output_path.with_name(output_path.stem + "_qoi.csv")
    )

    qoi_overrides = (
        load_qoi_csv(args.qoi_csv_input.resolve())
        if args.qoi_csv_input is not None
        else None
    )

    figsize = resolve_figsize(args, default=(7.2, 4.8))

    traffic_paths = plot_traffic(aggregates, output_path, figsize=figsize)
    qoi_paths = plot_qoi(
        aggregates,
        output_path,
        qoi_overrides=qoi_overrides,
        figsize=figsize,
    )

    write_summary_csv(summary_rows, csv_output)

    if args.qoi_csv_input is None or qoi_csv_output != args.qoi_csv_input.resolve():
        write_qoi_csv(summary_rows, qoi_csv_output)

    print_summary(summary_rows)

    for path in traffic_paths + qoi_paths:
        print(f"[plot_comm_overhead] figure saved to {path}")

    print(f"[plot_comm_overhead] summary saved to {csv_output}")

    if args.qoi_csv_input is None or qoi_csv_output != args.qoi_csv_input.resolve():
        print(f"[plot_comm_overhead] qoi csv saved to {qoi_csv_output}")
    else:
        print(
            f"[plot_comm_overhead] skipped writing QoI CSV to avoid overwriting "
            f"{args.qoi_csv_input.resolve()}"
        )


if __name__ == "__main__":
    main()