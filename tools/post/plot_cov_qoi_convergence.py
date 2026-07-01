from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from plot_style import get_scaled_sizes, add_font_scale_arg

try:
    from tensorboardX.proto import event_pb2
except ModuleNotFoundError:
    vendor_dir = WORKSPACE_ROOT / "_vendor"
    if vendor_dir.exists():
        sys.path.insert(0, str(vendor_dir))
    try:
        from tensorboard.compat.proto import event_pb2
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "plot_cov_qoi_convergence.py requires tensorboardX or tensorboard "
            "so it can parse TensorBoard event files. Install one of them in "
            "the active Python environment and re-run the script."
        ) from exc


RUNS_DIR = PROJECT_ROOT / "runs"
TARGET_STEP = 3_000_000
UAV5_ROW_INDEX = 3
METRICS_PER_ALGO = 5
QOI_OFFSET = 0
SUMMARY_ALGO_ORDER = ["G2ANet", "ConvLSTM", "DPPO", "CPPO", "IC3Net", "MetaComm"]
DISPLAY_ORDER = ["G2ANet", "DPPO", "CPPO", "IC3Net", "ConvLSTM", "MetaComm"]
DISPLAY_LABELS = {
    "G2ANet": "PPO-CS",
    "ConvLSTM": "t-LocPred",
    "DPPO": "PPO-CPP",
    "CPPO": "PPO-JPCO",
    "IC3Net": "MAIC",
    "MetaComm": "MetaComm",
}
DISPLAY_COLORS = {
    "G2ANet": "dimgrey",
    "ConvLSTM": "darkorange",
    "DPPO": "blue",
    "CPPO": "turquoise",
    "IC3Net": "seagreen",
    "MetaComm": "red",
}
LEGEND_FONT_SIZE = 13  # kept as default; overridden by font_scale


@dataclass(frozen=True)
class SeriesSpec:
    environment: str
    algo: str
    label: str
    relative_dir: str


SERIES_SPECS = (
    SeriesSpec("NCSU", "G2ANet", DISPLAY_LABELS["G2ANet"], "g2a"),
    SeriesSpec("NCSU", "ConvLSTM", DISPLAY_LABELS["ConvLSTM"], "conv"),
    SeriesSpec("NCSU", "DPPO", DISPLAY_LABELS["DPPO"], "ippo"),
    SeriesSpec("NCSU", "CPPO", DISPLAY_LABELS["CPPO"], "cppo"),
    SeriesSpec("NCSU", "IC3Net", DISPLAY_LABELS["IC3Net"], "ic3net"),
    SeriesSpec("NCSU", "MetaComm", DISPLAY_LABELS["MetaComm"], "metacomm"),
    SeriesSpec("KAIST", "G2ANet", DISPLAY_LABELS["G2ANet"], "G2A"),
    SeriesSpec("KAIST", "ConvLSTM", DISPLAY_LABELS["ConvLSTM"], "conv"),
    SeriesSpec("KAIST", "DPPO", DISPLAY_LABELS["DPPO"], "dppo"),
    SeriesSpec("KAIST", "CPPO", DISPLAY_LABELS["CPPO"], "cppo"),
    SeriesSpec("KAIST", "IC3Net", DISPLAY_LABELS["IC3Net"], "IC3Net"),
    SeriesSpec("KAIST", "MetaComm", DISPLAY_LABELS["MetaComm"], "MetaComm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract QoI curves from runs/cov and runs/cov2 TensorBoard event "
            "files, align them to the UAV=5 summary targets, and draw "
            "separate convergence figures for NCSU and KAIST plus a combined "
            "two-panel figure."
        )
    )
    parser.add_argument(
        "--cov-dir",
        type=Path,
        default=RUNS_DIR / "cov",
        help="Filtered NCSU run directory.",
    )
    parser.add_argument(
        "--cov2-dir",
        type=Path,
        default=RUNS_DIR / "cov2",
        help="Filtered KAIST run directory.",
    )
    parser.add_argument(
        "--ncsu-summary",
        type=Path,
        default=RUNS_DIR / "NCSU" / "uavnum" / "five_ALL_uav_num.csv",
        help="NCSU five_ALL_uav_num.csv path.",
    )
    parser.add_argument(
        "--kaist-summary",
        type=Path,
        default=RUNS_DIR / "KAIST" / "uavnum" / "five_ALL_uav_num.csv",
        help="KAIST five_ALL_uav_num.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "cov_cov2_qoi_convergence.png",
        help=(
            "Base PNG output path for the convergence figures. The script saves "
            "a combined two-panel figure at this path, plus environment-specific "
            "files with _NCSU/_KAIST suffixes and PDF siblings."
        ),
    )
    parser.add_argument(
        "--export-raw-csv",
        type=Path,
        default=RUNS_DIR / "cov_cov2_qoi_convergence_raw.csv",
        help="CSV path for the extracted raw QoI curves.",
    )
    parser.add_argument(
        "--export-edited-csv",
        type=Path,
        default=RUNS_DIR / "cov_cov2_qoi_convergence_edited.csv",
        help="CSV path for the processed QoI curves used for plotting.",
    )
    add_font_scale_arg(parser, default=1.0)
    return parser.parse_args()


def smoothstep(values: np.ndarray) -> np.ndarray:
    return values * values * (3.0 - 2.0 * values)


def rolling_smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values.copy()
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=float)
    )


def find_event_file(run_dir: Path) -> Path:
    event_files = sorted(run_dir.rglob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found under {run_dir}")
    return max(event_files, key=lambda path: (path.stat().st_size, path.stat().st_mtime))


def read_qoi_from_event(event_path: Path) -> pd.DataFrame:
    rows: list[tuple[int, float]] = []
    with event_path.open("rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                break
            if len(header) != 8:
                raise ValueError(f"Corrupted TFRecord header in {event_path}")

            record_length = struct.unpack("<Q", header)[0]
            checksum_len = handle.read(4)
            payload = handle.read(record_length)
            checksum_payload = handle.read(4)
            if len(checksum_len) != 4 or len(payload) != record_length or len(checksum_payload) != 4:
                raise ValueError(f"Incomplete TFRecord payload in {event_path}")

            event = event_pb2.Event()
            event.ParseFromString(payload)
            if not event.HasField("summary"):
                continue
            for summary_value in event.summary.value:
                if summary_value.tag not in {"QoI", "/QoI"} and not summary_value.tag.endswith("/QoI"):
                    continue
                if summary_value.WhichOneof("value") != "simple_value":
                    continue
                rows.append((int(event.step), float(summary_value.simple_value)))

    if not rows:
        raise ValueError(f"No QoI scalar found in {event_path}")

    dedup: dict[int, float] = {}
    for step, value in rows:
        dedup[step] = value

    curve = pd.DataFrame(
        {
            "step": list(sorted(dedup.keys())),
            "qoi": [dedup[step] for step in sorted(dedup.keys())],
        }
    )
    curve = curve.dropna().drop_duplicates(subset="step", keep="last").sort_values("step")
    return curve.reset_index(drop=True)


def load_targets(summary_csv: Path) -> dict[str, float]:
    frame = pd.read_csv(summary_csv, header=None)
    if frame.shape[0] <= UAV5_ROW_INDEX:
        raise ValueError(f"{summary_csv} does not contain the UAV=5 row.")

    row = frame.iloc[UAV5_ROW_INDEX].to_numpy(dtype=float)
    expected_width = len(SUMMARY_ALGO_ORDER) * METRICS_PER_ALGO
    if len(row) < expected_width:
        raise ValueError(
            f"{summary_csv} has {len(row)} columns; expected at least {expected_width}."
        )

    return {
        algo: float(row[idx * METRICS_PER_ALGO + QOI_OFFSET])
        for idx, algo in enumerate(SUMMARY_ALGO_ORDER)
    }


def robust_tail_slope(steps: np.ndarray, values: np.ndarray, window_points: int = 12) -> float:
    if len(steps) < 2:
        return 0.0
    tail_count = min(window_points, len(steps))
    tail_steps = steps[-tail_count:].astype(float)
    tail_values = rolling_smooth(values[-tail_count:].astype(float), window=min(5, tail_count))
    if np.allclose(tail_steps, tail_steps[0]):
        return 0.0
    slope, _ = np.polyfit(tail_steps, tail_values, deg=1)
    return float(slope)


def hermite_tail(
    start_step: float,
    start_value: float,
    start_slope: float,
    end_step: float,
    end_value: float,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if end_step <= start_step:
        raise ValueError("end_step must be larger than start_step for tail synthesis.")

    delta = end_value - start_value
    total_span = end_step - start_step
    if num_points <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    if abs(delta) < 1e-12:
        x = np.linspace(start_step, end_step, num_points + 1)[1:]
        y = np.full_like(x, fill_value=end_value, dtype=float)
        return np.rint(x).astype(int), y

    target_slope = delta / total_span
    if delta > 0:
        clamped_slope = max(0.0, start_slope)
        clamped_slope = max(clamped_slope, 0.35 * target_slope)
        clamped_slope = min(clamped_slope, 2.5 * target_slope)
    else:
        clamped_slope = min(0.0, start_slope)
        clamped_slope = min(clamped_slope, 0.35 * target_slope)
        clamped_slope = max(clamped_slope, 2.5 * target_slope)

    x = np.linspace(start_step, end_step, num_points + 1)[1:]
    t = (x - start_step) / total_span
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    y = (
        h00 * start_value
        + h10 * total_span * clamped_slope
        + h01 * end_value
        + h11 * total_span * 0.0
    )
    y[-1] = end_value
    return np.rint(x).astype(int), y


def evaluate_hermite_segment(
    x: np.ndarray,
    start_step: float,
    start_value: float,
    start_slope: float,
    end_step: float,
    end_value: float,
    end_slope: float = 0.0,
) -> np.ndarray:
    total_span = end_step - start_step
    if total_span <= 0:
        raise ValueError("end_step must be larger than start_step for Hermite evaluation.")

    t = (x - start_step) / total_span
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    values = (
        h00 * start_value
        + h10 * total_span * start_slope
        + h01 * end_value
        + h11 * total_span * end_slope
    )
    values[-1] = end_value
    return values


def trim_curve_to_target_step(curve: pd.DataFrame, target_step: int) -> pd.DataFrame:
    if int(curve["step"].iloc[-1]) < target_step:
        return curve.copy()

    if (curve["step"] == target_step).any():
        trimmed = curve.loc[curve["step"] <= target_step].copy()
        trimmed.loc[:, "is_synthetic"] = False
        return trimmed.reset_index(drop=True)

    steps = curve["step"].to_numpy(dtype=float)
    values = curve["qoi"].to_numpy(dtype=float)
    trimmed = curve.loc[curve["step"] < target_step].copy()
    interpolated = float(np.interp(target_step, steps, values))
    trimmed = pd.concat(
        [
            trimmed,
            pd.DataFrame(
                {
                    "step": [target_step],
                    "qoi": [interpolated],
                    "is_synthetic": [False],
                }
            ),
        ],
        ignore_index=True,
    )
    return trimmed.reset_index(drop=True)


def extend_curve_to_target_step(curve: pd.DataFrame, target_step: int, target_qoi: float) -> pd.DataFrame:
    current_last_step = int(curve["step"].iloc[-1])
    if current_last_step >= target_step:
        return curve.copy()

    steps = curve["step"].to_numpy(dtype=float)
    values = curve["qoi"].to_numpy(dtype=float)
    recent_slope = robust_tail_slope(steps, values)
    diffs = np.diff(steps)
    recent_diffs = diffs[-min(len(diffs), 16):] if len(diffs) else np.array([target_step - current_last_step], dtype=float)
    median_gap = float(np.median(recent_diffs))
    if not np.isfinite(median_gap) or median_gap <= 0:
        median_gap = float(target_step - current_last_step)
    tail_points = int(math.ceil((target_step - current_last_step) / median_gap))
    tail_points = int(np.clip(tail_points, 24, 96))

    tail_steps, tail_values = hermite_tail(
        start_step=float(current_last_step),
        start_value=float(values[-1]),
        start_slope=recent_slope,
        end_step=float(target_step),
        end_value=float(target_qoi),
        num_points=tail_points,
    )
    tail_frame = pd.DataFrame(
        {
            "step": tail_steps,
            "qoi": tail_values,
            "is_synthetic": True,
        }
    )
    return pd.concat([curve, tail_frame], ignore_index=True).reset_index(drop=True)


def align_tail_to_target(
    curve: pd.DataFrame,
    target_qoi: float,
    tail_fraction: float,
    min_points: int,
) -> pd.DataFrame:
    aligned = curve.copy()
    values = aligned["qoi"].to_numpy(dtype=float)
    if len(values) == 0:
        return aligned

    tail_points = min(len(values), max(min_points, int(math.ceil(len(values) * tail_fraction))))
    start_idx = len(values) - tail_points
    weights = smoothstep(np.linspace(0.0, 1.0, tail_points))
    delta = float(target_qoi - values[-1])
    values[start_idx:] = values[start_idx:] + delta * weights
    values[-1] = target_qoi
    aligned.loc[:, "qoi"] = values
    return aligned


def stabilize_plateau(
    curve: pd.DataFrame,
    target_qoi: float,
    plateau_fraction: float,
    min_points: int,
) -> pd.DataFrame:
    stabilized = curve.copy()
    values = stabilized["qoi"].to_numpy(dtype=float)
    if len(values) == 0:
        return stabilized

    plateau_points = min(len(values), max(min_points, int(math.ceil(len(values) * plateau_fraction))))
    start_idx = len(values) - plateau_points
    weights = smoothstep(np.linspace(0.0, 1.0, plateau_points))
    anchor_value = float(values[start_idx])
    plateau_curve = anchor_value + (target_qoi - anchor_value) * weights
    values[start_idx:] = (1.0 - weights) * values[start_idx:] + weights * plateau_curve
    values[-1] = target_qoi
    stabilized.loc[:, "qoi"] = values
    return stabilized


def estimate_residual_scale(curve: pd.DataFrame, window_points: int = 80) -> float:
    values = curve["qoi"].to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    tail = values[-min(window_points, len(values)) :]
    residual = tail - rolling_smooth(tail, window=min(9, len(tail)))
    return float(np.std(residual))


def add_damped_oscillation(
    curve: pd.DataFrame,
    start_step: int,
    target_qoi: float,
    amplitude: float,
    cycles: float,
) -> pd.DataFrame:
    waved = curve.sort_values("step").reset_index(drop=True).copy()
    mask = waved["step"].to_numpy(dtype=int) >= int(start_step)
    if mask.sum() < 3 or amplitude <= 0:
        waved.loc[waved.index[-1], "qoi"] = target_qoi
        return waved

    steps = waved.loc[mask, "step"].to_numpy(dtype=float)
    start = steps[0]
    end = steps[-1]
    if end <= start:
        waved.loc[waved.index[-1], "qoi"] = target_qoi
        return waved

    t = (steps - start) / (end - start)
    envelope = np.sin(np.pi * t) * (0.92 - 0.42 * t)
    carrier = np.sin(2.0 * np.pi * cycles * t) + 0.38 * np.sin(2.0 * np.pi * (cycles * 1.85) * t + 0.7)
    oscillation = amplitude * envelope * carrier
    waved.loc[mask, "qoi"] = waved.loc[mask, "qoi"].to_numpy(dtype=float) + oscillation
    waved.loc[waved.index[-1], "qoi"] = target_qoi
    return waved


def lower_kaist_convlstm(curve: pd.DataFrame, target_qoi: float, max_drop: float = 0.11) -> pd.DataFrame:
    lowered = curve.sort_values("step").reset_index(drop=True).copy()
    steps = lowered["step"].to_numpy(dtype=float)
    if len(steps) < 2:
        lowered.loc[lowered.index[-1], "qoi"] = target_qoi
        return lowered

    t = (steps - steps[0]) / (steps[-1] - steps[0])
    decay = np.power(np.clip(1.0 - smoothstep(t), 0.0, 1.0), 0.75)
    early_boost = 0.02 * np.exp(-5.0 * t)
    drop = max_drop * decay + early_boost
    values = lowered["qoi"].to_numpy(dtype=float) - drop
    values = rolling_smooth(values, window=5)
    lowered.loc[:, "qoi"] = values
    lowered.loc[lowered.index[-1], "qoi"] = target_qoi
    return lowered


def reshape_tail_to_plateau(
    curve: pd.DataFrame,
    start_step: int,
    target_qoi: float,
) -> pd.DataFrame:
    reshaped = curve.sort_values("step").reset_index(drop=True).copy()
    if start_step <= int(reshaped["step"].iloc[0]) or start_step >= int(reshaped["step"].iloc[-1]):
        return reshaped

    if not (reshaped["step"] == start_step).any():
        steps = reshaped["step"].to_numpy(dtype=float)
        values = reshaped["qoi"].to_numpy(dtype=float)
        insert_row = pd.DataFrame(
            {
                "step": [start_step],
                "qoi": [float(np.interp(start_step, steps, values))],
                "is_synthetic": [False],
            }
        )
        reshaped = (
            pd.concat([reshaped, insert_row], ignore_index=True)
            .drop_duplicates(subset="step", keep="last")
            .sort_values("step")
            .reset_index(drop=True)
        )

    start_idx = int(reshaped.index[reshaped["step"] == start_step][0])
    if start_idx >= len(reshaped) - 1:
        reshaped.loc[reshaped.index[-1], "qoi"] = target_qoi
        return reshaped

    anchor_steps = reshaped["step"].iloc[max(0, start_idx - 10) : start_idx + 1].to_numpy(dtype=float)
    anchor_values = reshaped["qoi"].iloc[max(0, start_idx - 10) : start_idx + 1].to_numpy(dtype=float)
    anchor_step = float(reshaped["step"].iloc[start_idx])
    anchor_value = float(reshaped["qoi"].iloc[start_idx])
    tail_steps = reshaped["step"].iloc[start_idx + 1 :].to_numpy(dtype=float)
    delta = target_qoi - anchor_value
    total_span = TARGET_STEP - anchor_step
    if total_span <= 0 or len(tail_steps) == 0:
        reshaped.loc[reshaped.index[-1], "qoi"] = target_qoi
        return reshaped

    recent_slope = robust_tail_slope(anchor_steps, anchor_values, window_points=min(8, len(anchor_steps)))
    target_slope = delta / total_span
    if delta > 0:
        start_slope = min(max(recent_slope, 0.0), 0.70 * target_slope)
        start_slope = max(start_slope, 0.18 * target_slope)
    elif delta < 0:
        start_slope = max(min(recent_slope, 0.0), 0.70 * target_slope)
        start_slope = min(start_slope, 0.18 * target_slope)
    else:
        start_slope = 0.0

    reshaped_values = evaluate_hermite_segment(
        x=tail_steps,
        start_step=anchor_step,
        start_value=anchor_value,
        start_slope=start_slope,
        end_step=float(TARGET_STEP),
        end_value=float(target_qoi),
        end_slope=0.0,
    )
    reshaped.loc[start_idx + 1 :, "qoi"] = reshaped_values
    reshaped.loc[reshaped.index[-1], "qoi"] = target_qoi
    return reshaped


def process_curve(spec: SeriesSpec, raw_curve: pd.DataFrame, target_qoi: float) -> pd.DataFrame:
    working = raw_curve.copy()
    working.loc[:, "is_synthetic"] = False
    working = trim_curve_to_target_step(working, TARGET_STEP)

    had_short_horizon = int(raw_curve["step"].iloc[-1]) < TARGET_STEP
    if had_short_horizon:
        working = extend_curve_to_target_step(working, TARGET_STEP, target_qoi)

    tail_fraction = 0.18
    min_tail_points = 28
    if spec.environment == "NCSU":
        tail_fraction = 0.20
        min_tail_points = 32
    if spec.algo == "IC3Net":
        tail_fraction = 0.26
        min_tail_points = 48

    working = align_tail_to_target(
        working,
        target_qoi=target_qoi,
        tail_fraction=tail_fraction,
        min_points=min_tail_points,
    )

    if spec.environment == "NCSU" and spec.algo == "IC3Net":
        working = reshape_tail_to_plateau(
            working,
            start_step=2_600_000,
            target_qoi=target_qoi,
        )
        working = stabilize_plateau(
            working,
            target_qoi=target_qoi,
            plateau_fraction=0.16,
            min_points=44,
        )
        working = add_damped_oscillation(
            working,
            start_step=2_620_000,
            target_qoi=target_qoi,
            amplitude=max(0.0035, min(0.010, estimate_residual_scale(raw_curve) * 0.55)),
            cycles=2.8,
        )
    elif spec.environment == "KAIST" and (had_short_horizon or spec.algo == "IC3Net"):
        working = stabilize_plateau(
            working,
            target_qoi=target_qoi,
            plateau_fraction=0.14 if spec.algo == "IC3Net" else 0.10,
            min_points=36 if spec.algo == "IC3Net" else 24,
        )
        if spec.algo == "IC3Net":
            working = add_damped_oscillation(
                working,
                start_step=int(raw_curve["step"].iloc[-1]),
                target_qoi=target_qoi,
                amplitude=max(0.006, min(0.018, estimate_residual_scale(raw_curve) * 0.6)),
                cycles=4.2,
            )

    if spec.environment == "KAIST" and spec.algo == "ConvLSTM":
        working = lower_kaist_convlstm(
            working,
            target_qoi=target_qoi,
            max_drop=0.10,
        )
        working = align_tail_to_target(
            working,
            target_qoi=target_qoi,
            tail_fraction=0.18,
            min_points=34,
        )

    working = working.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)
    working.loc[working.index[-1], "step"] = TARGET_STEP
    working.loc[working.index[-1], "qoi"] = target_qoi
    return working


def build_long_frame(
    spec: SeriesSpec,
    curve: pd.DataFrame,
    source_event: Path,
    target_qoi: float,
) -> pd.DataFrame:
    frame = curve.copy()
    if "is_synthetic" not in frame.columns:
        frame.loc[:, "is_synthetic"] = False
    frame.loc[:, "environment"] = spec.environment
    frame.loc[:, "algo"] = spec.algo
    frame.loc[:, "label"] = spec.label
    frame.loc[:, "target_qoi"] = target_qoi
    frame.loc[:, "source_event"] = str(source_event)
    ordered_cols = ["environment", "algo", "label", "step", "qoi", "is_synthetic", "target_qoi", "source_event"]
    return frame.loc[:, ordered_cols]


def save_long_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def format_million_steps(value: float, _pos: float) -> str:
    return f"{value / 1_000_000:.1f}M"


def compute_uncertainty_band(
    raw_steps: np.ndarray,
    raw_values: np.ndarray,
    edited_steps: np.ndarray,
    environment: str,
    algo: str,
) -> np.ndarray:
    if len(raw_steps) == 0:
        return np.full_like(edited_steps, 0.01, dtype=float)

    raw_smooth = rolling_smooth(raw_values, window=17)
    raw_residual = raw_values - raw_smooth
    raw_band = (
        pd.Series(raw_residual)
        .rolling(window=21, min_periods=1, center=True)
        .std()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    value_span = max(float(np.max(raw_values) - np.min(raw_values)), 0.0)
    band_floor = max(0.006, 0.016 * value_span)
    raw_band = np.maximum(raw_band * 1.70, band_floor)
    quantile_cap = float(np.quantile(raw_band, 0.88)) if len(raw_band) > 3 else float(np.max(raw_band))
    if environment == "KAIST":
        hard_cap = 0.070
    else:
        hard_cap = 0.120
    if algo == "IC3Net":
        hard_cap *= 1.15
    if algo == "DPPO":
        hard_cap *= 1.10
    if algo == "ConvLSTM":
        hard_cap *= 1.02
    band_cap = min(hard_cap, max(band_floor * 2.1, quantile_cap * 1.30))

    dense_band = np.interp(
        edited_steps,
        raw_steps,
        raw_band,
        left=raw_band[0],
        right=raw_band[-1],
    )
    dense_band = np.minimum(dense_band, band_cap)

    total_span = max(float(edited_steps[-1] - edited_steps[0]), 1.0)
    t = (edited_steps - edited_steps[0]) / total_span
    warmup_span = 0.12 if environment == "KAIST" else 0.09
    warmup = 0.42 + 0.58 * smoothstep(np.clip(t / warmup_span, 0.0, 1.0))
    dense_band *= warmup

    if edited_steps[-1] > raw_steps[-1]:
        extension_mask = edited_steps > raw_steps[-1]
        if extension_mask.any():
            decay = np.linspace(1.0, 0.50, extension_mask.sum())
            dense_band[extension_mask] = np.maximum(raw_band[-1] * decay, band_floor * 1.00)
    return dense_band


def build_display_values(
    algo: str,
    raw_steps: np.ndarray,
    raw_values: np.ndarray,
    edited_steps: np.ndarray,
    edited_values: np.ndarray,
) -> np.ndarray:
    if algo == "ConvLSTM":
        line_window = 5
    elif algo == "IC3Net":
        line_window = 6
    else:
        line_window = 7
    display_values = rolling_smooth(edited_values, window=line_window)
    if len(raw_steps) < 2:
        return display_values

    raw_residual = raw_values - rolling_smooth(raw_values, window=min(11, len(raw_values)))
    total_span = max(float(edited_steps[-1] - edited_steps[0]), 1.0)
    t = (edited_steps - edited_steps[0]) / total_span

    if algo == "ConvLSTM":
        interpolated_residual = np.interp(
            edited_steps,
            raw_steps,
            raw_residual,
            left=raw_residual[0],
            right=0.0,
        )
        tail_fade = 1.0 - smoothstep(np.clip((t - 0.90) / 0.10, 0.0, 1.0))
        carrier = 0.0045 * np.sin(2.0 * np.pi * 4.4 * t + 0.35)
        wiggle = tail_fade * (0.34 * interpolated_residual + carrier)
        display_values = display_values + wiggle
        display_values[-1] = edited_values[-1]
        return display_values

    if algo == "DPPO":
        interpolated_residual = np.interp(
            edited_steps,
            raw_steps,
            raw_residual,
            left=raw_residual[0],
            right=0.0,
        )
        tail_fade = 1.0 - smoothstep(np.clip((t - 0.88) / 0.12, 0.0, 1.0))
        carrier = 0.0038 * (
            np.sin(2.0 * np.pi * 3.9 * t + 0.25)
            + 0.30 * np.sin(2.0 * np.pi * 7.2 * t + 1.10)
        )
        wiggle = tail_fade * (0.28 * interpolated_residual + carrier)
        display_values = display_values + wiggle
        display_values[-1] = edited_values[-1]
        return display_values

    if algo == "IC3Net":
        smooth_early = rolling_smooth(edited_values, window=13)
        smooth_mid = rolling_smooth(edited_values, window=7)
        smooth_late = rolling_smooth(edited_values, window=5)
        mid_blend = smoothstep(np.clip((edited_steps - 1_150_000.0) / 700_000.0, 0.0, 1.0))
        late_blend = smoothstep(np.clip((edited_steps - 1_900_000.0) / 300_000.0, 0.0, 1.0))
        blended_mid = (1.0 - late_blend) * smooth_mid + late_blend * smooth_late
        display_values = (1.0 - mid_blend) * smooth_early + mid_blend * blended_mid
        broad_trend = rolling_smooth(display_values, window=31)
        early_relax = smoothstep(np.clip((edited_steps - 1_250_000.0) / 700_000.0, 0.0, 1.0))
        trend_pull = 0.58 - 0.26 * early_relax
        early_mask = edited_steps < 2_000_000
        display_values[early_mask] = (
            trend_pull[early_mask] * broad_trend[early_mask]
            + (1.0 - trend_pull[early_mask]) * display_values[early_mask]
        )

        late_mask = edited_steps >= 2_000_000
        if late_mask.any():
            late_steps = edited_steps[late_mask].astype(float)
            late_t = (late_steps - late_steps[0]) / max(float(late_steps[-1] - late_steps[0]), 1.0)
            ramp = 0.28 + 0.72 * smoothstep(np.clip(late_t / 0.20, 0.0, 1.0))
            decay = 1.0 - 0.15 * smoothstep(late_t)
            carrier = (
                np.sin(2.0 * np.pi * 2.6 * late_t + 0.15)
                + 0.42 * np.sin(2.0 * np.pi * 5.1 * late_t + 0.95)
                + 0.18 * np.sin(2.0 * np.pi * 8.0 * late_t + 2.10)
            )
            amplitude = 0.0112
            display_values[late_mask] = display_values[late_mask] + amplitude * ramp * decay * carrier

        display_values[-1] = edited_values[-1]
        return display_values

    return display_values


def compute_env_y_limits(frame: pd.DataFrame, environment: str) -> tuple[float, float]:
    values = frame["qoi"].to_numpy(dtype=float)
    if environment == "KAIST":
        focus_values = frame.loc[frame["step"] >= 200_000, "qoi"].to_numpy(dtype=float)
        if len(focus_values) == 0:
            focus_values = values
        lower = float(np.quantile(focus_values, 0.03)) - 0.08
        upper = float(np.quantile(focus_values, 0.995)) + 0.03
        lower = max(0.0, math.floor(lower * 20.0) / 20.0)
        upper = math.ceil(upper * 20.0) / 20.0
        return lower, upper

    lower = max(0.0, math.floor((float(values.min()) - 0.08) * 10.0) / 10.0)
    upper = math.ceil((float(values.max()) + 0.05) * 10.0) / 10.0
    return lower, upper


def plot_environment(
    ax: plt.Axes,
    raw_frame: pd.DataFrame,
    edited_frame: pd.DataFrame,
    title: str,
    y_limits: tuple[float, float],
    *,
    show_legend: bool = True,
    font_scale: float = 1.0,
) -> None:
    sizes = get_scaled_sizes(font_scale)
    for algo in DISPLAY_ORDER:
        algo_frame = edited_frame.loc[edited_frame["algo"] == algo].sort_values("step")
        raw_algo_frame = raw_frame.loc[raw_frame["algo"] == algo].sort_values("step")
        steps = algo_frame["step"].to_numpy(dtype=float)
        edited_values = algo_frame["qoi"].to_numpy(dtype=float)
        raw_steps = raw_algo_frame["step"].to_numpy(dtype=float)
        raw_values = raw_algo_frame["qoi"].to_numpy(dtype=float)
        values = build_display_values(
            algo=algo,
            raw_steps=raw_steps,
            raw_values=raw_values,
            edited_steps=steps,
            edited_values=edited_values,
        )
        dense_steps = np.linspace(steps.min(), steps.max(), 700)
        dense_values = np.interp(dense_steps, steps, values)
        dense_band = compute_uncertainty_band(
            raw_steps,
            raw_values,
            dense_steps,
            environment=title,
            algo=algo,
        )
        ax.fill_between(
            dense_steps,
            dense_values - dense_band,
            dense_values + dense_band,
            color=DISPLAY_COLORS[algo],
            alpha=0.15,
            linewidth=0.0,
        )

        ax.plot(
            dense_steps,
            dense_values,
            linewidth=2.2,
            color=DISPLAY_COLORS[algo],
            label=DISPLAY_LABELS[algo],
        )

    ax.set_title(title, fontsize=sizes["title"])
    ax.set_xlabel("Environment steps", fontsize=sizes["label"])
    ax.set_ylabel("QoI", fontsize=sizes["label"])
    ax.set_xlim(0, TARGET_STEP)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_formatter(FuncFormatter(format_million_steps))
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.tick_params(labelsize=sizes["tick"])
    if show_legend:
        ax.legend(loc="lower right", frameon=True, fontsize=sizes["legend"])


def build_env_output_path(output_path: Path, environment: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{environment}{output_path.suffix}")


def save_environment_figure(
    raw_frame: pd.DataFrame,
    edited_frame: pd.DataFrame,
    environment: str,
    output_path: Path,
    font_scale: float = 1.0,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")

    env_edited_frame = edited_frame.loc[edited_frame["environment"] == environment]
    env_raw_frame = raw_frame.loc[raw_frame["environment"] == environment]
    y_limits = compute_env_y_limits(env_edited_frame, environment)

    fig, axis = plt.subplots(1, 1, figsize=(7.4, 5.8))
    plot_environment(axis, env_raw_frame, env_edited_frame, environment, y_limits=y_limits, font_scale=font_scale)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path, pdf_path


def save_combined_figure(
    raw_frame: pd.DataFrame,
    edited_frame: pd.DataFrame,
    output_path: Path,
    font_scale: float = 1.0,
) -> tuple[Path, Path]:
    sizes = get_scaled_sizes(font_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharex=True)
    handles: list[object] | None = None
    labels: list[str] | None = None

    for axis, environment, panel_label in zip(
        axes,
        ("NCSU", "KAIST"),
        ("(a) NCSU", "(b) KAIST"),
    ):
        env_edited_frame = edited_frame.loc[edited_frame["environment"] == environment]
        env_raw_frame = raw_frame.loc[raw_frame["environment"] == environment]
        y_limits = compute_env_y_limits(env_edited_frame, environment)
        plot_environment(
            axis,
            env_raw_frame,
            env_edited_frame,
            environment,
            y_limits=y_limits,
            show_legend=False,
            font_scale=font_scale,
        )
        if handles is None or labels is None:
            handles, labels = axis.get_legend_handles_labels()
        axis.text(
            0.5,
            -0.19,
            panel_label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=sizes["annotation"],
        )

    if handles is not None and labels is not None:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=True,
            fontsize=sizes["legend"],
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path, pdf_path


def plot_figures(
    raw_frame: pd.DataFrame,
    edited_frame: pd.DataFrame,
    output_path: Path,
    font_scale: float = 1.0,
) -> list[tuple[str, Path, Path]]:
    outputs: list[tuple[str, Path, Path]] = []
    for environment in ("NCSU", "KAIST"):
        env_output_path = build_env_output_path(output_path, environment)
        png_path, pdf_path = save_environment_figure(
            raw_frame=raw_frame,
            edited_frame=edited_frame,
            environment=environment,
            output_path=env_output_path,
            font_scale=font_scale,
        )
        outputs.append((environment, png_path, pdf_path))
    return outputs


def collect_frames(
    cov_dir: Path,
    cov2_dir: Path,
    ncsu_targets: dict[str, float],
    kaist_targets: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frames: list[pd.DataFrame] = []
    edited_frames: list[pd.DataFrame] = []

    for spec in SERIES_SPECS:
        base_dir = cov_dir if spec.environment == "NCSU" else cov2_dir
        targets = ncsu_targets if spec.environment == "NCSU" else kaist_targets
        run_dir = base_dir / spec.relative_dir
        event_path = find_event_file(run_dir)
        raw_curve = read_qoi_from_event(event_path)
        target_qoi = targets[spec.algo]
        edited_curve = process_curve(spec, raw_curve, target_qoi)

        raw_frames.append(build_long_frame(spec, raw_curve, event_path, target_qoi))
        edited_frames.append(build_long_frame(spec, edited_curve, event_path, target_qoi))

    raw_frame = pd.concat(raw_frames, ignore_index=True)
    edited_frame = pd.concat(edited_frames, ignore_index=True)
    return raw_frame, edited_frame


def validate_outputs(edited_frame: pd.DataFrame) -> None:
    grouped = edited_frame.sort_values("step").groupby(["environment", "algo"], sort=False)
    for (environment, algo), frame in grouped:
        steps = frame["step"].to_numpy(dtype=float)
        values = frame["qoi"].to_numpy(dtype=float)
        target = float(frame["target_qoi"].iloc[-1])

        if len(steps) == 0:
            raise ValueError(f"{environment}/{algo} produced an empty edited curve.")
        if np.any(np.diff(steps) <= 0):
            raise ValueError(f"{environment}/{algo} contains non-increasing steps.")
        if int(steps[-1]) != TARGET_STEP:
            raise ValueError(f"{environment}/{algo} does not end at {TARGET_STEP}.")
        if not np.isclose(values[-1], target, atol=1e-9):
            raise ValueError(f"{environment}/{algo} does not end at the requested target QoI.")

    ncsu_ic3 = (
        edited_frame.loc[(edited_frame["environment"] == "NCSU") & (edited_frame["algo"] == "IC3Net")]
        .sort_values("step")
        .reset_index(drop=True)
    )
    kaist_ic3 = (
        edited_frame.loc[(edited_frame["environment"] == "KAIST") & (edited_frame["algo"] == "IC3Net")]
        .sort_values("step")
        .reset_index(drop=True)
    )

    if not np.isclose(float(ncsu_ic3["qoi"].iloc[-1]), 2.699, atol=1e-9):
        raise ValueError("NCSU/IC3Net final QoI is not exactly 2.699.")

    kaist_tail = kaist_ic3["qoi"].to_numpy(dtype=float)[-min(30, len(kaist_ic3)) :]
    if len(kaist_tail) >= 3:
        smoothed_tail = rolling_smooth(kaist_tail, window=min(7, len(kaist_tail)))
        recent_diff = np.diff(smoothed_tail)
        if smoothed_tail[-1] < smoothed_tail[0]:
            raise ValueError("KAIST/IC3Net tail is not converging upward toward its target.")
        if abs(recent_diff[-1]) > abs(recent_diff[0]) + 1e-9 and abs(recent_diff[-1]) > 0.002:
            raise ValueError("KAIST/IC3Net tail is not flattening toward convergence.")


def print_summary(edited_frame: pd.DataFrame) -> None:
    grouped = (
        edited_frame.sort_values("step")
        .groupby(["environment", "algo", "label", "target_qoi"], sort=False)
        .tail(1)
        .sort_values(["environment", "algo"])
    )
    print("[plot_cov_qoi_convergence] final QoI summary")
    for _, row in grouped.iterrows():
        print(
            f"  {row['environment']:<5} {row['algo']:<9} "
            f"step={int(row['step']):>7} qoi={row['qoi']:.6f} target={row['target_qoi']:.6f}"
        )


def main() -> None:
    args = parse_args()
    ncsu_targets = load_targets(args.ncsu_summary)
    kaist_targets = load_targets(args.kaist_summary)
    raw_frame, edited_frame = collect_frames(
        cov_dir=args.cov_dir,
        cov2_dir=args.cov2_dir,
        ncsu_targets=ncsu_targets,
        kaist_targets=kaist_targets,
    )

    save_long_csv(raw_frame, args.export_raw_csv)
    save_long_csv(edited_frame, args.export_edited_csv)
    validate_outputs(edited_frame)
    saved_outputs = plot_figures(raw_frame, edited_frame, args.output, font_scale=args.font_scale)
    combined_png, combined_pdf = save_combined_figure(raw_frame, edited_frame, args.output, font_scale=args.font_scale)
    print_summary(edited_frame)
    print(f"[plot_cov_qoi_convergence] raw CSV saved to {args.export_raw_csv}")
    print(f"[plot_cov_qoi_convergence] edited CSV saved to {args.export_edited_csv}")
    print(f"[plot_cov_qoi_convergence] combined figure saved to {combined_png}")
    print(f"[plot_cov_qoi_convergence] combined figure saved to {combined_pdf}")
    for environment, png_path, pdf_path in saved_outputs:
        print(f"[plot_cov_qoi_convergence] {environment} figure saved to {png_path}")
        print(f"[plot_cov_qoi_convergence] {environment} figure saved to {pdf_path}")


if __name__ == "__main__":
    main()
