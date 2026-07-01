from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "source_code"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from main_DPPO import getAlgArgs, getRunArgs, override  # noqa: E402
from algorithms.utils import LogClient, LogServer  # noqa: E402
from algorithms.algo.main import OnPolicyRunner  # noqa: E402
from algorithms.algo.agent.MetaComm import MetaCommAgent  # noqa: E402
from envs.env_mobile import EnvMobile  # noqa: E402
from env_configs.wrappers.env_wrappers import SubprocVecEnv  # noqa: E402


DEFAULTS = {
    "debug": False,
    "test": False,
    "test_with_shenbi": False,
    "test_save_heatmap": False,
    "user": "yyx",
    "env": "Mobile",
    "algo": "MetaComm",
    "device": "cuda:0",
    "dataset": "NCSU",
    "poi_num": 48,
    "tag": "",
    "output_dir": "runs/feature_eval",
    "group": "feature_eval",
    "mute_wandb": True,
    "checkpoint": None,
    "n_thread": 1,
    "n_iter": -1,
    "seed": 1,
    "lr": None,
    "lr_v": None,
    "lr_colla": None,
    "use_stack_frame": False,
    "g2a_hidden_dim": 64,
    "tau": 1.0,
    "tau_schedule": "fixed",
    "tau_start": None,
    "tau_end": None,
    "tau_anneal_end_iter": 0,
    "map_size": 6,
    "g2a_hops": 1,
    "update_colla_by_v_0307": False,
    "fixed_range": True,
    "collect_range": 500,
    "dyna_level": "2",
    "init_energy": 719280,
    "w_noise": -107,
    "user_data_amount": 0.75,
    "update_num": 10,
    "uav_num": 3,
    "fixed_col_time": True,
    "aoith": 30,
    "txth": 3,
    "uav_height": 100,
    "hao02191630": True,
    "always_fixed_antenna02230040": -1,
    "max_episode_step": 120,
    "future_obs": 0,
    "use_snrmap": False,
    "n_head": -1,
    "sparsity": -1.0,
    "metacomm_topology": "learned",
    "high_level_dont_use_snrmap": False,
    "high_level_knn_coefficient": -1,
    "aVPS": 0.2,
    "tVPS": 0.2,
    "reward_pref": None,
    "effective_aVPS": 0.2,
    "effective_tVPS": 0.2,
    "agent_field": 750,
    "failure_rate": 0.0,
    "failure_seed": 1,
    "failure_step_min_frac": 0.2,
    "failure_step_max_frac": 0.3,
    "agent_order_mode": "fixed",
    "agent_order_seed": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the best historical checkpoint for each attention-head setting, "
            "re-run evaluation with newly added feature-stability metrics, and "
            "export a summary CSV."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "n_heads",
        help="Root directory containing head_1/head_2/... folders.",
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=REPO_ROOT / "runs" / "n_heads" / "analysis_head2_anchor" / "nhead_run_summary.csv",
        help="Historical per-run summary used to choose the best checkpoint per head.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="QoI",
        help="Historical metric used to select the best run within each head count.",
    )
    parser.add_argument(
        "--selection-stat",
        choices=["best", "final", "tail_mean"],
        default="best",
        help="Historical statistic used to choose the best run within each head count.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=6,
        help="Number of evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used during checkpoint re-evaluation.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "runs" / "n_heads" / "analysis_feature_eval" / "best_head_feature_metrics.csv",
        help="CSV file collecting the re-evaluated metrics.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing eval_metrics.csv if present instead of re-running evaluation.",
    )
    return parser.parse_args()


def restore_input_args(stored: dict, checkpoint_dir: Path, eval_dir: Path, device: str) -> SimpleNamespace:
    merged = dict(DEFAULTS)
    merged.update(stored)

    if merged.get("dataset") == "NCSU" and merged.get("poi_num", 116) == 116:
        merged["poi_num"] = 48

    if merged.get("tau_start") is None:
        merged["tau_start"] = float(merged["tau"])
    if merged.get("tau_end") is None:
        merged["tau_end"] = float(merged["tau"])

    merged["test"] = True
    merged["test_with_shenbi"] = False
    merged["debug"] = False
    merged["mute_wandb"] = True
    merged["checkpoint"] = str(checkpoint_dir)
    merged["group"] = "feature_eval"
    merged["n_thread"] = 1
    merged["output_dir"] = str(eval_dir)
    merged["device"] = device

    return SimpleNamespace(**merged)


def build_env_args(input_args: SimpleNamespace) -> dict:
    env_args = {
        "max_episode_step": input_args.max_episode_step,
        "collect_range": input_args.collect_range,
        "initial_energy": input_args.init_energy,
        "user_data_amount": input_args.user_data_amount,
        "update_num": input_args.update_num,
        "uav_num": input_args.uav_num,
        "AoI_THRESHOLD": input_args.aoith,
        "RATE_THRESHOLD": input_args.txth,
        "uav_height": input_args.uav_height,
        "aoi_vio_penalty_scale": getattr(input_args, "effective_aVPS", input_args.aVPS),
        "tx_vio_penalty_scale": getattr(input_args, "effective_tVPS", input_args.tVPS),
        "hao02191630": input_args.hao02191630,
        "w_noise": input_args.w_noise,
        "agent_field": input_args.agent_field,
    }
    if getattr(input_args, "poi_num", None) is not None:
        env_args["poi_num"] = input_args.poi_num
    return env_args


def select_best_runs(summary_csv: Path, metric: str, stat: str) -> pd.DataFrame:
    frame = pd.read_csv(summary_csv)
    stat_col = stat
    sub = frame[frame["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"Cannot find metric={metric} in {summary_csv}")
    ordered = sub.sort_values(["n_head", stat_col], ascending=[True, False])
    best = ordered.groupby("n_head", as_index=False).first()
    return best.sort_values("n_head").reset_index(drop=True)


def run_single_evaluation(run_dir: Path, n_test: int, device: str, reuse: bool) -> dict:
    params_path = run_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.json under {run_dir}")

    eval_dir = run_dir / f"feature_eval_n{n_test}"
    metrics_csv = eval_dir / "graph_snapshots" / "eval_metrics.csv"

    if reuse and metrics_csv.exists():
        eval_frame = pd.read_csv(metrics_csv)
        return eval_frame.iloc[-1].to_dict()

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    input_args = restore_input_args(params["input_args"], checkpoint_dir=run_dir, eval_dir=eval_dir, device=device)
    env_args = build_env_args(input_args)

    run_args = getRunArgs(input_args)
    dummy_env = EnvMobile(env_args, input_args, phase="dummy")
    alg_args = getAlgArgs(run_args, input_args, dummy_env)
    alg_args, run_args, input_args = override(alg_args, run_args, input_args, dummy_env, MetaCommAgent)
    alg_args.n_test = int(n_test)

    envs_train = SubprocVecEnv([EnvMobile(env_args, input_args, phase="train") for _ in range(1)])
    envs_test = SubprocVecEnv([EnvMobile(env_args, input_args, phase="test") for _ in range(1)])

    try:
        logger = LogServer({"run_args": run_args, "algo_args": alg_args, "input_args": input_args})
        logger = LogClient(logger)
        agent = MetaCommAgent(logger, run_args.device, alg_args.agent_args, input_args)
        runner = OnPolicyRunner(
            logger=logger,
            agent=agent,
            envs_learn=envs_train,
            envs_test=envs_test,
            dummy_env=dummy_env,
            run_args=run_args,
            alg_args=alg_args,
            input_args=input_args,
        )
        runner.run()
    finally:
        envs_train.close()
        envs_test.close()
        dummy_env.close()

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Expected evaluation metrics at {metrics_csv}")

    eval_frame = pd.read_csv(metrics_csv)
    return eval_frame.iloc[-1].to_dict()


def main() -> None:
    args = parse_args()
    best_runs = select_best_runs(args.run_summary.resolve(), args.selection_metric, args.selection_stat)
    rows = []

    for row in best_runs.itertuples(index=False):
        run_dir = args.runs_dir.resolve() / f"head_{int(row.n_head)}" / row.run_name
        print(f"[eval_best_nhead_feature_metrics] evaluating head={int(row.n_head)} run={row.run_name}")
        metrics = run_single_evaluation(run_dir, n_test=args.n_test, device=args.device, reuse=args.reuse)
        metrics.update(
            {
                "n_head": int(row.n_head),
                "run_name": row.run_name,
                "selection_metric": args.selection_metric,
                "selection_stat": args.selection_stat,
            }
        )
        rows.append(metrics)

    out_path = args.output_csv.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("n_head").to_csv(out_path, index=False)
    print(f"[eval_best_nhead_feature_metrics] saved to {out_path}")


if __name__ == "__main__":
    main()
