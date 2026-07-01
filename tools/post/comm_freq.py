# -*- coding: utf-8 -*-
"""
通信频率分析脚本 (Communication Frequency Analysis)
=================================================
从多次实验运行的邻接矩阵中统计每架无人机的通信频率，并绘制
分组柱状图 (类似论文 Fig.15(b))。

通信频率定义:
    对于 UAV_i，其通信频率 = 它在整个 episode 中 **与任意其他 UAV
    存在通信边** 的时间步占比 (排除自环)。

用法示例:
    python tools/post/comm_freq.py --datasets NCSU KAIST Beijing \
        --run_dirs "runs/NCSU/run1,runs/NCSU/run2,..." \
                   "runs/KAIST/run1,runs/KAIST/run2,..." \
                   "runs/Beijing/run1,run2,..."
    
    或者用 --base_dir 自动扫描:
    python tools/post/comm_freq.py --datasets NCSU KAIST --base_dir ../runs \
        --algo_keyword MetaComm --tag test

输出: 在当前目录生成 comm_freq.pdf / comm_freq.png
"""

from pathlib import Path
import sys, os

HERE = Path(__file__).resolve()
project_root = None
for parent in HERE.parents:
    if (parent / 'tools' / 'macro' / 'macro.py').exists():
        project_root = parent
        break
if project_root is None:
    project_root = HERE.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from plot_style import add_figsize_args, resolve_figsize


# ==================== UAV 颜色定义 (与 vis.py 保持一致) ====================
UAV_COLORS = {
    0: '#%02X%02X%02X' % (0, 128, 0),    # Green UAV
    1: '#%02X%02X%02X' % (0, 0, 255),     # Blue UAV
    2: '#%02X%02X%02X' % (130, 43, 226),  # Purple UAV
}
UAV_LABELS = {
    0: 'Green UAV',
    1: 'Blue UAV',
    2: 'Purple UAV',
}


def compute_comm_freq_from_adj(adj_path):
    """
    从 adj.npz 文件计算每架无人机的通信频率。

    Args:
        adj_path: eps_best_adj.npz 文件路径

    Returns:
        comm_freqs: np.array of shape [n_agent], 每架 UAV 的通信频率
    """
    data = np.load(adj_path)
    adj = data['arr_0']  # shape: [T, n_agent, n_agent]
    T, n_agent, _ = adj.shape

    comm_freqs = np.zeros(n_agent)
    for i in range(n_agent):
        # 对于 UAV_i, 统计每个时间步是否与至少一架其他 UAV 通信
        # adj[t, i, j] == 1 表示 t 时刻 i→j 有通信边
        # 排除自环 (i != j)
        mask = np.ones(n_agent, dtype=bool)
        mask[i] = False
        has_comm = (adj[:, i, mask].sum(axis=1) > 0)  # [T], bool
        comm_freqs[i] = has_comm.mean()

    return comm_freqs


def find_adj_files(output_dir, tag='test'):
    """在一个实验输出目录中查找 adj 文件"""
    saved_dir = os.path.join(output_dir, f'{tag}_saved_trajs')
    candidates = [
        os.path.join(saved_dir, 'eps_best_adj.npz'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # 尝试找任何 _adj.npz
    if os.path.exists(saved_dir):
        for f in os.listdir(saved_dir):
            if f.endswith('_adj.npz'):
                return os.path.join(saved_dir, f)
    return None


def scan_runs(base_dir, dataset, algo_keyword='MetaComm'):
    """在 base_dir 下扫描指定 dataset 的实验目录"""
    run_dirs = []
    # 常见目录结构: base_dir/dataset/*/algo_keyword*/
    # 或者: base_dir/*/dataset_algo_keyword*/
    for root, dirs, files in os.walk(base_dir):
        if 'params.json' in files:
            params_file = os.path.join(root, 'params.json')
            try:
                with open(params_file, 'r') as f:
                    params = json.load(f)
                    ds = params.get('input_args', {}).get('dataset', '')
                    algo = params.get('input_args', {}).get('algo', '')
                    if ds == dataset and algo_keyword in algo:
                        run_dirs.append(root)
            except:
                pass
    return run_dirs


def plot_comm_freq(results, output_path='comm_freq', n_agent=3, figsize=None):
    """
    绘制通信频率分组柱状图

    Args:
        results: dict, {dataset_name: np.array of shape [n_runs, n_agent]}
        output_path: 输出文件名 (不含扩展名)
        n_agent: UAV 数量
        figsize: (width, height) tuple, 默认自动计算
    """
    datasets = list(results.keys())
    n_datasets = len(datasets)

    # ---------- 图表样式参数 ----------
    fig_width = max(6, n_datasets * 2.8)
    if figsize is None:
        figsize = (fig_width, 5)
    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.22
    group_gap = 0.15
    group_width = n_agent * bar_width + group_gap

    for d_idx, dataset in enumerate(datasets):
        data = results[dataset]  # [n_runs, n_agent]

        for uav_idx in range(min(n_agent, data.shape[1])):
            mean_val = data[:, uav_idx].mean()
            std_val = data[:, uav_idx].std() if data.shape[0] > 1 else 0

            x_pos = d_idx * group_width + uav_idx * bar_width
            color = UAV_COLORS.get(uav_idx, '#808080')
            label = UAV_LABELS.get(uav_idx, f'UAV {uav_idx}') if d_idx == 0 else None

            ax.bar(x_pos, mean_val, width=bar_width * 0.85,
                   color=color, label=label,
                   edgecolor='black', linewidth=0.5)
            ax.errorbar(x_pos, mean_val, yerr=std_val,
                        fmt='none', ecolor='black', capsize=3, linewidth=1.5)

    # ---------- 坐标轴设置 ----------
    ax.set_ylabel(r'Communication Frequency', fontsize=14)
    ax.set_ylim(0.0, 0.7)
    ax.set_yticks(np.arange(0, 0.8, 0.1))

    # X 轴标签放在每组中央
    x_centers = [d_idx * group_width + (n_agent - 1) * bar_width / 2
                 for d_idx in range(n_datasets)]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(datasets, fontsize=13)

    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # ---------- 保存 ----------
    pdf = PdfPages(output_path + '.pdf')
    pdf.savefig(fig)
    pdf.close()
    fig.savefig(output_path + '.png', dpi=200)
    plt.close(fig)
    print(f"[comm_freq] 图表已保存: {output_path}.pdf / .png")


def main():
    parser = argparse.ArgumentParser(description='分析并绘制无人机通信频率 (Fig.15b)')
    parser.add_argument('--datasets', nargs='+', default=['NCSU', 'KAIST', 'Beijing'],
                        help='数据集名称列表')
    parser.add_argument('--run_dirs', nargs='+', default=None,
                        help='每个 dataset 对应的实验目录 (逗号分隔多次运行)')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='实验根目录, 用于自动扫描')
    parser.add_argument('--algo_keyword', type=str, default='MetaComm',
                        help='算法关键词, 用于自动扫描过滤')
    parser.add_argument('--tag', type=str, default='test', choices=['train', 'test'],
                        help='使用 train 还是 test 的邻接矩阵')
    parser.add_argument('--output', type=str, default='comm_freq',
                        help='输出文件名 (不含扩展名)')
    add_figsize_args(parser)
    args = parser.parse_args()

    results = {}

    if args.run_dirs is not None:
        # 手动指定实验目录
        assert len(args.datasets) == len(args.run_dirs), \
            f"datasets ({len(args.datasets)}) 和 run_dirs ({len(args.run_dirs)}) 数量必须一致"

        for dataset, run_dirs_str in zip(args.datasets, args.run_dirs):
            dirs = [d.strip() for d in run_dirs_str.split(',')]
            all_freqs = []
            for d in dirs:
                adj_file = find_adj_files(d, tag=args.tag)
                if adj_file is None:
                    print(f"[警告] 未找到 adj 文件: {d}")
                    continue
                freq = compute_comm_freq_from_adj(adj_file)
                all_freqs.append(freq)
                print(f"  {d}: {freq}")
            if all_freqs:
                results[dataset] = np.stack(all_freqs)

    elif args.base_dir is not None:
        # 自动扫描模式
        for dataset in args.datasets:
            dirs = scan_runs(args.base_dir, dataset, args.algo_keyword)
            print(f"\n[{dataset}] 找到 {len(dirs)} 个实验目录:")
            all_freqs = []
            for d in dirs:
                adj_file = find_adj_files(d, tag=args.tag)
                if adj_file is None:
                    # 也试试 train
                    adj_file = find_adj_files(d, tag='train')
                if adj_file is None:
                    print(f"  [跳过] {d} (无 adj 文件)")
                    continue
                freq = compute_comm_freq_from_adj(adj_file)
                all_freqs.append(freq)
                print(f"  {os.path.basename(d)}: comm_freq={np.round(freq, 3)}")
            if all_freqs:
                results[dataset] = np.stack(all_freqs)

    else:
        parser.error("请指定 --run_dirs 或 --base_dir")

    if not results:
        print("[错误] 未找到有效的 adj 数据")
        return

    # 确定 n_agent
    n_agent = max(v.shape[1] for v in results.values())

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("通信频率统计摘要")
    print("=" * 60)
    for dataset, data in results.items():
        print(f"\n{dataset} ({data.shape[0]} runs, {data.shape[1]} UAVs):")
        for uav_idx in range(data.shape[1]):
            mean = data[:, uav_idx].mean()
            std = data[:, uav_idx].std() if data.shape[0] > 1 else 0
            label = UAV_LABELS.get(uav_idx, f'UAV {uav_idx}')
            print(f"  {label}: {mean:.3f} ± {std:.3f}")

    _fs = resolve_figsize(args, default=(max(6, len(results) * 2.8), 5))
    plot_comm_freq(results, output_path=args.output, n_agent=n_agent, figsize=_fs)

    # ==================== 结果意义分析 ====================
    print("\n" + "=" * 60)
    print("结果意义分析")
    print("=" * 60)
    print("""
通信频率 (Communication Frequency) 反映了 CommFormer 学习到的
智能体间通信拓扑结构。该指标的含义如下：

1. 通信频率的高低：
   - 高频率 (如 0.5-0.7) 表示该 UAV 在大部分时间步需要与其他
     UAV 交换信息才能做出良好决策 —— 说明该环境/区域下
     多机协作至关重要。
   - 低频率 (如 0.1-0.3) 表示该 UAV 大部分时间可以独立决策，
     偶尔才需要协调 —— 说明任务可以被较好地分解。

2. UAV 间的差异：
   - 不同 UAV 的通信频率不同，说明 CommFormer 学到了
     非对称的通信结构 —— 某些 UAV 扮演"通信枢纽"角色。
   - 这验证了可学习通信拓扑优于固定全连接通信的假设：
     并非所有 UAV 都需要同等频率的通信。

3. 跨数据集的差异：
   - 不同数据集的用户分布、移动模式不同，导致最优通信
     结构不同。CommFormer 能自适应地为不同场景学习不同
     的通信拓扑，体现了其泛化能力。

4. 误差棒 (Error Bar)：
   - 较小的误差棒说明通信结构在多次运行中是一致的，
     即 CommFormer 能稳定地发现相似的通信拓扑。
   - 较大的误差棒可能表明存在多种近似最优的通信策略。
""")


if __name__ == '__main__':
    main()
