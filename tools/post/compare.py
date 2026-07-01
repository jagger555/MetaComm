from pathlib import Path
import sys, os

HERE = Path(__file__).resolve()
project_root = None
for parent in HERE.parents:
    if (parent / 'tools' / 'macro' / 'macro.py').exists():
        project_root = parent
        break
if project_root is None:
    # 兜底（原来逻辑的三级上层）
    project_root = HERE.parents[2]

project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 可选：临时打印调试信息（运行后可删）
print('DEBUG: project_root =', project_root_str)
print('DEBUG: cwd =', os.getcwd())
print('DEBUG: sys.path[0:5] =', sys.path[:5])
print('DEBUG: macro exists =', (project_root / 'tools' / 'macro' / 'macro.py').exists())
import copy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
from tools.macro.macro import *
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False




def compare_plot(output_dir, xlabel, ylabel, xname, yname, x, yrange, ours, DPPO, CPPO, IC3Net, ConvLSTM, MetaComm,
                 figsize=(13, 13)):
    output_dir += f'/../pdf'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    pdf = PdfPages(output_dir + '/%s-%s.pdf' % (xname, yname))
    plt.figure(figsize=figsize)

    plt.xlabel(xlabel, fontsize=50)  # 42 by default
    plt.ylabel(ylabel, fontsize=50)
    plt.xticks(fontsize=50)  # 42 by default
    plt.yticks(fontsize=50)

    plt.plot(x, ours, color='dimgrey', marker='o', label='PPO-CS', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)
    plt.plot(x, DPPO, color='blue', marker='^', label=r'PPO-CPP', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)

    plt.plot(x, CPPO, color='turquoise', marker='s', label='PPO-JPCO', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, IC3Net, color='seagreen', marker='v', label='MAIC', markersize=30, markeredgewidth=5,  # Shortest Path
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, ConvLSTM, color='darkorange', marker='d', label='t-LocPred', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, MetaComm, color='red', marker='D', label='MetaComm', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)

    plt.xticks(x, x)

    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ylim(yrange[0], yrange[1])


    plt.grid(True)
    plt.grid(linestyle='--')

    if yname == 'Episodic AoI':
        plt.legend(loc='upper center', fontsize=42, ncol=2, markerscale=0.9, columnspacing=0.5,
                   )  # default, columnspacing = 2.0
    else:
        plt.legend(loc='lower center', fontsize=42, ncol=2, markerscale=0.9, columnspacing=0.5,
               )  # default, columnspacing = 2.0
    plt.tight_layout()

    pdf.savefig()
    plt.close()
    pdf.close()


def get_data(x_dir):
    df = None
    for file in os.listdir(x_dir):
        if not (file.endswith('csv') and 'ALL' in file) : continue
        df = pd.read_csv(os.path.join(x_dir, file), header=None)
    return df


def compare(x_dir):
    df = get_data(x_dir)

    if x_dir.endswith('uavnum'):
        x = 'uav_num'
        ticks = FIVE_UN_INDEX
    elif x_dir.endswith('aoith'):
        x = 'aoith'
        ticks = FIVE_AT_INDEX
    elif x_dir.endswith('txth'):
        x = 'txth'
        ticks = FIVE_TT_INDEX
    elif x_dir.endswith('amount'):
        x = 'user_data_amount'
        ticks = FIVE_AM_INDEX
    elif x_dir.endswith('updatenum'):
        x = 'update_num'
        ticks = FIVE_UPN_INDEX


    for i, metric in enumerate(METRICS):
        if metric == 'energy_consuming': continue
        compare_plot(output_dir=x_dir,
                     xlabel=xlabels[x],
                     ylabel=ylabels[metric],
                     xname=xnames[x],
                     yname=ynames[metric],
                     x=ticks,
                     yrange=yranges[metric],
                     ours=df.values[:, ALGOS.index('G2ANet')*5+METRICS.index(metric)],
                     DPPO=df.values[:, ALGOS.index('DPPO')*5+METRICS.index(metric)],
                     CPPO=df.values[:, ALGOS.index('CPPO')*5+METRICS.index(metric)],
                     IC3Net=df.values[:, ALGOS.index('IC3Net')*5+METRICS.index(metric)],
                     ConvLSTM=df.values[:, ALGOS.index('ConvLSTM')*5+METRICS.index(metric)],
                     MetaComm=df.values[:, ALGOS.index('MetaComm')*5+METRICS.index(metric)]
                     )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_dir', type=str)
    args = parser.parse_args()

    compare(args.x_dir)



