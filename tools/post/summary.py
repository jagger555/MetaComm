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
import json
import numpy as np
import os
import argparse
import pandas as pd
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('---------------------')
print(proj_dir)
os.chdir(proj_dir)

assert os.getcwd().endswith('source_code'), '请将工作路径设为source_code，否则无法正确导入包'

sys.path.append(os.getcwd())

from tools.macro.macro import *


def write_summary(x_dir, sum_file):
    # 确保目录存在
    os.makedirs(x_dir, exist_ok=True)

    with open(sum_file, 'w') as f:
        for root, dirs, files in os.walk(x_dir):
            for file in files:
                if not file == 'train_output.txt': continue
                result_file = os.path.join(root, file)
                f.write(result_file + '\n')
                with open(result_file, 'r') as re_f:
                    text = re_f.readlines()
                    case1 = 'ConvLSTM' in result_file
                    case2 = 'DPPO' in result_file
                    if case1 or case2:
                        trunc = len(text) // 2 if len(text) % 4 == 0 else len(text) // 2 + 1
                        # 如果 trunc 是奇数，就减 1 变成偶数
                        if trunc % 2 != 0:
                            trunc -= 1

                        assert trunc % 2 == 0  # 现在这一行一定能通过
                        text = text[:trunc]
                    metrics = text[-1]
                    f.write(metrics + '\n')
                print(1)


'''生成hyper tuning表格'''
def gen_hypertune_csv(x_dir, sum_file):
    ans = np.zeros((12, 5))

    with open(sum_file, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if line == '\n': continue
            # 如下两个if-else交替执行
            if line.endswith('output.txt\n'):  # 定位往哪填
                json_file = os.path.dirname(line) + '\\params.json'
                params = json.load(open(json_file, 'r'))
                idx1 = int(params['input_args']['g2a_hops'])
                idx2 = int(params['input_args']['map_size']) // 3 - 1
            else:  # 填数
                values = [0 for _ in range(len(METRICS))]
                for col, metric in enumerate(METRICS):
                    if metric in line:
                        start = line.index(metric) + len(metric) + 2  # +2 是适配output.txt的格式
                        end = start + line[start:].index('.') + 4  # 每个scalar数据保留三位小数
                        values[col] = float(line[start:end])
                    else:
                        raise ValueError
                ans[idx1*4+idx2] = values
    pd.DataFrame(ans).to_csv(x_dir + f'/hyper_ALL.csv', index=None, header=0)




'''生成five表格'''
def gen_five_csv(x_dir, sum_file):
    print(x_dir)
    if x_dir.endswith('uavnum'):
        x_index = FIVE_UN_INDEX
        key = 'uav_num'
    elif x_dir.endswith('aoith'):
        x_index = FIVE_AT_INDEX
        key = 'aoith'
    elif x_dir.endswith('txth'):
        x_index = FIVE_TT_INDEX
        key = 'txth'
    elif x_dir.endswith('amount'):
        x_index = FIVE_AM_INDEX
        key = 'user_data_amount'
    elif x_dir.endswith('updatenum'):
        x_index = FIVE_UPN_INDEX
        key = 'update_num'

    else:
        raise NotImplementedError('未实现的五点图自变量')

    metrics = METRICS

    dfs = [pd.DataFrame(np.zeros((len(x_index), len(metrics))), columns=metrics) \
            for _ in range(len(ALGOS))]  # 每个agent一个表

    with open(sum_file, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if line == '\n': continue
            # 如下两个if-else交替执行
            if line.endswith('output.txt\n'):  # 定位往哪填
                json_file = os.path.dirname(line) + '\\params.json'
                params = json.load(open(json_file, 'r'))

                # =========== 修改开始 ===========
                # 获取文件里的算法名
                algo_name = params['input_args']['algo']

                # 如果读到的是旧名字，强行改成新名字
                if algo_name == 'CommG2ANet':
                    algo_name = 'MetaComm'
                # ===============================

                row = x_index.index(params['input_args'][key])

                # 这里把 params['input_args']['algo'] 替换成处理过的 algo_name
                # 确保你的 macro.py 的 ALGOS 列表里包含了 'MetaComm'
                df = dfs[ALGOS.index(algo_name)]
            else:  # 填数
                item = dict()
                for col in metrics:
                    if col in line:
                        start = line.index(col) + len(col) + 2  # +2 是适配output.txt的格式
                        end = start + line[start:].index('.') + 4  # 每个scalar数据保留三位小数
                        item[col] = line[start:end]
                    else:
                        item[col] = '0.0'
                # 遇到重复记录时，保留 QoI 更大的那条
                try:
                    old_qoi = float(df.loc[row, 'QoI'])
                except Exception:
                    old_qoi = -1e9
                try:
                    new_qoi = float(item['QoI'])
                except Exception:
                    new_qoi = -1e9
                if new_qoi >= old_qoi:
                    df.loc[row] = item

    for i, df in enumerate(dfs):
        df.index = x_index
        df.columns.name = '{}-{}'.format(ALGOS[i], key)
        df.to_csv(x_dir + f'/five_{ALGOS[i]}_{key}.csv')
    np_all = np.hstack([df.values for df in dfs])
    pd.DataFrame(np_all).to_csv(x_dir + f'/five_ALL_{key}.csv', index=None, header=0)


def main(x_dir, five=False, hypertune=False):
    sum_file = x_dir + '/summary.txt'
    write_summary(x_dir, sum_file)
    if five:
        gen_five_csv(x_dir, sum_file)
    if hypertune:
        gen_hypertune_csv(x_dir, sum_file)
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--x_dir', type=str)
    args = parser.parse_args()
    main(args.x_dir, five=True)

    

