import os
import os.path as osp
import csv
from datetime import datetime
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GraphConvolutionalModel, MLP, CategoricalActor
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer
from gym.spaces.box import Box
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pickle
from copy import deepcopy as dp
from algorithms.models import CategoricalActor
import random
import multiprocessing as mp
# import torch.multiprocessing as mp
from torch import distributed as dist
import argparse
from algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, ModelBuffer
import time

SUMMARY_INFO_KEYS = (
    'QoI',
    'episodic_aoi',
    'aoi_satis_ratio',
    'data_satis_ratio',
    'tx_satis_ratio',
    'tx_reward',
    'good_reward',
    'aoi_penalty_reward',
    'tx_penalty_reward',
    'knn_reward',
    'energy_reward',
    'effective_aoi_task_reward',
    'effective_tx_task_reward',
    'effective_joint_task_reward',
    'energy_consuming',
    'a_poi_collect_ratio',
    'b_emergency_violation_ratio',
    'c_emergency_time',
    'd_aoi',
    'e_weighted_aoi',
    'f_weighted_bar_aoi',
    'h_total_energy_consuming',
    'h_energy_consuming_ratio',
    'f_episode_step',
)

OUTPUT_INFO_KEYS = (
    'QoI',
    'episodic_aoi',
    'aoi_satis_ratio',
    'data_satis_ratio',
    'tx_satis_ratio',
    'tx_reward',
    'good_reward',
    'aoi_penalty_reward',
    'effective_aoi_task_reward',
    'effective_tx_task_reward',
    'effective_joint_task_reward',
    'energy_consuming',
    'a_poi_collect_ratio',
    'b_emergency_violation_ratio',
    'c_emergency_time',
    'd_aoi',
    'e_weighted_aoi',
    'f_weighted_bar_aoi',
    'h_total_energy_consuming',
    'h_energy_consuming_ratio',
    'f_episode_step',
)


def mean_env_info(env_info, keys=SUMMARY_INFO_KEYS):
    summary = {}
    if not env_info:
        return summary
    for key in keys:
        values = [item[key] for item in env_info if key in item]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def write_output(info, output_dir, tag='train'):
    os.makedirs(output_dir, exist_ok=True)
    logging_path = osp.join(output_dir, f'{tag}_output.txt')
    with open(logging_path, 'a') as f:
        f.write('[' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']\n')
        metrics = []
        for key in OUTPUT_INFO_KEYS:
            if key in info:
                metrics.append(f"{key}: {'%.3f' % info[key]}")
        f.write(" ".join(metrics) + '\n')


class OnPolicyRunner:
    def __init__(self, logger, agent, envs_learn, envs_test, dummy_env,
                 run_args, alg_args, input_args, **kwargs):
        self.run_args = run_args
        self.input_args = input_args
        self.debug = self.run_args.debug
        self.logger = logger
        self.name = run_args.name
        # agent initialization
        self.agent = agent
        self.num_agent = agent.n_agent
        self.device = self.agent.device if hasattr(self.agent, "device") else "cpu"

        if run_args.checkpoint is not None:  # not train from scratch
            self.agent.load_nets(run_args.checkpoint, best=True)
            logger.log(interaction=run_args.start_step)
        self.start_step = run_args.start_step
        self.env_name = input_args.env
        self.algo_name = input_args.algo
        self.n_thread = input_args.n_thread

        # yyx add
        self.best_episode_reward = float('-inf')
        self.best_test_episode_reward = float('-inf')
        self.best_count = 0

        # algorithm arguments
        self.n_iter = alg_args.n_iter
        self.n_inner_iter = alg_args.n_inner_iter
        self.n_warmup = alg_args.n_warmup if not self.run_args.debug else 1
        self.n_model_update = alg_args.n_model_update
        self.n_model_update_warmup = alg_args.n_model_update_warmup if not self.run_args.debug else 1
        self.n_test = alg_args.n_test
        self.test_interval = alg_args.test_interval
        self.rollout_length = alg_args.rollout_length
        self.use_stack_frame = alg_args.use_stack_frame

        # environment initialization
        self.envs_learn = envs_learn
        self.envs_test = envs_test
        self.dummy_env = dummy_env

        # buffer initialization
        self.model_based = alg_args.model_based
        self.model_batch_size = alg_args.model_batch_size
        if self.model_based:
            self.n_traj = alg_args.n_traj
            self.model_traj_length = alg_args.model_traj_length
            self.model_error_thres = alg_args.model_error_thres
            self.model_buffer = ModelBuffer(alg_args.model_buffer_size)
            self.model_update_length = alg_args.model_update_length
            self.model_validate_interval = alg_args.model_validate_interval
            self.model_prob = alg_args.model_prob
        # 一定注意，PPO并不是在每次调用rollout时reset，一次rollout和是否reset没有直接对应关系
        _, self.episode_len = self.envs_learn.reset(), 0
        self.current_obs = self.envs_learn.get_obs_from_outside()
        if hasattr(self.agent, 'reset_memory'):
            self.agent.reset_memory()
        self.agent.train_saved_hard_att = []
        self.infer_times = []
        # 每个环境分别记录episode_reward
        self.episode_reward = np.zeros((self.input_args.n_thread))

        # load pretrained env model when model-based
        self.load_pretrained_model = alg_args.load_pretrained_model
        if self.model_based and self.load_pretrained_model:
            self.agent.load_model(alg_args.pretrained_model)

    def _append_eval_metrics(self, iter_idx, average_ret, average_len, snapshot_metrics, extra_metrics=None):
        snapshot_dir = osp.join(self.run_args.output_dir, 'graph_snapshots')
        os.makedirs(snapshot_dir, exist_ok=True)
        csv_path = osp.join(snapshot_dir, 'eval_metrics.csv')
        file_exists = osp.exists(csv_path)

        global_step = int(self.logger.server.step * self.input_args.n_thread)
        row = {
            'iter': int(iter_idx),
            'global_step': global_step,
            'test_episode_reward': float(average_ret),
            'test_episode_len': float(average_len),
            'graph_tau': float(snapshot_metrics.get('graph_tau', 0.0)),
            'graph_margin': float(snapshot_metrics.get('graph_margin', 0.0)),
            'graph_entropy_soft': float(snapshot_metrics.get('graph_entropy_soft', 0.0)),
            'graph_exact_edge_count': float(snapshot_metrics.get('graph_exact_edge_count', 0.0)),
            'graph_exact_density': float(snapshot_metrics.get('graph_exact_density', 0.0)),
            'graph_delta_exact': float(snapshot_metrics.get('graph_delta_exact', 0.0)),
            'graph_snapshot_file': snapshot_metrics.get('graph_snapshot_file', ''),
        }
        if extra_metrics:
            row.update(extra_metrics)
        fieldnames = list(row.keys())
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def run(self):  # 被launcher.py调用的主循环

        if self.run_args.test:
            if hasattr(self.agent, 'set_training_progress'):
                self.agent.set_training_progress(self.n_iter, self.n_iter)
            self.test(iter_idx=self.n_iter)
            return

        self.routine_count = 0
        self.rr = 0
        # for iter in trange(self.n_iter, desc='rollout env'):
        for iter in range(self.n_iter):
            if hasattr(self.agent, 'set_training_progress'):
                self.agent.set_training_progress(iter, self.n_iter)
            if iter % self.test_interval == 0:
                self.test(iter_idx=iter)
            if iter % 1000 == 0:
                self.agent.save_nets(dir_name=self.run_args.output_dir, iter=iter)  # routine

            trajs = self.rollout_env(iter)
            if self.model_based:
                self.model_buffer.storeTrajs(trajs)
                if iter % 10 == 0:
                    self.updateModel()

            agentInfo = []
            # for inner in trange(self.n_inner_iter, desc='inner-iter updateAgent'):
            for inner in range(self.n_inner_iter):
                if self.model_based and np.random.uniform() < self.model_prob:  # Use the model with a certain probability
                    trajs = self.rollout_model(trajs)
                info = self.agent.updateAgent(trajs)
                agentInfo.append(info)
                if self.agent.checkConverged(agentInfo):
                    break

            if not self.input_args.algo == 'Random':
                self.logger.log(inner_iter=inner + 1, iter=iter)

        if hasattr(self.agent, 'set_training_progress'):
            self.agent.set_training_progress(self.n_iter, self.n_iter)
        self.test(iter_idx=self.n_iter)

    def test(self, iter_idx=None):
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        eval_iter = self.n_iter if iter_idx is None else int(iter_idx)
        returns = []
        lengths = []
        test_metric_history = {key: [] for key in SUMMARY_INFO_KEYS}
        feature_metric_history = {}

        # for i in trange(self.n_test, desc='test'):
        for i in range(self.n_test):
            done, ep_ret, ep_len = False, np.zeros((1,)), 0  # ep_ret改为分threads存
            envs = self.envs_test
            envs.reset()
            s = envs.get_obs_from_outside()
            if hasattr(self.agent, 'reset_memory'):
                self.agent.reset_memory()
            self.agent.test_saved_hard_att = []
            infer_times = []
            while not done:  # 测试时限定一个episode最大为length步
                start = time.time()
                action_out = self.agent.act(s, phase='test')  # shape = (-1, 3)
                end = time.time()
                infer_times.append(end-start)
                # if self.input_args.test: print('inference_time:', end-start)
                # 0221凌晨test改为取概率最大动作
                # action1 = a['branch1'].sample()
                # action2 = a['branch2'].sample()
                if isinstance(action_out, dict) and 'action' in action_out:
                    a = action_out['action']
                else:
                    action1 = action_out['branch1'].probs.argmax(dim=-1)
                    action2 = action_out['branch2'].probs.argmax(dim=-1)
                    a = torch.stack([action1, action2], dim=-1)
                # if len(a.shape) == 2 and a.shape[0] == 1:  # for IA2C and IC3Net 注意：向量环境下这个需要改！
                #     a = a.squeeze(0)
                a = a.detach().cpu().numpy()  # # shape should be (UAV_NUM, )
                s1, r, done, envs_info = envs.step(a.tolist())
                done = done.any()
                if not done:
                    s = s1
                ep_ret += r.sum(axis=-1)  # 对各agent的奖励求和
                ep_len += 1
                self.logger.log(interaction=None)

            episode_metrics = mean_env_info(envs_info)
            for key, value in episode_metrics.items():
                test_metric_history[key].append(value)

            if hasattr(self.agent, 'feature_tracker'):
                feature_stats = self.agent.feature_tracker.get_episode_stats()
                for key, value in feature_stats.items():
                    feature_metric_history.setdefault(key, []).append(float(value))
                self.agent.feature_tracker.reset()

            # Log communication overhead for test episodes
            if hasattr(self.agent, 'comm_tracker'):
                comm_stats = self.agent.comm_tracker.get_episode_stats()
                self.logger.log(
                    test_comm_total_bytes=comm_stats['comm_total_bytes'],
                    test_comm_total_KB=comm_stats['comm_total_KB'],
                    test_comm_bytes_per_step=comm_stats['comm_bytes_per_step'],
                    test_comm_total_messages=comm_stats['comm_total_messages'],
                    test_comm_total_active_edges=comm_stats['comm_total_active_edges'],
                    test_comm_active_edges_per_step=comm_stats['comm_active_edges_per_step'],
                )
                if self.input_args.debug:
                    print(f'[CommTracker] test episode comm: '
                          f"{comm_stats['comm_total_KB']:.2f} KB total, "
                          f"{comm_stats['comm_bytes_per_step']:.0f} bytes/step, "
                          f"{comm_stats['comm_total_messages']} messages")
                self.agent.comm_tracker.reset()

            import matplotlib.pyplot as plt
            if self.input_args.test:
                print('average inference time:', np.mean(infer_times))
                # plt.hist(infer_times, bins=20, range=(0.003, 0.005))
                # plt.show()

            if ep_ret.max() > self.best_test_episode_reward:
                max_id = ep_ret.argmax()
                self.best_test_episode_reward = ep_ret.max()
                best_eval_trajs = self.envs_test.get_saved_trajs()
                poi_aoi_history = self.envs_test.get_poi_aoi_history()
                serves = self.envs_learn.get_serves()
                write_output(envs_info[max_id], self.run_args.output_dir, tag='test')
                adj = np.stack(self.agent.test_saved_hard_att, axis=1)[max_id] if self.input_args.algo in ('G2ANet', 'MetaComm', 'UCSMAPPO') else None
                self.dummy_env.save_trajs_2(
                    best_eval_trajs[max_id], poi_aoi_history[max_id], serves[max_id], phase='test', is_newbest=True, adj=adj)

            returns += [ep_ret.sum()]
            lengths += [ep_len]
        returns = np.stack(returns, axis=0)
        lengths = np.stack(lengths, axis=0)
        test_log_data = {}
        for key, values in test_metric_history.items():
            if values:
                test_log_data[f'test_{key}'] = float(np.mean(values))
        for key, values in feature_metric_history.items():
            if values:
                test_log_data[f'test_{key}'] = float(np.mean(values))
        self.logger.log(test_episode_reward=returns.mean(),
                        test_episode_len=lengths.mean(),
                        test_round=None,
                        **test_log_data)
        average_ret = returns.mean()
        average_len = lengths.mean()
        if hasattr(self.agent, 'save_graph_snapshot'):
            snapshot_metrics = self.agent.save_graph_snapshot(
                self.run_args.output_dir,
                iter_idx=eval_iter,
                global_step=self.logger.server.step * self.input_args.n_thread,
            )
            self.logger.log(
                graph_tau=snapshot_metrics['graph_tau'],
                graph_margin=snapshot_metrics['graph_margin'],
                graph_entropy_soft=snapshot_metrics['graph_entropy_soft'],
                graph_exact_edge_count=snapshot_metrics['graph_exact_edge_count'],
                graph_exact_density=snapshot_metrics['graph_exact_density'],
                graph_delta_exact=snapshot_metrics['graph_delta_exact'],
            )
            extra_eval_metrics = {
                key: value for key, value in test_log_data.items()
                if key.startswith('test_attn_')
                or key.startswith('test_inter_head_')
                or key.startswith('test_obs_rep_')
            }
            self._append_eval_metrics(
                eval_iter,
                average_ret,
                average_len,
                snapshot_metrics,
                extra_metrics=extra_eval_metrics,
            )
        if self.input_args.debug:
            print(f"{self.n_test} episodes average accumulated reward: {average_ret}")

        return average_ret

    def rollout_env(self, iter):  # 与环境交互得到trajs
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        self.routine_count += 1

        trajBuffer = TrajectoryBuffer(device=self.device)
        envs = self.envs_learn
        for t in range(int(self.rollout_length / self.input_args.n_thread)):  # 加入向量环境后，控制总训练步数不变
            s = self.current_obs
            start = time.time()
            action_out = self.agent.act(s, phase='train')
            end = time.time()
            self.infer_times.append(end-start)
            traj_extra = {}
            if isinstance(action_out, dict) and 'action' in action_out and 'logp' in action_out:
                a = action_out['action']
                logp = action_out['logp']
                if 'perm' in action_out:
                    traj_extra['perm'] = action_out['perm']
            else:
                a = []
                logp = []
                for key in ['branch1', 'branch2']:
                    a_tmp = action_out[key].sample()
                    logp_tmp = action_out[key].log_prob(a_tmp)
                    a.append(a_tmp)
                    logp.append(logp_tmp)
                a = torch.stack(a, dim=-1)
                logp = torch.stack(logp, dim=-1)

            a_np = a.detach().cpu().numpy()
            s1, r, done, env_info = envs.step(a_np.tolist())
            self.current_obs = s1
            done = done.any()
            trajBuffer.store(s, a_np, r, s1,
                             np.full((self.n_thread, self.num_agent), done),
                             logp, **traj_extra)
            episode_r = r
            assert episode_r.ndim > 1
            episode_r = episode_r.sum(axis=-1)  # 对各agent奖励求和
            self.episode_reward += episode_r
            self.episode_len += 1
            self.logger.log(interaction=None)

            if done:
                ep_r = self.episode_reward
                if self.input_args.debug:
                    print('train episode reward:', ep_r)
                self.logger.log(mean_episode_reward=ep_r.mean(), episode_len=self.episode_len, episode=None)
                self.logger.log(max_episode_reward=ep_r.max(), episode_len=self.episode_len, episode=None)
                if ep_r.max() > self.best_episode_reward:
                    max_id = ep_r.argmax()
                    self.best_episode_reward = ep_r.max()
                    self.best_count += 1
                    self.agent.save_nets(dir_name=self.run_args.output_dir, is_newbest=True)
                    best_train_trajs = self.envs_learn.get_saved_trajs()
                    poi_aoi_history = self.envs_learn.get_poi_aoi_history()
                    serves = self.envs_learn.get_serves()
                    write_output(env_info[max_id], self.run_args.output_dir)
                    adj = np.stack(self.agent.train_saved_hard_att, axis=1)[max_id] if self.input_args.algo in ('G2ANet', 'MetaComm', 'UCSMAPPO') else None
                    self.dummy_env.save_trajs_2(
                        best_train_trajs[max_id], poi_aoi_history[max_id], serves[max_id], phase='train', is_newbest=True, adj=adj, best_count=self.best_count)


                if self.routine_count // 500 > 0:  # routinely vis (OK)
                    max_id = ep_r.argmax()
                    best_train_trajs = self.envs_learn.get_saved_trajs()
                    poi_aoi_history = self.envs_learn.get_poi_aoi_history()
                    serves = self.envs_learn.get_serves()
                    adj = np.stack(self.agent.train_saved_hard_att, axis=1)[max_id] if self.input_args.algo in ('G2ANet', 'MetaComm', 'UCSMAPPO') else None
                    self.dummy_env.save_trajs_2(
                        best_train_trajs[max_id], poi_aoi_history[max_id], serves[max_id],
                        iter=self.rr*500, phase='train', adj=adj)
                    self.rr += self.routine_count // 500
                    self.routine_count = self.routine_count % 500

                self.logger.log(**mean_env_info(env_info))
                # Log communication overhead to TensorBoard
                if hasattr(self.agent, 'comm_tracker'):
                    comm_stats = self.agent.comm_tracker.get_episode_stats()
                    self.logger.log(
                        comm_total_bytes=comm_stats['comm_total_bytes'],
                        comm_total_KB=comm_stats['comm_total_KB'],
                        comm_bytes_per_step=comm_stats['comm_bytes_per_step'],
                        comm_total_messages=comm_stats['comm_total_messages'],
                        comm_total_active_edges=comm_stats['comm_total_active_edges'],
                        comm_active_edges_per_step=comm_stats['comm_active_edges_per_step'],
                    )
                '''执行env的reset'''
                try:
                    if self.debug:
                        print('average inference time:', np.mean(self.infer_times))
                    _, self.episode_len = self.envs_learn.reset(), 0
                    self.current_obs = self.envs_learn.get_obs_from_outside()
                    if hasattr(self.agent, 'reset_memory'):
                        self.agent.reset_memory()
                    self.agent.train_saved_hard_att = []
                    self.infer_times = []
                    self.episode_reward = np.zeros((self.input_args.n_thread))
                    # Reset communication tracker for next episode
                    if hasattr(self.agent, 'comm_tracker'):
                        self.agent.comm_tracker.reset()
                except Exception as e:
                    raise NotImplementedError

        return trajBuffer.retrieve()

