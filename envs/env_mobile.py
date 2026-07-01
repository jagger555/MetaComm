from envs.config_3d import Config

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import math
import warnings
import torch
import seaborn as sns
from matplotlib.colors import ListedColormap
from gym import spaces
import os
import os.path as osp
from env_configs.roadmap_env.roadmap_utils import Roadmap


def get_heatmap(data, path, min, max, test=False):
    # 1. Create figure
    f, ax = plt.subplots(figsize=(6, 6))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap.set_under('lightgray')

    # 2. Plot Heatmap
    # cbar=False: Removes the right-side indicator bar
    sns.heatmap(data, vmin=min, vmax=max, cmap=cmap,
                xticklabels=[], yticklabels=[], cbar=True)

    if not test or not os.path.exists(path):
        # 3. Save Figure
        # bbox_inches='tight', pad_inches=0: Removes the white border/margin
        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    plt.close(f)  # Close the figure to free memory

class EnvMobile():
    ids = ['EnvMobile-v0']

    def __init__(self, env_args, input_args, **kwargs):
        assert input_args.fixed_col_time

        if input_args.test_with_shenbi:
            traj_file = r'F:\PycharmProjects\jsac\DRL_dyna_AoI\runs\0225-KAIST-traj\2023-02-25_16-58-35_KAIST_G2ANetAgent_UAVNum=3_UseSNRMAP_KNN=1.5_\train_saved_trajs\eps_1500.npz'
            self.shenbi_uav_trajs = list(np.load(traj_file)['arr_0'])

        self.config = Config(env_args, input_args)
        self.input_args = input_args
        self.debug = self.input_args.debug
        self.test = self.input_args.test
        self.phase = kwargs['phase']
        self.failure_rate = float(getattr(self.input_args, 'failure_rate', 0.0))
        self.failure_seed = int(getattr(self.input_args, 'failure_seed', 1))
        self.failure_step_min_frac = float(getattr(self.input_args, 'failure_step_min_frac', 0.2))
        self.failure_step_max_frac = float(getattr(self.input_args, 'failure_step_max_frac', 0.3))
        self.failure_rng = np.random.RandomState(self.failure_seed)
        self.failure_episode_idx = 0
        # roadmap
        self.rm = Roadmap(self.input_args.dataset, self.config.dict)

        self.USE_SNRMAP = self.input_args.use_snrmap
        self.KNN_COEFFCICENT = self.input_args.knn_coefficient
        self.MAP_X = self.rm.max_dis_x
        self.MAP_Y = self.rm.max_dis_y
        self.WEIGHTED_MODE = self.config("weighted_mode")
        self.SCALE = self.config("scale")
        self.UAV_NUM = self.config("uav_num")
        self.INITIAL_ENERGY = self.config('initial_energy')
        self.EPSILON = self.config("epsilon")
        self.ACTION_ROOT = self.config("action_root")
        self.MAX_EPISODE_STEP = self.config("max_episode_step")
        self.TIME_SLOT = self.config("time_slot")
        self.TOTAL_TIME = self.MAX_EPISODE_STEP * self.TIME_SLOT
        self.UAV_SPEED = self.config("uav_speed")
        self.POI_VISIBLE_NUM = self.config("poi_visible_num")
        self.W_NOISE = self.config("w_noise")

        self.UPDATE_NUM = self.config("update_num")
        self.COLLECT_RANGE = self.config("collect_range")
        self.POI_NUM = self.config("poi_num")
        self.RATE_THRESHOLD = self.config("RATE_THRESHOLD")
        self.TX_NORM = max(2.0 * self.RATE_THRESHOLD, 1e-8)
        self.AoI_THRESHOLD = self.config("AoI_THRESHOLD")
        self.aoi_vio_penalty_scale = self.config("aoi_vio_penalty_scale")
        self.tx_vio_penalty_scale = self.config("tx_vio_penalty_scale")
        self.UPDATE_USER_NUM = self.config("update_user_num")
        self.USER_DATA_AMOUNT = self.config("user_data_amount")
        self.UAV_HEIGHT = self.config("uav_height")
        self.hao02191630 = self.config("hao02191630")

        self.n_agents = self.UAV_NUM
        self.n_actions = self.ACTION_ROOT
        self.agent_field = self.config("agent_field")

        # 碰撞和出界惩罚参数
        self.COLLISION_TOLERANCE = 5.0   # 碰撞容忍距离（米），下一位置有其他无人机在此范围内即判定占用
        self.COLLISION_PENALTY = 1.0     # 碰撞惩罚（固定值），≈1步good_reward，确保约束有效
        self.BOUNDARY_PENALTY_SCALE = 1.0  # 出界惩罚系数，与碰撞惩罚对齐

        self.MAX_FIY_DISTANCE = self.TIME_SLOT * self.UAV_SPEED / self.SCALE

        # self.OBSTACLE = self.config('obstacle')
        self.OBSTACLE = []

        self._get_energy_coefficient()
        # self.action_space = spaces.Discrete(9)
        self.action_space = spaces.MultiDiscrete([9, self.UPDATE_NUM])

        self.cell_num = input_args.map_size
        
        if self.input_args.algo=='ConvLSTM': self.cell_num = 20

        
        self.cell_span_x = self.MAP_X / self.cell_num
        self.cell_span_y = self.MAP_Y / self.cell_num

        '''these mat is **read-only** 因此可以放在init中 而不必放在reset中每次episode开始时都读'''
        self.poi_mat = self.rm.init_pois(self.MAX_EPISODE_STEP)

        # self.poi_arrival = np.load(os.path.join(data_file_dir, 'arrival.npy'))[:self.POI_NUM, :self.MAX_EPISODE_STEP + 1]  # shape = (33, 121)，其中33是poi数，121是episode的时间步数
        # 每个时间步都生成一个包  # OK
        self.poi_arrival = np.ones((self.POI_NUM, self.MAX_EPISODE_STEP + 1))  # 下标从0到120

        # self.poi_QoS = np.load(os.path.join(data_file_dir, f'QoS{self.MAX_EPISODE_STEP}/poi_QoS{self.input_args.dyna_level}.npy'))

        # self.QoS_MAX, self.QoS_MIN = self.poi_QoS.max(), self.poi_QoS.min()

        # 位置2 + aoi1
        self.poi_property_num = 2 + 1
        info = self.get_env_info()

        obs_dict = {
            'Box': spaces.Box(low=-1, high=1, shape=(self.n_agents, info['obs_shape'])),
            'available_actions': spaces.Box(low=0, high=1, shape=(self.n_agents, self.ACTION_ROOT)),
        }
        self.obs_space = spaces.Dict(obs_dict)
        self.observation_space = self.obs_space

        self.obs = None
        self.stacked_obs = None
        self.reset()

    def reset(self):

        # self.remain_data = [0 for _ in range(self.POI_NUM)]
        self.poi_aoi = [0 for _ in range(self.POI_NUM)]
        # self.poi_aoi_area = [0 for _ in range(self.POI_NUM)]


        self.abilities = []
        self.tx_vio_num = 0  # uavnum
        self.tx_satis_num = 0  # uavnum

        self.uav_trace = [[] for i in range(self.UAV_NUM)]
        self.uav_state = [[] for i in range(self.UAV_NUM)]
        self.uav_energy_consuming_list = [[] for i in range(self.UAV_NUM)]

        self.dead_uav_list = [False for i in range(self.UAV_NUM)]
        self.failed_uav_mask = np.zeros(self.UAV_NUM, dtype=bool)
        self.pending_failed_uav_mask = np.zeros(self.UAV_NUM, dtype=bool)
        self.failure_trigger_step = None
        self.failure_triggered = False

        self.poi_history = [{'pos': None, 'aoi': [0 for _ in range(self.POI_NUM)]}]  # episode结束后，长度为121
        self.serves = np.zeros((self.MAX_EPISODE_STEP+1, self.POI_NUM))
        self.aoi_vio_ratio_list = []  # 当前时间步有多大比例的PoI违反了aoi阈值
        self.tx_satis_ratio_list = []  # 当前时间步有多大比例的被服务aoi满足了data rate阈值
        # 监视下面几个reward乘上ratio之前的尺度
        self.good_reward_list = []
        self.aoi_penalty_reward_list = []
        self.tx_penalty_reward_list = []
        self.tx_reward_list = []
        self.knn_reward_list = []
        self.energy_reward_list = []
        self.collision_reward_list = []
        self.boundary_reward_list = []

        self.aoi_history = []  # 每个时间步的user平均aoi

        self.step_count = 0

        # 之前把这四个元素的初始化放在init中，导致跨episode时没有被reset
        self.uav_energy = np.asarray(
            [self.INITIAL_ENERGY for i in range(self.UAV_NUM)],
            dtype=np.float64)
        self.uav_position = np.asarray(
            [[self.MAP_X / 2, self.MAP_Y / 2] for _ in range(self.UAV_NUM)],
            dtype=np.float32)
        self.poi_position = copy.deepcopy(self.poi_mat[:, 0, :])  # 0意为t=0时poi的初始位置

        self.collision_count = 0
        self.boundary_violation_count = 0

        # 添加初始信息到一些数组中
        for uav_index in range(self.UAV_NUM):
            self.uav_trace[uav_index].append(self.uav_position[uav_index].tolist())

        self._setup_test_failures()


        self.check_arrival()
        # self.cpu_preprocessor.reset()
        self.stacked_obs = [None for _ in range(4)]
        return self.get_obs()

    def render(self, mode='human'):
        pass

    def _setup_test_failures(self):
        self.failed_uav_mask[:] = False
        self.pending_failed_uav_mask[:] = False
        self.failure_trigger_step = None
        self.failure_triggered = False

        test_phase = self.phase == 'test' or bool(getattr(self.input_args, 'test', False))
        if (not test_phase) or self.failure_rate <= 0.0:
            return

        n_fail = int(round(self.UAV_NUM * self.failure_rate))
        n_fail = min(max(n_fail, 0), self.UAV_NUM)
        if n_fail == 0:
            return

        failed_ids = self.failure_rng.choice(self.UAV_NUM, size=n_fail, replace=False)
        self.pending_failed_uav_mask[failed_ids] = True

        min_step = int(self.MAX_EPISODE_STEP * self.failure_step_min_frac)
        max_step = int(self.MAX_EPISODE_STEP * self.failure_step_max_frac)
        min_step = max(0, min(min_step, self.MAX_EPISODE_STEP - 1))
        max_step = max(min_step, min(max_step, self.MAX_EPISODE_STEP - 1))
        self.failure_trigger_step = int(self.failure_rng.randint(min_step, max_step + 1))
        self.failure_episode_idx += 1

    def _activate_test_failures_if_due(self):
        if self.failure_triggered or self.failure_trigger_step is None:
            return
        if self.step_count >= self.failure_trigger_step:
            self.failed_uav_mask |= self.pending_failed_uav_mask
            for uav_id, failed in enumerate(self.failed_uav_mask):
                if failed:
                    self.dead_uav_list[uav_id] = True
            self.failure_triggered = True

    def _is_uav_failed(self, uav_id):
        self._activate_test_failures_if_due()
        return bool(self.failed_uav_mask[uav_id])

    def _human_move(self):
        try:
            self.poi_position = self.poi_mat[:, self.step_count, :].copy()
        except:
            print(1)

    def _uavs_access_users(self, max_access_num):
        max_access_num = np.array(max_access_num) + 1  # +1将0~8映射到1~9
        if self.input_args.always_fixed_antenna02230040 != -1:
            max_access_num = [self.input_args.always_fixed_antenna02230040 for _ in range(self.UAV_NUM)]
        access_lists = []
        for uav_id, uav_pos in enumerate(self.uav_position):
            if self._is_uav_failed(uav_id):
                access_lists.append([])
                continue
            access_list = []
            for poi_id in range(self.POI_NUM):
                data_rate, dis = self._get_data_rate(uav_pos, self.poi_position[poi_id])
                if dis < self.COLLECT_RANGE:
                    access_list.append((poi_id, data_rate, dis))
            access_list.sort(key=lambda x: x[2])
            access_list = access_list[:max_access_num[uav_id]]
            if self.hao02191630:
                access_list = [(x[0], x[1] / max_access_num[uav_id], x[2]) for x in access_list]
            else:
                n_access = len(access_list)
                access_list = [(x[0], x[1] / n_access, x[2]) for x in access_list] if n_access > 0 else []
            access_lists.append(access_list)
        return access_lists

    def shenbi_interact(self, uav_index):
        new_x, new_y = self.shenbi_uav_trajs[uav_index][self.step_count]
        distance = np.linalg.norm(np.array(self.uav_trace[uav_index][-1]) - np.array([new_x, new_y]))
        energy_consume = self._cal_energy_consuming(distance)
        return new_x, new_y, distance, energy_consume

    def step(self, action, collect_time=12.5):
        action1, action2 = [item[0] for item in action], [item[1] for item in action]
        # if self.input_args.test: print(action2)  # 打印接入策略

        # 若max_episode_step=120, 则执行120次step方法。episode结束时保存120个poi和uav的位置点，而不是icde的121个，把poi、uav的初始位置扔掉！
        self.step_count += 1

        # 在当前时间步的收集前统计soft emergency的user数量
        soft_emergency_list = []
        for i in range(self.POI_NUM):
            if self.poi_aoi[i] > 0.8 * self.AoI_THRESHOLD:  # 0.8是soft超参
                soft_emergency_list.append(i)

        # poi移动
        self._human_move()

        energy_rs = np.zeros(self.UAV_NUM)
        boundary_rs = np.zeros(self.UAV_NUM)

        # uav移动 + 碰撞检测（下一位置被占用则不动）
        collision_rs = np.zeros(self.UAV_NUM)
        for uav_index in range(self.UAV_NUM):
            new_x, new_y, distance, energy_consuming = self._cal_uav_next_pos(uav_index, action1[uav_index])  # 调用关键函数，uav移动
            if self.input_args.test_with_shenbi:
                new_x, new_y, distance, energy_consuming = self.shenbi_interact(uav_index)

            # 1. 出界检测
            out_of_bound = self._judge_obstacle(self.uav_position[uav_index], (new_x, new_y))
            # 2. 碰撞检测：检查下一位置是否已被其他无人机占用（小范围容忍）
            occupied = False
            for other_idx in range(self.UAV_NUM):
                if other_idx == uav_index or self._is_uav_failed(other_idx):
                    continue
                ox, oy = self.uav_position[other_idx]
                if abs(new_x - ox) < self.COLLISION_TOLERANCE and abs(new_y - oy) < self.COLLISION_TOLERANCE:
                    occupied = True
                    break

            if out_of_bound:  # 出界：不更新位置，施加惩罚
                if not self._is_uav_failed(uav_index):
                    self.boundary_violation_count += 1
                    boundary_rs[uav_index] -= self.BOUNDARY_PENALTY_SCALE
            elif occupied:  # 碰撞：下一位置有无人机，不移动，施加惩罚
                if not self._is_uav_failed(uav_index):
                    self.collision_count += 1
                    collision_rs[uav_index] -= self.COLLISION_PENALTY
            else:  # 安全：更新位置
                self.uav_position[uav_index] = (new_x, new_y)

            self.uav_trace[uav_index].append(self.uav_position[uav_index].tolist())  # 维护uav_trace
            self._use_energy(uav_index, energy_consuming)  # 这个要体现在罚项中
            energy_rs[uav_index] -= energy_consuming / self.INITIAL_ENERGY  # 总能量

        # uav收集
        ## 确定uav的服务对象，元素是（poi_id，data rate，dis）三元组
        access_lists = self._uavs_access_users(action2)
        for uav_index in range(self.UAV_NUM):
            self._use_energy(uav_index, len(access_lists[uav_index]) * 10)  # 服务每个用户带来10J消耗  # TODO 这个要体现在罚项中
            energy_rs[uav_index] -= len(access_lists[uav_index]) * 10 / self.INITIAL_ENERGY

        ## 计算各poi的总data rate
        sum_rates = np.zeros(self.POI_NUM)

        for uav_id, access_list in enumerate(access_lists):
            for poi_id, rate, dis in access_list:
                sum_rates[poi_id] += rate

        clipped_tx_ratio = np.minimum(sum_rates / max(self.RATE_THRESHOLD, 1e-8), 1.0)
        step_tx_reward = float(np.mean(clipped_tx_ratio))

        ## 若poi的总data rate满足阈值，则更新aoi

        uav_rewards = np.zeros(self.UAV_NUM)
        step_tx_vio = 0
        for poi_id, sum_rate in enumerate(sum_rates):
            if sum_rate < self.RATE_THRESHOLD:
                self.tx_vio_num += 1  # Count every unmet POI, including zero-service users.
                step_tx_vio += 1
                continue  # 不满足阈值
            self.tx_satis_num += 1  # uavnum
            self.serves[self.step_count][poi_id] = 1
            ability = int(collect_time / (self.USER_DATA_AMOUNT / sum_rate))  # int向下取整没问题
            self.abilities.append(ability)  # uavnum
            real = min(self.poi_aoi[poi_id], ability)
            before = self.poi_aoi[poi_id]
            self.poi_aoi[poi_id] -= real  # aoi reset

            ## 计算aoi reset reward和bonus reward
            rate_contribute_to_that_poi = np.zeros(self.UAV_NUM)
            for uav_id, access_list in enumerate(access_lists):
                for pid, rate, dis in access_list:
                    if poi_id == pid:
                        # data coll * aoi vio * credit assignment
                        # r = (before - self.poi_aoi[poi_id]) * (before / self.MAX_EPISODE_STEP) * (rate / sum_rate)
                        r = (real / self.MAX_EPISODE_STEP) * (rate / sum_rate)
                        uav_rewards[uav_id] += r
                        rate_contribute_to_that_poi[uav_id] = rate

        self.good_reward_list.extend(uav_rewards)
        self.energy_reward_list.extend(energy_rs)
        uav_rewards -= energy_rs  # 尺度大概在-0.003所以不用乘scale


        if self.KNN_COEFFCICENT > -1:
            uav_trajectory = []
            for i in range(self.UAV_NUM):
                uav_trajectory.extend(self.uav_trace[i])

            intrinsic_rewards = np.zeros(self.UAV_NUM)
            for uav_index in range(self.UAV_NUM):
                d_map = list(map(
                    lambda x: ((x[0] - self.uav_position[uav_index][0]) ** 2 +
                               (x[1] - self.uav_position[uav_index][1]) ** 2) ** 0.5,
                    uav_trajectory
                ))
                d_map.sort(reverse=False)
                intrinsic_rewards[uav_index] = (
                    np.mean(d_map[:10]) / 1000 * self.KNN_COEFFCICENT
                    if len(d_map) > 0 else 0
                )

            uav_rewards += intrinsic_rewards
            self.knn_reward_list.extend((intrinsic_rewards / self.KNN_COEFFCICENT).tolist())

        # 碰撞惩罚
        uav_rewards += collision_rs
        self.collision_reward_list.extend(collision_rs.tolist())
        # 出界惩罚
        uav_rewards += boundary_rs
        self.boundary_reward_list.extend(boundary_rs.tolist())

        done = self._is_episode_done()
        if not done:
            self.check_arrival()

        '''step2. 维护当前时间步对aoi值的相关统计值'''
        now_aoi = 0  # 当前时间步所有poi的aoi值总和
        em_now = 0
        aoi_list = []  # 当前时间步各poi的aoi值
        for i in range(self.POI_NUM):
            aoi = self.poi_aoi[i]
            if aoi > self.AoI_THRESHOLD:  # 超过了AoI阈值
                em_now += 1
            now_aoi += aoi
            aoi_list.append(aoi)

        self.poi_history.append({
            'pos': self.poi_position.copy(),
            'aoi': np.array(aoi_list)
        })
        self.aoi_history.append(now_aoi / self.POI_NUM)
        self.aoi_vio_ratio_list.append(em_now / self.POI_NUM)
        self.tx_satis_ratio_list.append(1.0 - (step_tx_vio / self.POI_NUM))

        # 惩罚基于当前时刻违反AoIth的user的比例，所有uav的惩罚相同
        if em_now != 0:
            penalty_base = -(em_now / self.POI_NUM)
            penalty_r = np.ones(self.UAV_NUM) * penalty_base * self.aoi_vio_penalty_scale
            self.aoi_penalty_reward_list.extend(np.full(self.UAV_NUM, penalty_base))
            uav_rewards += penalty_r
        else:
            self.aoi_penalty_reward_list.extend(np.zeros(self.UAV_NUM))  # 统计尺度时把episode里没有惩罚的步数也要算上

        # Use a dense throughput reward so the weighting factor changes policy behavior
        # continuously instead of only firing on hard-threshold violations.
        tx_reward_r = np.ones(self.UAV_NUM) * step_tx_reward * self.tx_vio_penalty_scale
        self.tx_reward_list.extend(np.full(self.UAV_NUM, step_tx_reward))
        self.tx_penalty_reward_list.extend(np.zeros(self.UAV_NUM))
        uav_rewards += tx_reward_r

        '''step3. episode结束时的后处理'''
        info = {}
        if done:
            if self.debug:
                print('达到txth的user的平均收集能力', np.mean(self.abilities))
                print('全体user中吞吐量未满足阈值的比例：', self.tx_vio_num/(self.tx_vio_num+self.tx_satis_num))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                info = self.summary_info(info)

        self.get_obs()
        return self.get_obs_from_outside(), uav_rewards, done, info

    def summary_info(self, info):
        t_e = np.sum(np.sum(self.uav_energy_consuming_list))

        episodic_aoi = np.mean(self.aoi_history) if len(self.aoi_history) != 0 else 0.0
        aoi_satis_ratio = sum(1 - np.array(self.aoi_vio_ratio_list)) / self.step_count
        data_satis_ratio = 1 - sum(self.poi_aoi) / (self.POI_NUM * self.MAX_EPISODE_STEP)
        tx_satis_ratio = float(np.mean(self.tx_satis_ratio_list)) if len(self.tx_satis_ratio_list) != 0 else 0.0

        info['episodic_aoi'] = episodic_aoi
        info['aoi_satis_ratio'] = aoi_satis_ratio
        info['data_satis_ratio'] = data_satis_ratio
        info['tx_satis_ratio'] = tx_satis_ratio
        # info['soft_tx_satis_ratio'] = soft_tx_satis_ratio
        info['energy_consuming'] = t_e / 10 ** 6  # 单位：MJ
        # info['energy_consuming_ratio'] = energy_consuming_ratio
        info['QoI'] = min(aoi_satis_ratio, data_satis_ratio) / (t_e / self.UAV_NUM / 10 ** 6)
        info['good_reward'] = float(np.mean(self.good_reward_list)) if len(self.good_reward_list) != 0 else 0.0
        info['aoi_penalty_reward'] = float(np.mean(self.aoi_penalty_reward_list)) if len(self.aoi_penalty_reward_list) != 0 else 0.0
        info['tx_reward'] = float(np.mean(self.tx_reward_list)) if len(self.tx_reward_list) != 0 else 0.0
        info['tx_penalty_reward'] = float(np.mean(self.tx_penalty_reward_list)) if len(self.tx_penalty_reward_list) != 0 else 0.0
        info['knn_reward'] = float(np.mean(self.knn_reward_list)) if len(self.knn_reward_list) != 0 else 0.0
        info['energy_reward'] = float(np.mean(self.energy_reward_list)) if len(self.energy_reward_list) != 0 else 0.0
        info['collision_count'] = self.collision_count
        info['boundary_violation_count'] = self.boundary_violation_count
        info['collision_reward'] = float(np.mean(self.collision_reward_list)) if self.collision_reward_list else 0.0
        info['boundary_reward'] = float(np.mean(self.boundary_reward_list)) if self.boundary_reward_list else 0.0
        info['effective_aoi_task_reward'] = (
            info['good_reward'] + self.aoi_vio_penalty_scale * info['aoi_penalty_reward']
        )
        info['effective_tx_task_reward'] = self.tx_vio_penalty_scale * info['tx_reward']
        info['effective_joint_task_reward'] = (
            info['effective_aoi_task_reward'] + info['effective_tx_task_reward']
        )


        if self.debug: print(info)

        return info

    def _cal_distance(self, pos1, pos2):
        assert len(pos1) == len(
            pos2) == 2, 'cal_distance function only for 2d vector'
        distance = np.sqrt(
            np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2)  # 这里算距离不要再*scale了~pos已经是以米为单位的距离了
            + np.power(self.UAV_HEIGHT, 2)
        )
        return distance

    def _cal_theta(self, pos1, pos2):
        assert len(pos1) == len(
            pos2) == 2, 'cal_theta function only for 3d vector'
        r = np.sqrt(np.power(pos1[0] * self.SCALE - pos2[0] * self.SCALE, 2) + np.power(
            pos1[1] * self.SCALE - pos2[1] * self.SCALE, 2))
        h = self.UAV_HEIGHT
        theta = math.atan2(h, r)
        return theta

    def _cal_energy_consuming(self, move_distance):
        moving_time = move_distance / self.UAV_SPEED
        hover_time = self.TIME_SLOT - moving_time
        return self.Power_flying * moving_time + self.Power_hovering * hover_time

    def _cal_uav_next_pos(self, uav_index, action):
        dx, dy = self._get_vector_by_action(int(action))  # 形如[1.5, 0]或[sqrt(1.5), sqrt(1.5)]
        distance = np.sqrt(np.power(dx * self.SCALE, 2) +
                           np.power(dy * self.SCALE, 2))
        energy_consume = self._cal_energy_consuming(distance)

        if self.uav_energy[uav_index] >= energy_consume:
            new_x, new_y = self.uav_position[uav_index] + [dx * self.SCALE, dy * self.SCALE]
        else:
            new_x, new_y = self.uav_position[uav_index]

        return new_x, new_y, distance, min(self.uav_energy[uav_index], energy_consume)

    def _get_vector_by_action(self, action):
        single = 1.5
        base = single / math.sqrt(2)
        action_table = [
            [0, 0],
            [-base, base],
            [0, single],
            [base, base],
            [-single, 0],
            [single, 0],
            [-base, -base],
            [0, -single],
            [base, -base],

            [0, self.MAX_FIY_DISTANCE],
            [0, -self.MAX_FIY_DISTANCE],
            [self.MAX_FIY_DISTANCE, 0],
            [-self.MAX_FIY_DISTANCE, 0],
        ]
        return action_table[action]

    def _is_uav_out_of_energy(self, uav_index):
        return self.uav_energy[uav_index] < self.EPSILON

    def _is_episode_done(self):
        if self.step_count >= self.MAX_EPISODE_STEP:
            return True
        return False

    def _judge_obstacle(self, cur_pos, next_pos):
        if (0 <= next_pos[0] <= self.MAP_X) and (0 <= next_pos[1] <= self.MAP_Y):
            return False
        else:
            return True

    def _use_energy(self, uav_index, energy_consuming):
        self.uav_energy_consuming_list[uav_index].append(
            min(energy_consuming, self.uav_energy[uav_index]))
        self.uav_energy[uav_index] = max(
            self.uav_energy[uav_index] - energy_consuming, 0)

        if self._is_uav_out_of_energy(uav_index):
            self.dead_uav_list[uav_index] = True
            self.uav_state[uav_index].append(0)
        else:
            self.uav_state[uav_index].append(1)

    def _get_energy_coefficient(self):

        P0 = 58.06  # blade profile power, W
        P1 = 79.76  # derived power, W
        U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
        v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
        d0 = 0.2  # fuselage drag ratio
        rho = 1.225  # density of air,kg/m^3
        s0 = 0.05  # the rotor solidity
        A = 0.503  # the area of the rotor disk, m^2
        Vt = self.config("uav_speed")  # velocity of the UAV,m/s ???

        self.Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                            P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                            0.5 * d0 * rho * s0 * A * Vt ** 3

        self.Power_hovering = P0 + P1

    def _get_data_rate(self, uav_position, poi_position):
        eta = 2
        alpha = 4.88
        beta = 0.43
        distance = self._cal_distance(uav_position, poi_position)
        theta = self._cal_theta(uav_position, poi_position)
        path_loss = 54.05 + 10 * eta * math.log10(distance) + (-19.9) / (1 + alpha * math.exp(-beta * (theta - alpha)))

        # fc= 24
        # path_loss2 = (1 + alpha * math.exp(-beta * (theta - alpha)))*(28.0+22*math.log10(distance)+20*math.log10(fc))+(1-(1 + alpha * math.exp(-beta * (theta - alpha))))*(-17.5+(46-7*math.log10(100))*math.log10(distance)+20*math.log10(4*math.pi*fc/3))
        # print(path_loss,path_loss2)
        w_tx = 20
        w_s_t = w_tx - path_loss - self.W_NOISE
        w_w_s_t = math.pow(10, (w_s_t - 30) / 10)
        bandwidth = 20e6
        data_rate = bandwidth * math.log2(1 + w_w_s_t)
        return data_rate / 1e6, distance  # 返回到外面的data rate的单位是Mbps

    def get_obs_from_outside(self):
        if self.input_args.use_stack_frame:
            return torch.concat(self.stacked_obs, dim=-1)  # shape = (3, obs_dim*4)
        else:
            return self.obs  # shape = (3, obs_dim)

    def get_obs(self, aoi_now=None, aoi_next=None):
        agents_obs = [self.get_obs_agent(i) for i in range(self.UAV_NUM)]  # 每个元素shape = (1715, )
        try:
            agents_obs = np.vstack(agents_obs)  # shape = (3, 1715)
        except:
            pass
        obs_dict = {
            'Box': agents_obs,
            'available_actions': self.get_avail_actions()
        }
        obs_array = np.asarray(obs_dict['Box'], dtype=np.float32)
        self.obs = torch.from_numpy(obs_array)
        if self.step_count == 0:
            self.stacked_obs = [self.obs for _ in range(4)]
        else:
            self.stacked_obs.pop()
            self.stacked_obs.append(self.obs)

        return obs_dict

    def get_obs_agent(self, agent_id):
        # aoi退化为G-A-W 删除了obs的很多维度
        obs = []

        my_pos = self.uav_position[agent_id]
        for i in range(self.UAV_NUM):
            if i == agent_id:
                obs.append(self.uav_position[i][0] / self.MAP_X)
                obs.append(self.uav_position[i][1] / self.MAP_Y)
            else:
                dis = self._cal_distance(self.uav_position[i], my_pos)
                if dis < self.agent_field:
                    obs.append(self.uav_position[i][0] / self.MAP_X)
                    obs.append(self.uav_position[i][1] / self.MAP_Y)
                else:
                    obs.extend([0, 0])

        for poi_index in range(self.POI_NUM):
            dis = self._cal_distance(self.poi_position[poi_index], my_pos)
            if dis >= self.agent_field:
                for _ in range(self.poi_property_num):
                    obs.append(0)
            else:
                obs.append(self.poi_position[poi_index][0] / self.MAP_X)
                obs.append(self.poi_position[poi_index][1] / self.MAP_Y)
                obs.append(self.poi_aoi[poi_index] / 121)

            for t in range(self.input_args.future_obs):
                stub = min(self.step_count + t + 1, self.MAX_EPISODE_STEP)
                next_pos = self.poi_mat[poi_index, stub, :]
                obs.append(next_pos[0] / self.MAP_X)
                obs.append(next_pos[1] / self.MAP_Y)
                delta_step = 121 - self.MAX_EPISODE_STEP
                stub2 = min(delta_step + self.step_count + t + 1, self.MAX_EPISODE_STEP)
                obs.append(self.poi_arrival[poi_index, stub2])

        if self.USE_SNRMAP:
            obs.extend(self._get_snrmap(agent_id))

        obs.append(self.step_count / self.MAX_EPISODE_STEP)
        obs = np.asarray(obs, dtype=np.float32)
        return obs

    def _get_snrmap(self, uav_id):
        snrmap = np.zeros((self.cell_num, self.cell_num))
        for i in range(self.cell_num):
            for j in range(self.cell_num):
                center_x = (i + 0.5) * self.cell_span_x
                center_y = (j + 0.5) * self.cell_span_y
                dist = self._cal_distance(self.uav_position[uav_id], [center_x, center_y])
                if dist < self.agent_field:
                    t = min(self.step_count + 1, self.poi_mat.shape[1] - 1)
                    for poi_index in range(self.POI_NUM):
                        next_pos = self.poi_mat[poi_index, t, :]
                        pi = min(int(next_pos[0] / self.cell_span_x), self.cell_num - 1)
                        pj = min(int(next_pos[1] / self.cell_span_y), self.cell_num - 1)
                        if pi == i and pj == j:
                            if self.poi_aoi[poi_index] > self.AoI_THRESHOLD:
                                snrmap[i, j] += 1
                            else:
                                snrmap[i, j] += self.poi_aoi[poi_index] / self.MAX_EPISODE_STEP

        snrmap = snrmap.reshape(self.cell_num * self.cell_num)

        if self.phase == 'test' and self.input_args.test_save_heatmap:
            heatmap_dir = os.path.join(self.input_args.output_dir, './heatmap')
            if not os.path.exists(heatmap_dir): os.makedirs(heatmap_dir)
            get_heatmap(snrmap.reshape(self.cell_num, self.cell_num),
                        heatmap_dir + '/step_%03d' % (self.step_count) + '.png',
                        min=0, max=5, test=True)

        return snrmap.tolist()

    def get_obs_size(self):
        size = 2 * self.UAV_NUM + self.POI_NUM * self.poi_property_num + 1  # 1是step_count
        # add future obs
        size += self.POI_NUM * self.input_args.future_obs * 3
        # add snr-map
        if self.USE_SNRMAP:
            size += self.cell_num * self.cell_num
        return size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return np.vstack(avail_actions)

    def get_avail_agent_actions(self, agent_id):
        avail_actions = []
        temp_x, temp_y = self.uav_position[agent_id]
        for i in range(self.ACTION_ROOT):
            dx, dy = self._get_vector_by_action(i)
            if not self._judge_obstacle((temp_x, temp_y), (dx + temp_x, dy + temp_y)):
                avail_actions.append(1)
            else:
                avail_actions.append(0)

        return np.array(avail_actions)

    def get_total_actions(self):
        return self.n_actions

    def get_num_of_agents(self):
        return self.UAV_NUM

    def close(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "MAX_EPISODE_STEP": self.MAX_EPISODE_STEP}
        return env_info

    def check_arrival(self, step=0):  # 数据生成
        for i in range(self.POI_NUM):
            self.poi_aoi[i] += 1

    def _plot_aoi_trend(self, poi_index):
        assert len(self.poi_history) == 121
        x = range(121)
        y = [self.poi_history[t]['aoi'][poi_index] for t in range(121)]
        plt.plot(x, y)
        plt.show()


    def _plot_histograms(self, data):
        plt.hist(data, bins=20, rwidth=0.8)
        plt.show()

    def save_trajs_2(self, best_trajs, poi_aoi_history, serves, iter=None,
                     phase='train', is_newbest=False, adj=None, best_count=None):

        save_traj_dir = osp.join(self.input_args.output_dir, f'{phase}_saved_trajs')
        if not osp.exists(save_traj_dir): os.makedirs(save_traj_dir)
        postfix = 'best' if is_newbest else str(iter)
        np.savez(osp.join(save_traj_dir, f'eps_{postfix}.npz'), best_trajs)
        np.savez(osp.join(save_traj_dir, f'eps_{postfix}_aoi.npz'), poi_aoi_history)
        np.savez(osp.join(save_traj_dir, f'eps_{postfix}_serve.npz'), serves)

        if best_count is not None:
            postfix = f'best{best_count}'  # 保存所有比之前更好的模型
            np.savez(osp.join(save_traj_dir, f'eps_{postfix}.npz'), best_trajs)
            np.savez(osp.join(save_traj_dir, f'eps_{postfix}_aoi.npz'), poi_aoi_history)
            np.savez(osp.join(save_traj_dir, f'eps_{postfix}_serve.npz'), serves)

        if adj is not None:
            np.savez(osp.join(save_traj_dir, f'eps_{postfix}_adj.npz'), adj)

        try:
            from tools.post.vis import render_HTML
        except ModuleNotFoundError as exc:
            if exc.name != 'geopandas':
                raise
            if self.debug:
                print('Skip HTML trajectory rendering because geopandas is not installed.')
            return
        render_HTML(self.input_args.output_dir,
                    tag=phase, iter=iter, best=is_newbest, best_count=best_count)
