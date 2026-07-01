# yyx: 和 Mobile_DPPO 的内容保持一致，仅替换通信机制为 CommFormer

import numpy as np
import torch.nn
from gym.spaces import Box
from algorithms.models import MLP
from algorithms.utils import Config


def getArgs(radius_v, radius_pi, env, input_args):
    alg_args = Config()

    # ===================== 基本训练超参数 =====================
    alg_args.n_iter = 5000                # 总训练迭代次数
    alg_args.n_inner_iter = 5
    alg_args.n_warmup = 0
    alg_args.n_model_update = 5
    alg_args.n_model_update_warmup = 10
    alg_args.n_test = 1
    alg_args.test_interval = 20
    alg_args.rollout_length = 600         # PPO horizon
    alg_args.max_episode_len = 600
    alg_args.model_based = False
    alg_args.load_pretrained_model = False
    alg_args.pretrained_model = None
    alg_args.model_batch_size = 128
    alg_args.model_buffer_size = 0
    alg_args.n_traj = 2048
    alg_args.model_traj_length = 8
    alg_args.model_error_thres = 0.0

    # ===================== Agent 参数 =====================
    agent_args = Config()
    agent_args.n_agent = env.UAV_NUM

    from envs.neighbor_graph import get_adj
    agent_args.adj = get_adj(env.UAV_NUM)

    agent_args.gamma = 0.99
    agent_args.lamda = 0.5
    agent_args.clip = 0.2
    agent_args.target_kl = 0.01
    agent_args.v_coeff = 1.0
    agent_args.v_thres = 0.0
    agent_args.entropy_coeff = 0.01

    agent_args.lr = 5e-5
    agent_args.lr_colla = 5e-5
    agent_args.lr_v = 5e-4
    agent_args.n_update_v = 30
    agent_args.n_update_pi = 10
    agent_args.n_minibatch = 1
    agent_args.use_reduced_v = False
    agent_args.use_rtg = True
    agent_args.use_gae_returns = False
    agent_args.advantage_norm = True

    # ===================== 环境信息 =====================
    agent_args.observation_dim = env.observation_space['Box'].shape[1]  # 每个agent的观测维度
    agent_args.action_space = env.action_space
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    agent_args.squeeze = False
    agent_args.p_args = None

    # ===================== 网络结构 =====================
    # Critic 网络
    v_args = Config()
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 64, 64, 1]
    agent_args.v_args = v_args

    # Actor 网络
    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.sizes = [-1, 64, 64]
    pi_args.branchs = [env.action_space[0].n, env.action_space[1].n]
    pi_args.have_last_branch = False
    pi_args.squash = False
    agent_args.pi_args = pi_args

    # ===================== CommFormer 模块专属参数 =====================
    agent_args.commformer_args = Config()
    agent_args.commformer_args.n_block = 2
    agent_args.commformer_args.n_embd = 128
    agent_args.commformer_args.n_head = 2
    agent_args.commformer_args.encode_state = False
    agent_args.commformer_args.action_type = 'Discrete'  # 连续动作环境改为 "Continuous"
    agent_args.commformer_args.dec_actor = False
    agent_args.commformer_args.share_actor = False
    agent_args.commformer_args.sparsity = 0.4
    agent_args.commformer_args.warmup = 10
    agent_args.commformer_args.post_stable = False
    agent_args.commformer_args.post_ratio = 0.5
    agent_args.commformer_args.self_loop_add = True
    agent_args.commformer_args.no_relation_enhanced = False
    agent_args.commformer_args.inter_head_diversity_coef = 0.0
    agent_args.action_dim = sum(pi_args.branchs)

    alg_args.agent_args = agent_args
    return alg_args
