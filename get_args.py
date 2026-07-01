import argparse
import secrets


_SEED_MAX = 2 ** 32 - 1
def generate_runtime_seed():
    return secrets.randbelow(_SEED_MAX) + 1


def resolve_seed(input_args):
    if not hasattr(input_args, "seed_base"):
        input_args.seed_base = None if input_args.seed is None else int(input_args.seed)
    if getattr(input_args, "seed_resolved", False):
        return int(input_args.seed)

    explicit_seed = input_args.seed_base
    if explicit_seed is not None:
        explicit_seed = int(explicit_seed)

    if explicit_seed is not None and explicit_seed > 0:
        input_args.seed = explicit_seed
        input_args.seed_source = "manual"
    else:
        input_args.seed = generate_runtime_seed()
        input_args.seed_source = "runtime_random"

    input_args.seed_resolved = True
    return input_args.seed

def parse_args():
    parser = argparse.ArgumentParser()
    # 已经验证这里的参数可被存入params.json
    parser.add_argument('--knn_coefficient', type=float, default=0.1, help='KNN coefficient for G2ANet algorithm')
    parser.add_argument('--debug', action='store_true', default=False, )
    parser.add_argument('--test', action='store_true', default=False, )
    parser.add_argument('--test-with-shenbi', action='store_true')
    parser.add_argument('--test-save-heatmap', action='store_true')
    parser.add_argument('--user', type=str, default='yyx')
    parser.add_argument('--env', type=str, default='Mobile')
    parser.add_argument('--algo', type=str, required=False, default='IPPO', help="algorithm(G2ANet/G2ANet_CommFormer/IC3Net/CPPO/DPPO/IA2C/IPPO/CommNet/MetaComm/UCSMAPPO/Random) ")
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help="device(cpu/cuda:0/cuda:1/...) ")
    parser.add_argument("--dataset", type=str, default='NCSU', choices=['NCSU', 'KAIST', 'purdue', 'beijing', 'sanfrancisco'])
    parser.add_argument("--poi_num", type=int, default=116)  # KAIST
    parser.add_argument("--tag", type=str, default='', help='每个单独实验的备注')
    # dirs
    parser.add_argument("--output_dir", type=str, default='runs/uav_num', help="which fold to save under 'runs/'")
    parser.add_argument('--group', type=str, default='uavnum', help='填写我对一组实验的备注，作用与wandb的group和tb的实验保存路径')
    # system stub
    parser.add_argument('--mute_wandb', default=True, action='store_false')  # 后期服务器联网不稳定 网断了进程就会sleep 不要用wandb！
    # tune agent
    parser.add_argument('--checkpoint', type=str)  # load pretrained model
    parser.add_argument('--n_thread', type=int, default=16)
    parser.add_argument('--n_iter', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for the whole run. Omit or set <= 0 to comm-sample a new seed each run.')
    # tune algo
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_v', type=float)
    parser.add_argument('--lr_colla', type=float)
    parser.add_argument('--use-stack-frame', action='store_true')
    # parser.add_argument('--use_extended_value', action='store_false', help='反逻辑，仅用于DPPO')
    # parser.add_argument('--use-mlp-model', action='store_true', help='将model改为最简单的mlp，仅用于DMPO')
    # parser.add_argument('--multi-mlp', action='store_true', help='在model中分开预测obs中不同类别的信息，仅用于DMPO')
    parser.add_argument('--g2a_hidden_dim', type=int, default=64, help='在model中分开预测obs中不同类别的信息，仅用于DMPO')
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--tau_schedule', type=str, default='fixed',
                        choices=['fixed', 'linear'],
                        help='Temperature schedule for MetaComm Gumbel-Softmax relaxation')
    parser.add_argument('--tau_start', type=float, default=None,
                        help='Starting tau for scheduled runs; defaults to --tau')
    parser.add_argument('--tau_end', type=float, default=None,
                        help='Ending tau for scheduled runs; defaults to --tau')
    parser.add_argument('--tau_anneal_end_iter', type=int, default=0,
                        help='Last outer iteration using annealing; <=0 means use total training iterations')
    parser.add_argument('--metacomm_lr_schedule', type=str, default='linear',
                        choices=['fixed', 'linear'],
                        help='Learning-rate schedule for MetaComm optimizers')
    parser.add_argument('--metacomm_lr_final_scale', type=float, default=0.1,
                        help='Final learning-rate scale for MetaComm relative to the initial lr')
    parser.add_argument('--metacomm_lr_anneal_end_iter', type=int, default=0,
                        help='Last outer iteration using MetaComm lr annealing; <=0 means use total training iterations')
    parser.add_argument('--map_size', type=int, default=6)  # hyper
    parser.add_argument('--g2a_hops', type=int, default=1)  # hyper
    parser.add_argument('--update_colla_by_v_0307', action='store_true')  # hyper

    # tune env
    ## setting
    parser.add_argument('--fixed-range', action='store_false')  # 重要，sensing range现在固定了
    parser.add_argument('--collect_range', type=float, default=500)
    parser.add_argument('--dyna_level', type=str, default='1', help='指明读取不同难度的poi_QoS.npy')
    parser.add_argument('--init_energy', type=float, default=719280)
    parser.add_argument('--w_noise', type=float, default=-107)  # 0222morning determined
    parser.add_argument('--user_data_amount', type=float, default=0.75)
    parser.add_argument('--update_num', type=int, default=10)  # 两个数据集统一将default设为10
    parser.add_argument('--uav_num', type=int, default=5)
    parser.add_argument('--fixed-col-time', action='store_false')
    parser.add_argument('--aoith', default=30, type=int)  # 0222morning determined
    parser.add_argument('--txth', default=3, type=float)
    parser.add_argument('--uav_height', default=100, type=int)
    parser.add_argument('--hao02191630', action='store_false')
    parser.add_argument('--always_fixed_antenna02230040', default=-1, type=int)

    ## MDP
    parser.add_argument('--max_episode_step', type=int, default=120)
    parser.add_argument('--future_obs', type=int, default=0)
    parser.add_argument('--use_snrmap', action='store_true', default=False)
    parser.add_argument('--n_head', type=int, default=-1, help='override MetaComm CommFormer n_head')
    parser.add_argument('--metacomm_head_dim', type=int, default=0,
                        help='If > 0, keep MetaComm per-head width fixed and set n_embd = n_head * metacomm_head_dim.')
    parser.add_argument('--metacomm_inter_head_coef', type=float, default=0.0,
                        help='Auxiliary coefficient that penalizes similar MetaComm attention maps across heads.')
    parser.add_argument('--sparsity', type=float, default=-1.0, help='override MetaComm sparsity (-1 means use config default)')
    parser.add_argument('--metacomm_topology', type=str, default='learned',
                        choices=['learned', 'fixed_grid', 'full'],
                        help='MetaComm communication topology: learned, fixed_grid, or full')
    parser.add_argument('--history_len_L', type=int, default=20, choices=[10, 20, 30, 40],
                        help='UCS GTrXL UAV history observation length L. This is not PPO rollout_length.')
    parser.add_argument('--gtrxl_mem_len', type=int, default=None, choices=[10, 20, 30, 40],
                        help='Alias for --history_len_L.')
    parser.add_argument('--use_transformer', default=True, action=argparse.BooleanOptionalAction,
                        help='Enable GTrXL history encoder for UCSMAPPO.')
    parser.add_argument('--gtrxl_n_layers', type=int, default=2)
    parser.add_argument('--gtrxl_n_heads', type=int, default=2)
    parser.add_argument('--gtrxl_embd_dim', type=int, default=128)
    parser.add_argument('--gtrxl_dropout', type=float, default=0.1)
    parser.add_argument('--gtrxl_bg', type=float, default=2.0)
    parser.add_argument('--gtrxl_gating', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--gtrxl_ln', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_intrinsic', action='store_true', default=False,
                        help='Enable UCS RND intrinsic reward.')
    parser.add_argument('--intrinsic_coef', type=float, default=0.01)
    parser.add_argument('--rnd_output_dim', type=int, default=128)
    parser.add_argument('--rnd_lr', type=float, default=1e-4)
    # 修改后的代码
    # 默认不开启，用户通过命令行显式开启
    parser.add_argument('--high_level_dont_use_snrmap', action='store_true')  #
    parser.add_argument('--high_level_knn_coefficient', type=float, default=-1)  #
    parser.add_argument('--aVPS', type=float, default=0.2)
    parser.add_argument('--tVPS', type=float, default=0.2)
    parser.add_argument(
        '--reward_pref',
        type=float,
        default=None,
        help='Single reward preference in [0, 1]. 0 favors AoI, 1 favors throughput, and overrides aVPS/tVPS with a fixed total weight of 0.4.',
    )
    parser.add_argument('--agent_field', type=float, default=750)
    # Test-phase robustness evaluation
    parser.add_argument('--failure_rate', type=float, default=0.0,
                        help='Target UAV failure rate used only during testing/evaluation.')
    parser.add_argument('--failure_seed', type=int, default=1,
                        help='Random seed for test-phase UAV failure sampling.')
    parser.add_argument('--failure_step_min_frac', type=float, default=0.2,
                        help='Earliest failure trigger step as a fraction of episode length.')
    parser.add_argument('--failure_step_max_frac', type=float, default=0.3,
                        help='Latest failure trigger step as a fraction of episode length.')
    # Autoregressive ordering ablation
    parser.add_argument('--agent_order_mode', type=str, default='fixed',
                        choices=['fixed', 'random', 'custom'],
                        help='Decoder agent ordering: fixed=natural, random=per-step shuffle, custom=seed-based fixed perm')
    parser.add_argument('--agent_order_seed', type=int, default=0,
                        help='Seed for custom agent ordering (0=natural order, >0=permuted)')
    input_args = parser.parse_args()


    if input_args.test_with_shenbi:
        input_args.test = True


    if input_args.algo == 'Random':
        input_args.mute_wandb = True
        input_args.n_thread = 1

    if input_args.high_level_dont_use_snrmap:
        input_args.use_snrmap = False
    if input_args.high_level_knn_coefficient != -1:
        input_args.knn_coefficient = input_args.high_level_knn_coefficient
    if input_args.reward_pref is not None:
        input_args.reward_pref = float(input_args.reward_pref)
        if not 0.0 <= input_args.reward_pref <= 1.0:
            parser.error('--reward_pref must be in [0, 1].')
        total_pref_weight = 0.4
        input_args.effective_aVPS = total_pref_weight * (1.0 - input_args.reward_pref)
        input_args.effective_tVPS = total_pref_weight * input_args.reward_pref
        input_args.aVPS = input_args.effective_aVPS
        input_args.tVPS = input_args.effective_tVPS
    else:
        input_args.effective_aVPS = float(input_args.aVPS)
        input_args.effective_tVPS = float(input_args.tVPS)

    input_args.failure_rate = float(input_args.failure_rate)
    if not 0.0 <= input_args.failure_rate <= 1.0:
        parser.error('--failure_rate must be in [0, 1].')
    input_args.failure_seed = int(input_args.failure_seed)
    input_args.failure_step_min_frac = float(input_args.failure_step_min_frac)
    input_args.failure_step_max_frac = float(input_args.failure_step_max_frac)
    if not 0.0 <= input_args.failure_step_min_frac <= 1.0:
        parser.error('--failure_step_min_frac must be in [0, 1].')
    if not 0.0 <= input_args.failure_step_max_frac <= 1.0:
        parser.error('--failure_step_max_frac must be in [0, 1].')
    if input_args.failure_step_min_frac > input_args.failure_step_max_frac:
        parser.error('--failure_step_min_frac must be <= --failure_step_max_frac.')

    if input_args.debug:
        input_args.group = 'uavnum'
        input_args.n_thread = 2
    input_args.output_dir = f'runs/{input_args.group}'

    if input_args.test:
        input_args.group = 'test'
        input_args.n_thread = 1
        input_args.output_dir = f'{input_args.checkpoint}/test'

    if input_args.gtrxl_mem_len is not None:
        input_args.history_len_L = int(input_args.gtrxl_mem_len)
    input_args.gtrxl_mem_len = int(input_args.history_len_L)

    if input_args.env == 'UCS':
        if input_args.dataset in ('NCSU', 'KAIST', 'purdue'):
            input_args.dataset = 'beijing'
        input_args.dataset = str(input_args.dataset).lower()
        if input_args.uav_num == 5:
            input_args.uav_num = 3
        if input_args.max_episode_step == 120:
            input_args.max_episode_step = 240
        if input_args.user_data_amount == 0.75:
            input_args.user_data_amount = 1.0
        if input_args.agent_field == 750:
            input_args.agent_field = 500
        if input_args.aoith == 30:
            input_args.aoith = 100
        if input_args.txth == 3:
            input_args.txth = 0.05


    if input_args.dataset == 'NCSU':  # 在NCSU的默认值
        if input_args.poi_num == 116:
            input_args.poi_num = 48

    if input_args.tau_start is None:
        input_args.tau_start = float(input_args.tau)
    if input_args.tau_end is None:
        input_args.tau_end = float(input_args.tau)
    if input_args.tau_schedule == 'fixed':
        input_args.tau_start = float(input_args.tau)
        input_args.tau_end = float(input_args.tau)
    input_args.tau = float(input_args.tau)
    input_args.tau_start = float(input_args.tau_start)
    input_args.tau_end = float(input_args.tau_end)
    input_args.tau_anneal_end_iter = int(input_args.tau_anneal_end_iter)
    input_args.metacomm_lr_final_scale = float(input_args.metacomm_lr_final_scale)
    if not 0.0 < input_args.metacomm_lr_final_scale <= 1.0:
        parser.error('--metacomm_lr_final_scale must be in (0, 1].')
    input_args.metacomm_lr_anneal_end_iter = int(input_args.metacomm_lr_anneal_end_iter)
    input_args.metacomm_head_dim = int(input_args.metacomm_head_dim)
    if input_args.metacomm_head_dim < 0:
        parser.error('--metacomm_head_dim must be >= 0.')
    input_args.metacomm_inter_head_coef = float(input_args.metacomm_inter_head_coef)
    if input_args.metacomm_inter_head_coef < 0.0:
        parser.error('--metacomm_inter_head_coef must be >= 0.')

    resolve_seed(input_args)

    if input_args.env == 'UCS':
        env_args = {
            "dataset": input_args.dataset,
            "max_episode_step": input_args.max_episode_step,
            "collect_range": input_args.collect_range,
            "initial_energy": input_args.init_energy,
            "user_data_amount": input_args.user_data_amount,
            "update_num": input_args.update_num,
            "num_uav": input_args.uav_num,
            "emergency_threshold": input_args.aoith,
            "rate_threshold": input_args.txth,
            "agent_field": input_args.agent_field,
            "debug_mode": input_args.debug,
            "test_mode": input_args.test,
            "seed": input_args.seed,
        }
    else:
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
            "aoi_vio_penalty_scale": input_args.effective_aVPS,
            "tx_vio_penalty_scale": input_args.effective_tVPS,
            "hao02191630": input_args.hao02191630,
            "w_noise": input_args.w_noise,
            "agent_field": input_args.agent_field,
        }
        if input_args.poi_num is not None:
            env_args["poi_num"] = input_args.poi_num

    return input_args, env_args
