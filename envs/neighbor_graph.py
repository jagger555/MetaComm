import math
import numpy as np


def get_ring_adj(agent_num):
    if agent_num <= 1:
        return np.zeros((agent_num, agent_num), dtype=np.float32)

    if agent_num == 2:
        return np.array([
            [0, 1],
            [1, 0],
        ], dtype=np.float32)

    if agent_num == 3:
        return np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.float32)

    adj = np.zeros((agent_num, agent_num), dtype=np.float32)
    for i in range(agent_num):
        adj[i][(i + 1) % agent_num] = 1.0
        adj[i][(i - 1 + agent_num) % agent_num] = 1.0
    return adj


def get_grid_adj(agent_num, seed=None):
    if agent_num <= 1:
        return np.zeros((agent_num, agent_num), dtype=np.float32)

    cols = int(math.ceil(math.sqrt(agent_num)))
    rows = int(math.ceil(agent_num / cols))
    occupied_cells = [(r, c) for r in range(rows) for c in range(cols)][:agent_num]

    agent_ids = np.arange(agent_num)
    if seed is not None:
        rng = np.random.RandomState(seed)
        agent_ids = rng.permutation(agent_num)

    cell_to_agent = {
        occupied_cells[idx]: int(agent_ids[idx])
        for idx in range(agent_num)
    }

    adj = np.zeros((agent_num, agent_num), dtype=np.float32)
    for (row, col), src_agent in cell_to_agent.items():
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dst_agent = cell_to_agent.get((row + d_row, col + d_col))
            if dst_agent is not None:
                adj[src_agent, dst_agent] = 1.0
    return adj


def get_adj(agent_num, fully_collect=False):
    if fully_collect:
        return np.ones((agent_num, agent_num), dtype=np.float32)
    return get_ring_adj(agent_num)
