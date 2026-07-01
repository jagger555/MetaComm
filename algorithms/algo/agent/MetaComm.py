from torch.distributions.categorical import Categorical
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from algorithms.algo.agent.DPPO import DPPOAgent
from algorithms.models import MLP, CategoricalActor
from torch.optim import Adam
from cformer_algorithms.mat.algorithm.cformer import CommFormer
from cformer_algorithms.utils.util import check
from algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, ModelBuffer
from algorithms.comm_tracker import CommunicationTracker
from algorithms.feature_stability_tracker import FeatureStabilityTracker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# MetaCommAgent: 完整集成 CommFormer Encoder-Decoder 的多智能体通信学习
# ==========================================
# 核心功能:
#   1. CommFormer Encoder + 可学习通信拓扑 (Gumbel-Softmax Top-K)
#   2. 关系增强注意力 (GraphTransformerLayer with edge embeddings)
#   3. CommFormer Decoder + 自回归动作生成 (Autoregressive Agent-by-Agent)
#   4. 双分支动作空间: multi-hot 编码 = one_hot(a1, b1_dim) ⊕ one_hot(a2, b2_dim)
#   5. 三个优化器: pi(decoder), v(critic), colla(encoder+edges)
# ==========================================
class MetaCommAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        # 父类 DPPOAgent.__init__ 会创建默认的 actors/vs/optimizers
        super().__init__(logger, device, agent_args, input_args)

        # ==================== CommFormer 参数 ====================
        cf_args = agent_args.commformer_args
        obs_dim = self.observation_dim
        n_agent = self.n_agent
        n_embd = cf_args.n_embd

        self.n_embd = n_embd
        self.n_head = int(cf_args.n_head)
        self.head_dim = int(max(self.n_embd // max(self.n_head, 1), 1))
        self.inter_head_diversity_coef = float(getattr(cf_args, 'inter_head_diversity_coef', 0.0))
        self.branch_dims = agent_args.pi_args.branchs  # e.g. [9, 5]
        self.b1_dim = self.branch_dims[0]               # branch1 (移动方向)
        self.b2_dim = self.branch_dims[1]               # branch2 (接入策略)

        # ==================== CommFormer 模块 (Encoder + Decoder + Edges) ====================
        self.g2a = CommFormer(
            state_dim=obs_dim,
            obs_dim=obs_dim,
            action_dim=self.action_dim,  # = b1_dim + b2_dim
            n_agent=n_agent,
            n_block=cf_args.n_block,
            n_embd=n_embd,
            n_head=cf_args.n_head,
            encode_state=cf_args.encode_state,
            device=device,
            action_type=cf_args.action_type,
            dec_actor=cf_args.dec_actor,
            share_actor=cf_args.share_actor,
            sparsity=cf_args.sparsity,
            self_loop_add=cf_args.self_loop_add,
            no_relation_enhanced=cf_args.no_relation_enhanced,
            tau=self.input_args.tau,
            fixed_edge_mode=getattr(self.input_args, 'metacomm_topology', 'learned'),
            fixed_edge_seed=getattr(self.input_args, 'seed', 0),
        ).to(device)

        # ==================== Critic: 复用 Encoder 内建 Value Head ====================
        # 对齐原始 CommFormer: v_loc = encoder.head(rep)
        # Value loss 梯度流回 Encoder，与 Policy loss 共同优化通信拓扑

        # ==================== 两个优化器 (对齐原始 CommFormer) ====================
        # pi → Decoder 参数
        self.optimizer_pi = Adam(self.g2a.decoder.parameters(), lr=self.lr)
        # colla → Encoder (含 Value Head) + Edges 参数
        colla_params = (
            list(self.g2a.encoder.parameters()) +
            list(self.g2a.edges_embed.parameters()) +
            list(self.g2a.edge_parameters())
        )
        self.optimizer_colla = Adam(colla_params, lr=self.lr_colla)
        self.base_pi_lr = float(self.lr)
        self.base_colla_lr = float(self.lr_colla) if self.lr_colla is not None else None
        self.lr_schedule = getattr(input_args, 'metacomm_lr_schedule', 'linear')
        self.lr_final_scale = float(getattr(input_args, 'metacomm_lr_final_scale', 0.1))
        self.lr_anneal_end_iter = int(getattr(input_args, 'metacomm_lr_anneal_end_iter', 0))
        self.min_pi_lr = self.base_pi_lr * self.lr_final_scale
        self.min_colla_lr = None if self.base_colla_lr is None else self.base_colla_lr * self.lr_final_scale
        self.current_pi_lr = self.base_pi_lr
        self.current_colla_lr = self.base_colla_lr

        # ==================== 通信开销Tracker ====================
        self.comm_tracker = CommunicationTracker()
        self.feature_tracker = FeatureStabilityTracker()

        # ==================== 通信拓扑记录 ====================
        self.train_saved_hard_att = []
        self.test_saved_hard_att = []

        # ==================== warmup / post_stable 调度 ====================
        self.warmup = cf_args.warmup
        self.post_stable = cf_args.post_stable
        self.post_ratio = cf_args.post_ratio
        self.n_iter = 0
        raw_total_iter = int(getattr(input_args, 'n_iter', 5000))
        self.total_iter = raw_total_iter if raw_total_iter > 0 else 5000

        # ==================== tau 璋冨害 ====================
        self.tau_schedule = getattr(input_args, 'tau_schedule', 'fixed')
        self.fixed_tau = float(getattr(input_args, 'tau', 1.0))
        self.tau_start = float(getattr(input_args, 'tau_start', self.fixed_tau))
        self.tau_end = float(getattr(input_args, 'tau_end', self.fixed_tau))
        self.tau_anneal_end_iter = int(getattr(input_args, 'tau_anneal_end_iter', 0))
        self.current_outer_iter = 0
        self.current_tau = self.fixed_tau
        self.last_logged_exact_adj = None

        # ==================== 辅助变量 ====================
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        # ==================== 自回归排列消融 ====================
        self.agent_order_mode = getattr(input_args, 'agent_order_mode', 'fixed')
        self.agent_order_seed = getattr(input_args, 'agent_order_seed', 0)
        # 预计算固定排列 (custom 模式)
        if self.agent_order_mode == 'custom' and self.agent_order_seed > 0:
            rng = np.random.RandomState(self.agent_order_seed)
            perm_np = rng.permutation(self.n_agent)
            self.fixed_perm = torch.from_numpy(perm_np).long().to(device)
        else:
            self.fixed_perm = torch.arange(self.n_agent).long().to(device)
        print(f'[MetaComm] agent_order_mode={self.agent_order_mode}, '
              f'agent_order_seed={self.agent_order_seed}, '
              f'agent_perm={self.fixed_perm.cpu().tolist()}')
        print(f'[MetaComm] topology={self.g2a.fixed_edge_mode}, '
              f'topology_seed={self.g2a.fixed_edge_seed}')
        print(f'[MetaComm] n_embd={self.n_embd}, n_head={self.n_head}, '
              f'head_dim={self.head_dim}, inter_head_diversity_coef={self.inter_head_diversity_coef}')
        print(f'[MetaComm] tau_schedule={self.tau_schedule}, '
              f'tau={self.fixed_tau}, tau_start={self.tau_start}, '
              f'tau_end={self.tau_end}, tau_anneal_end_iter={self.tau_anneal_end_iter}')
        print(f'[MetaComm] lr_schedule={self.lr_schedule}, '
              f'pi_lr_start={self.base_pi_lr}, pi_lr_end={self.min_pi_lr}, '
              f'colla_lr_start={self.base_colla_lr}, colla_lr_end={self.min_colla_lr}, '
              f'lr_anneal_end_iter={self.lr_anneal_end_iter}')

        self.set_training_progress(iter_idx=0, total_iter=self.total_iter)

    def _resolve_linear_schedule(self, iter_idx, start_value, end_value, anneal_end):
        if anneal_end <= 0:
            anneal_end = self.total_iter
        anneal_end = max(int(anneal_end), 1)
        progress = min(max(float(iter_idx), 0.0), float(anneal_end)) / float(anneal_end)
        return start_value + (end_value - start_value) * progress

    def _resolve_tau(self, iter_idx):
        if self.tau_schedule != 'linear':
            return self.fixed_tau
        return self._resolve_linear_schedule(
            iter_idx,
            self.tau_start,
            self.tau_end,
            self.tau_anneal_end_iter,
        )

    def _resolve_lr(self, iter_idx, start_lr, end_lr):
        if start_lr is None or end_lr is None:
            return None
        if self.lr_schedule != 'linear':
            return start_lr
        return self._resolve_linear_schedule(
            iter_idx,
            start_lr,
            end_lr,
            self.lr_anneal_end_iter,
        )

    @staticmethod
    def _set_optimizer_lr(optimizer, lr):
        if optimizer is None or lr is None:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def set_training_progress(self, iter_idx, total_iter=None):
        if total_iter is not None and int(total_iter) > 0:
            self.total_iter = int(total_iter)
        self.current_outer_iter = int(iter_idx)
        self.current_tau = float(self._resolve_tau(self.current_outer_iter))
        self.g2a.tau = self.current_tau
        self.current_pi_lr = float(self._resolve_lr(self.current_outer_iter, self.base_pi_lr, self.min_pi_lr))
        self._set_optimizer_lr(self.optimizer_pi, self.current_pi_lr)
        if self.base_colla_lr is not None:
            self.current_colla_lr = float(
                self._resolve_lr(self.current_outer_iter, self.base_colla_lr, self.min_colla_lr)
            )
            self._set_optimizer_lr(self.optimizer_colla, self.current_colla_lr)
        return self.current_tau

    def _get_exact_graph_snapshot(self):
        if self.g2a.uses_fixed_topology:
            edge_logits = self.g2a.fixed_edges.detach().float()
        else:
            edge_logits = self.g2a.edges.detach().float()
        exact_relations = self.g2a.edge_return(exact=True, topk=-1).detach().float()
        exact_adj = (exact_relations > 0.5).to(torch.int64)
        return edge_logits, exact_adj

    def _compute_graph_metrics(self, edge_logits, exact_adj):
        tau = max(float(self.current_tau), 1e-8)
        probs = torch.softmax(edge_logits / tau, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()

        sorted_logits, _ = torch.sort(edge_logits, dim=-1, descending=True)
        if self.g2a.topk >= edge_logits.size(-1):
            margin = (sorted_logits[:, 0] - sorted_logits[:, -1]).mean().item()
        else:
            margin = (sorted_logits[:, self.g2a.topk - 1] - sorted_logits[:, self.g2a.topk]).mean().item()

        eye = torch.eye(self.n_agent, device=edge_logits.device, dtype=torch.int64)
        offdiag_adj = exact_adj * (1 - eye)
        n_offdiag = max(self.n_agent * (self.n_agent - 1), 1)
        active_edges = int(offdiag_adj.sum().item())
        density = float(active_edges) / float(n_offdiag)

        if self.last_logged_exact_adj is None:
            delta_exact = 0.0
        else:
            delta_exact = float(
                (offdiag_adj != self.last_logged_exact_adj).sum().item() / float(n_offdiag)
            )

        return {
            'graph_tau': float(self.current_tau),
            'graph_margin': float(margin),
            'graph_entropy_soft': float(entropy),
            'graph_exact_edge_count': float(active_edges),
            'graph_exact_density': float(density),
            'graph_delta_exact': float(delta_exact),
        }, offdiag_adj.detach().clone()

    def save_graph_snapshot(self, output_dir, iter_idx, global_step=-1):
        snapshot_dir = os.path.join(output_dir, 'graph_snapshots')
        os.makedirs(snapshot_dir, exist_ok=True)

        edge_logits, exact_adj = self._get_exact_graph_snapshot()
        metrics, offdiag_adj = self._compute_graph_metrics(edge_logits, exact_adj)
        snapshot_path = os.path.join(snapshot_dir, f'iter_{int(iter_idx):05d}.npz')

        np.savez_compressed(
            snapshot_path,
            iter=np.asarray([int(iter_idx)], dtype=np.int64),
            global_step=np.asarray([int(global_step)], dtype=np.int64),
            tau=np.asarray([float(self.current_tau)], dtype=np.float32),
            topk=np.asarray([int(self.g2a.topk)], dtype=np.int64),
            uses_fixed_topology=np.asarray([int(self.g2a.uses_fixed_topology)], dtype=np.int8),
            edge_logits=edge_logits.detach().cpu().numpy().astype(np.float32),
            exact_adj=exact_adj.detach().cpu().numpy().astype(np.int8),
        )

        self.last_logged_exact_adj = offdiag_adj
        metrics['graph_snapshot_file'] = os.path.basename(snapshot_path)
        return metrics

    def _compute_attention_entropy(self, attn_maps):
        entropies = []
        for attn in attn_maps:
            # attn: [tgt_len, src_len, batch, heads]
            probs = attn.permute(2, 3, 0, 1).float().clamp_min(1e-12)
            entropy = -(probs * torch.log(probs)).sum(dim=-1)
            entropies.append(entropy.mean())
        if not entropies:
            return 0.0
        return float(torch.stack(entropies).mean().item())

    def _compute_inter_head_diversity(self, attn_maps):
        diversities = []
        for attn in attn_maps:
            probs = attn.permute(2, 0, 3, 1).float()
            num_heads = probs.size(2)
            if num_heads <= 1:
                diversities.append(torch.zeros((), device=probs.device))
                continue
            head_vectors = F.normalize(probs, p=2, dim=-1)
            cosine = torch.matmul(head_vectors, head_vectors.transpose(-1, -2))
            pair_mask = torch.triu(
                torch.ones(num_heads, num_heads, dtype=torch.bool, device=cosine.device),
                diagonal=1,
            ).view(1, 1, num_heads, num_heads)
            pair_values = cosine.masked_select(pair_mask)
            if pair_values.numel() == 0:
                diversities.append(torch.zeros((), device=probs.device))
            else:
                diversities.append(1.0 - pair_values.mean())
        if not diversities:
            return 0.0
        return float(torch.stack(diversities).mean().item())

    def _compute_inter_head_similarity_loss(self, attn_maps):
        similarities = []
        for attn in attn_maps:
            probs = attn.permute(2, 0, 3, 1).float()
            num_heads = probs.size(2)
            if num_heads <= 1:
                continue
            head_vectors = F.normalize(probs, p=2, dim=-1)
            cosine = torch.matmul(head_vectors, head_vectors.transpose(-1, -2))
            pair_mask = torch.triu(
                torch.ones(num_heads, num_heads, dtype=torch.bool, device=cosine.device),
                diagonal=1,
            ).view(1, 1, num_heads, num_heads)
            pair_values = cosine.masked_select(pair_mask)
            if pair_values.numel() > 0:
                similarities.append(pair_values.mean())
        if not similarities:
            return torch.zeros((), device=self.device)
        return torch.stack(similarities).mean()

    def _compute_obs_rep_effective_rank(self, obs_rep):
        flat = obs_rep.reshape(-1, obs_rep.size(-1)).float()
        if flat.size(0) <= 1:
            return 1.0
        flat = flat - flat.mean(dim=0, keepdim=True)
        singular_vals = torch.linalg.svdvals(flat)
        singular_vals = singular_vals.clamp_min(1e-12)
        probs = singular_vals / singular_vals.sum()
        entropy = -(probs * torch.log(probs)).sum()
        return float(torch.exp(entropy).item())

    def _summarize_encoder_feature_stability(self, obs_rep, attn_maps):
        if not attn_maps:
            return {}
        return {
            'attn_entropy_mean': self._compute_attention_entropy(attn_maps),
            'inter_head_diversity': self._compute_inter_head_diversity(attn_maps),
            'obs_rep_effective_rank': self._compute_obs_rep_effective_rank(obs_rep),
        }

    # ==========================================================
    # 辅助: Encoder 前向 + Edge 学习
    # ==========================================================
    def _coerce_perm(self, perm, batch_size):
        if perm is None:
            perm = self._get_perm()
        perm = torch.as_tensor(perm, dtype=torch.long, device=self.device)
        if perm.dim() == 3 and perm.size(-1) == 1:
            perm = perm.squeeze(-1)
        if perm.dim() == 1:
            perm = perm.unsqueeze(0).expand(batch_size, -1)
        if perm.dim() != 2:
            raise ValueError(f'Unsupported permutation shape: {tuple(perm.shape)}')
        if perm.size(0) == 1 and batch_size != 1:
            perm = perm.expand(batch_size, -1)
        if perm.size(0) != batch_size or perm.size(1) != self.n_agent:
            raise ValueError(
                f'Permutation shape {tuple(perm.shape)} is incompatible with batch={batch_size}, n_agent={self.n_agent}'
            )
        return perm.contiguous()

    def _resolve_perm(self, batch_size, perm=None, require_stored=False):
        if perm is None and require_stored and self.agent_order_mode == 'random':
            raise ValueError('MetaComm random agent order requires storing the rollout permutation per sample.')
        return self._coerce_perm(perm, batch_size)

    def _encode_with_edges(self, obs, use_exact_edges=False, perm=None,
                           capture_feature_stats=False, return_attention=False):
        """
        CommFormer Encoder 前向传播 + edge 学习

        对照原始 CommFormer 三阶段:
          1. warmup (n_iter < warmup): 全连接
          2. 正常训练/评估: 确定性或可微 top-k
          3. 采样与更新使用同一张图，避免 PPO 比率失配
        """
        bs = obs.shape[0]

        if self.g2a.uses_fixed_topology:
            relations = self.g2a.edge_return(exact=True, topk=-1)
        elif self.n_iter < self.warmup:
            # Warmup 强制全连接，避免将浮点 edge logits 直接当作二值邻接与 embedding 索引。
            relations = torch.ones_like(self.g2a.edges)
        elif use_exact_edges:
            relations = self.g2a.edge_return(exact=True, topk=-1)
        else:
            relations = self.g2a.edge_return(exact=False, topk=-1)

        if (not self.g2a.uses_fixed_topology) and self.n_iter < self.warmup:
            relations = torch.ones_like(self.g2a.edges)

        if self.n_iter > int(self.post_ratio * self.total_iter) and self.post_stable:
            relations = self.g2a.edge_return(exact=True, topk=-1)

        relations = relations.unsqueeze(0).expand(bs, -1, -1)
        if perm is not None:
            perm = self._coerce_perm(perm, bs)
            relations = self._perm_relations(relations, perm)

        relations_embed = None
        if not self.g2a.no_relation_enhanced:
            relations_embed = self.g2a.edges_embed(relations.long())

        dec_agent = self.g2a.dec_actor
        feature_stats = None
        attn_maps = None
        need_attention = capture_feature_stats or return_attention
        if need_attention:
            v_loc, obs_rep, attn_maps = self.g2a.encoder(
                obs, obs, relations_embed,
                attn_mask=relations, dec_agent=dec_agent,
                return_attention=True,
            )
            if capture_feature_stats:
                feature_stats = self._summarize_encoder_feature_stability(obs_rep, attn_maps)
        else:
            v_loc, obs_rep = self.g2a.encoder(
                obs, obs, relations_embed,
                attn_mask=relations, dec_agent=dec_agent
            )
        return v_loc, obs_rep, relations, relations_embed, feature_stats, attn_maps

    # ==========================================================
    # 辅助: 构造 multi-hot 动作编码
    # ==========================================================
    def _make_multi_hot(self, a1, a2):
        """
        将双分支动作编码为 multi-hot 向量
        multi_hot = one_hot(a1, b1_dim) ⊕ one_hot(a2, b2_dim)

        Returns: [bs, b1_dim + b2_dim]
        """
        oh1 = F.one_hot(a1.long(), self.b1_dim).float()
        oh2 = F.one_hot(a2.long(), self.b2_dim).float()
        return torch.cat([oh1, oh2], dim=-1)

    # ==========================================================
    # 辅助: 自回归排列相关
    # ==========================================================
    def _get_perm(self):
        """根据 agent_order_mode 返回当前排列"""
        if self.agent_order_mode == 'random':
            return torch.randperm(self.n_agent, device=self.device)
        elif self.agent_order_mode == 'custom':
            return self.fixed_perm
        else:  # fixed
            return torch.arange(self.n_agent, device=self.device)

    def _apply_perm(self, x, perm):
        """沿 agent 维度 (dim=1) 重排，支持每个 batch 样本使用不同排列。"""
        perm = self._coerce_perm(perm, x.size(0))
        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        return x[batch_idx, perm]

    def _apply_inv_perm(self, x, perm):
        """逆排列恢复原始 agent 顺序"""
        perm = self._coerce_perm(perm, x.size(0))
        inv_perm = torch.argsort(perm, dim=-1)
        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        return x[batch_idx, inv_perm]

    def _perm_relations(self, relations, perm):
        """对 [bs, n, n] 的邻接矩阵双向排列"""
        perm = self._coerce_perm(perm, relations.size(0))
        n_agent = relations.size(1)
        perm_row = perm.unsqueeze(-1).expand(-1, -1, n_agent)
        perm_col = perm.unsqueeze(1).expand(-1, n_agent, -1)
        return relations.gather(1, perm_row).gather(2, perm_col)

    def _perm_relations_embed(self, relations_embed, perm):
        """对 [bs, n, n, d] 的关系嵌入双向排列"""
        perm = self._coerce_perm(perm, relations_embed.size(0))
        n_agent = relations_embed.size(1)
        dim = relations_embed.size(-1)
        perm_row = perm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_agent, dim)
        perm_col = perm.unsqueeze(1).unsqueeze(-1).expand(-1, n_agent, -1, dim)
        return relations_embed.gather(1, perm_row).gather(2, perm_col)

    # ==========================================================
    # Decoder 自回归生成动作 (Autoregressive, 推理时用)
    # ==========================================================
    def _autoregressive_act(self, obs_rep, obs, relations_embed, relations,
                             deterministic=False):
        """
        使用 CommFormer Decoder 自回归生成每个 agent 的双分支动作
        同时缓存每个 agent 的 logits, 避免后续重复前向

        Returns:
            output_a1: [bs, n_agent] branch1 动作
            output_a2: [bs, n_agent] branch2 动作
            all_logits: [bs, n_agent, action_dim] 缓存的 logits
        """
        bs = obs.shape[0]
        action_dim = self.action_dim  # b1_dim + b2_dim
        dec_agent = self.g2a.dec_actor

        shifted_action = torch.zeros((bs, self.n_agent, action_dim + 1),
                                     **self.tpdv)
        shifted_action[:, 0, 0] = 1

        output_a1 = torch.zeros((bs, self.n_agent), dtype=torch.long,
                                device=self.device)
        output_a2 = torch.zeros((bs, self.n_agent), dtype=torch.long,
                                device=self.device)
        all_logits = torch.zeros((bs, self.n_agent, action_dim),
                                 device=self.device)

        for i in range(self.n_agent):
            logit = self.g2a.decoder(
                shifted_action, obs_rep, obs,
                relations_embed, attn_mask=relations, dec_agent=dec_agent
            )[:, i, :]  # [bs, action_dim]

            all_logits[:, i, :] = logit

            logit1 = logit[:, :self.b1_dim]
            logit2 = logit[:, self.b1_dim:]

            dist1 = Categorical(logits=logit1)
            dist2 = Categorical(logits=logit2)

            if deterministic:
                a1 = dist1.probs.argmax(dim=-1)
                a2 = dist2.probs.argmax(dim=-1)
            else:
                a1 = dist1.sample()
                a2 = dist2.sample()

            output_a1[:, i] = a1
            output_a2[:, i] = a2

            if i + 1 < self.n_agent:
                shifted_action[:, i + 1, 1:] = self._make_multi_hot(a1, a2)

        return output_a1, output_a2, all_logits

    # ==========================================================
    # 并行 log-prob 计算 (Parallel, 训练时用)
    # ==========================================================
    def _parallel_logp(self, obs_rep, obs, a1_all, a2_all, relations_embed, relations):
        """
        使用 CommFormer Decoder 并行计算所有 agent 的 log-prob (训练时)
        """
        bs = obs.shape[0]
        action_dim = self.action_dim
        dec_agent = self.g2a.dec_actor

        shifted_action = torch.zeros((bs, self.n_agent, action_dim + 1),
                                     **self.tpdv)
        shifted_action[:, 0, 0] = 1

        if self.n_agent > 1:
            for i in range(self.n_agent - 1):
                shifted_action[:, i + 1, 1:] = self._make_multi_hot(
                    a1_all[:, i], a2_all[:, i])

        logit = self.g2a.decoder(
            shifted_action, obs_rep, obs,
            relations_embed, attn_mask=relations, dec_agent=dec_agent
        )  # [bs, n_agent, action_dim]

        logit1 = logit[:, :, :self.b1_dim]
        logit2 = logit[:, :, self.b1_dim:]

        dist1 = Categorical(logits=logit1)
        dist2 = Categorical(logits=logit2)

        lp1 = dist1.log_prob(a1_all.long())
        lp2 = dist2.log_prob(a2_all.long())

        return torch.stack([lp1, lp2], dim=-1)  # [bs, n_agent, 2]

    # ==========================================================
    # 重写 get_networked_s: CommFormer 内部自行处理通信
    # ==========================================================
    def get_networked_s(self, s, which_net, phase=None):
        return s

    # ==========================================================
    # 重写 act(): CommFormer Encoder + Decoder (复用自回归 logits)
    # ==========================================================
    def act(self, s, phase):
        """
        输入: s [bs, n_agent, obs_dim]
        输出:
            - branch1 / branch2: 与其他 agent 兼容的分支分布
            - action: 实际执行的联合动作
            - logp: 与 action 对齐的行为策略 log-prob
            - perm: 本次 rollout 使用的 agent 排列
        """
        with torch.no_grad():
            assert s.dim() == 3
            s = s.to(self.device)
            batch_size = s.size(0)

            # ---- 排列 ----
            perm = self._resolve_perm(batch_size, perm=None)
            s_perm = self._apply_perm(s, perm)

            capture_feature_stats = phase == 'test'
            _, obs_rep, relations, relations_embed, feature_stats, _ = self._encode_with_edges(
                s_perm, use_exact_edges=True, perm=perm, capture_feature_stats=capture_feature_stats)
            if capture_feature_stats:
                self.feature_tracker.record(feature_stats)

            # ---------- 记录通信拓扑 (用原始顺序) ----------
            inv_perm = torch.argsort(perm, dim=-1)
            relations_orig = self._perm_relations(relations, inv_perm)
            adj = (relations_orig[0] > 0.5).int().cpu().numpy()
            adj_expanded = np.tile(adj[None, ...], (batch_size, 1, 1))
            if phase == 'test':
                self.test_saved_hard_att.append(adj_expanded)
            elif phase == 'train':
                self.train_saved_hard_att.append(adj_expanded)

            # ---------- 记录通信开销 (仅限 Rollout 阶段) ----------
            if self.comm_tracker is not None:
                edge_mask = (relations > 0.5).float()
                eye = torch.eye(self.n_agent, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
                edge_mask = edge_mask * (1 - eye)
                n_active_edges = int(edge_mask.sum().item())
                self.comm_tracker.record(
                    n_sender=n_active_edges,
                    n_receiver=1,
                    vector_dim=self.n_embd,
                    precision_bytes=4,
                    active_edges=n_active_edges,
                )

            # ---------- Decoder 自回归 (排列后的顺序) ----------
            deterministic = phase == 'test'
            a1, a2, all_logits = self._autoregressive_act(
                obs_rep, s_perm, relations_embed, relations,
                deterministic=deterministic)

            logit1 = all_logits[:, :, :self.b1_dim]
            logit2 = all_logits[:, :, self.b1_dim:]
            dist1 = Categorical(logits=logit1)
            dist2 = Categorical(logits=logit2)
            logp = torch.stack([
                dist1.log_prob(a1.long()),
                dist2.log_prob(a2.long()),
            ], dim=-1)

            # ---- 逆排列恢复原始 agent 顺序 ----
            a1 = self._apply_inv_perm(a1.unsqueeze(-1), perm).squeeze(-1)
            a2 = self._apply_inv_perm(a2.unsqueeze(-1), perm).squeeze(-1)
            logp = self._apply_inv_perm(logp, perm)
            all_logits = self._apply_inv_perm(all_logits, perm)

            logit1 = all_logits[:, :, :self.b1_dim]
            logit2 = all_logits[:, :, self.b1_dim:]
            action = torch.stack([a1, a2], dim=-1)

            return {
                'branch1': Categorical(logits=logit1),
                'branch2': Categorical(logits=logit2),
                'action': action,
                'logp': logp,
                'perm': perm,
            }

    # ==========================================================
    # 重写 get_logp(): CommFormer Encoder + Decoder 并行计算
    # ==========================================================
    def get_logp(self, s, a, perm=None):
        """
        输入: s [bs, n_agent, obs_dim], a [bs, n_agent, 2]
        输出: log_prob [bs, n_agent, 2]
        """
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)

        while s.dim() <= 2:
            s = s.unsqueeze(0)
            a = a.unsqueeze(0)

        # ---- 排列 ----
        perm = self._resolve_perm(s.size(0), perm=perm, require_stored=True)
        s = self._apply_perm(s, perm)
        a = self._apply_perm(a, perm)

        a1_all = a[:, :, 0]
        a2_all = a[:, :, 1]

        _, obs_rep, relations, relations_embed, _, _ = self._encode_with_edges(
            s, use_exact_edges=True, perm=perm)

        dec_agent = self.g2a.dec_actor
        action_dim = self.action_dim

        shifted_action = torch.zeros(
            (s.shape[0], self.n_agent, action_dim + 1), **self.tpdv)
        shifted_action[:, 0, 0] = 1

        if self.n_agent > 1:
            for i in range(self.n_agent - 1):
                shifted_action[:, i + 1, 1:] = self._make_multi_hot(
                    a1_all[:, i], a2_all[:, i])

        logit = self.g2a.decoder(
            shifted_action, obs_rep, s,
            relations_embed, attn_mask=relations, dec_agent=dec_agent
        )  # [bs, n_agent, action_dim]

        logit1 = logit[:, :, :self.b1_dim]
        logit2 = logit[:, :, self.b1_dim:]

        dist1 = Categorical(logits=logit1)
        dist2 = Categorical(logits=logit2)

        lp1 = dist1.log_prob(a1_all.long())
        lp2 = dist2.log_prob(a2_all.long())

        # ---- 逆排列恢复 ----
        lp_perm = torch.stack([lp1, lp2], dim=-1)  # [bs, n_agent, 2]
        return self._apply_inv_perm(lp_perm, perm)

    # ==========================================================
    # 重写 _evalV(): 用 Encoder 表征 + Critic MLP
    # ==========================================================
    def _evalV(self, s, perm=None, return_attention=False):
        """
        使用 Encoder 内建 Value Head，保持端到端梯度。
        对齐原始 CommFormer: v_loc = encoder.head(rep)
        """
        s = s.to(self.device)
        perm = self._resolve_perm(s.size(0), perm=perm, require_stored=True)
        s_perm = self._apply_perm(s, perm)
        v_loc, _, _, _, _, attn_maps = self._encode_with_edges(
            s_perm,
            use_exact_edges=True,
            perm=perm,
            return_attention=return_attention,
        )
        v_loc = self._apply_inv_perm(v_loc, perm)
        if return_attention:
            return v_loc, attn_maps
        return v_loc  # [bs, n_agent, 1]

    # ==========================================================
    # 重写 updateAgent(): 三个优化器分别更新
    # ==========================================================
    def updateAgent(self, trajs, clip=None):
        time_t = time.time()
        self.logger.log(pi_lr=self.current_pi_lr, colla_lr=self.current_colla_lr)
        if clip is None:
            clip = self.clip
        n_minibatch = self.n_minibatch

        names = Trajectory.base_names()
        if trajs:
            names.extend([name for name in trajs[0].names if name not in names])
        traj_all = {name: [] for name in names}
        for traj in trajs:
            for name in names:
                traj_all[name].append(traj[name])
        traj = {name: torch.stack(value, dim=0) for name, value in traj_all.items()}

        # # ================= 计算并扣除通信惩罚 =================
        # with torch.no_grad():
        #     s_pen = traj['s'].to(self.device)
        #     r_pen = traj['r'].to(self.device)
        #     b_sz, T_sz, n_sz, dim_s_sz = s_pen.shape
        #     s_flat = s_pen.view(b_sz * T_sz, n_sz, dim_s_sz)
        #
        #     # 重新推理出当前整条轨迹的通信拓扑 (为了在 PPO GAE 结算前施加惩罚)
        #     _, relations_for_penalty, _ = self._encode_with_edges(s_flat, use_exact_edges=False)
        #
        #     # 统计有效的物理通信边（>0.5 且去除自环）
        #     edge_mask = (relations_for_penalty > 0.5).float()
        #     eye = torch.eye(n_sz, device=self.device).unsqueeze(0).expand(b_sz * T_sz, -1, -1)
        #     edge_mask = edge_mask * (1 - eye)
        #
        #     n_active_edges = edge_mask.sum(dim=(1, 2))  # [b_sz * T_sz]
        #     n_active_edges = n_active_edges.view(b_sz, T_sz)
        #
        #     # 兼容 r 的广播维度
        #     while n_active_edges.dim() < r_pen.dim():
        #         n_active_edges = n_active_edges.unsqueeze(-1)
        #
        #     # 获取 comm_penalty 系数，没有显式设置则默认为 0.005
        #     comm_penalty_coeff = getattr(self.input_args, 'comm_penalty', 0.005)
        #
        #     # 惩罚全局奖励并写回 CPU 字典供后续处理无缝调用
        #     traj['r'] = (r_pen - comm_penalty_coeff * n_active_edges).cpu()
        # # ====================================================

        # warmup 计数
        self.n_iter += 1

        for i_update in range(self.n_update_pi):
            s, a, r, s1, d, logp = traj['s'], traj['a'], traj['r'], traj['s1'], traj['d'], traj['logp']
            s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
            perm = traj.get('perm')
            if perm is not None:
                perm = perm.to(self.device).long()
                if perm.dim() == 4 and perm.size(-1) == 1:
                    perm = perm.squeeze(-1)

            value_old, returns, advantages, _ = self._process_traj(
                s=traj['s'],
                a=traj['a'],
                r=traj['r'],
                s1=traj['s1'],
                d=traj['d'],
                logp=traj['logp'],
                perm=traj.get('perm'),
            )
            advantages_old = advantages

            _, T, n, d_s = s.size()
            d_a = a.size()[-1]
            s = s.view(-1, n, d_s)
            a = a.view(-1, n, d_a)
            logp = logp.view(-1, n, d_a)
            advantages_old = advantages_old.view(-1, n, 1)
            returns = returns.view(-1, n, 1)
            value_old = value_old.view(-1, n, 1)
            perm_flat = None if perm is None else perm.reshape(-1, n)

            batch_total = logp.size()[0]
            batch_size = int(batch_total / n_minibatch)

            kl_all = []
            for i_pi in range(1):
                batch_state, batch_action, batch_logp, batch_advantages_old = [s, a, logp, advantages_old]
                batch_returns = returns
                batch_perm = perm_flat

                if n_minibatch > 1:
                    idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
                    [batch_state, batch_action, batch_logp, batch_advantages_old, batch_returns] = \
                        [item[idxs] for item in [batch_state, batch_action, batch_logp, batch_advantages_old, batch_returns]]
                    if batch_perm is not None:
                        batch_perm = batch_perm[idxs]

                # ===== Policy loss (通过 Decoder + Encoder) =====
                batch_logp_new = self.get_logp(batch_state, batch_action, perm=batch_perm)

                logp_diff = batch_logp_new.sum(-1, keepdim=True) - batch_logp.sum(-1, keepdim=True)
                kl = logp_diff.mean()
                ratio = torch.exp(logp_diff)
                surr1 = ratio * batch_advantages_old
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages_old
                loss_surr = torch.min(surr1, surr2).mean()
                loss_entropy = -torch.mean(batch_logp_new)
                loss_pi = -loss_surr - self.entropy_coeff * loss_entropy

                # ===== Value loss (通过 Encoder 内建 Head，梯度流回 Encoder) =====
                head_similarity_loss = torch.zeros((), device=self.device)
                if self.inter_head_diversity_coef > 0.0 and self.n_head > 1:
                    batch_v_new, attn_maps = self._evalV(
                        batch_state,
                        perm=batch_perm,
                        return_attention=True,
                    )
                    head_similarity_loss = self._compute_inter_head_similarity_loss(attn_maps)
                else:
                    batch_v_new = self._evalV(batch_state, perm=batch_perm)
                loss_v = ((batch_v_new - batch_returns) ** 2).mean()

                # ===== 合并 loss：Policy + Value → Encoder 同时接收两个优化信号 =====
                loss_total = (
                    loss_pi
                    + self.v_coeff * loss_v
                    + self.inter_head_diversity_coef * head_similarity_loss
                )

                self.optimizer_colla.zero_grad()
                self.optimizer_pi.zero_grad()
                loss_total.backward()
                self.optimizer_colla.step()
                self.optimizer_pi.step()

                var_v = ((batch_returns - batch_returns.mean()) ** 2).mean()
                rel_v_loss = loss_v / (var_v + 1e-8)
                head_similarity_value = float(head_similarity_loss.detach().item())
                self.logger.log(surr_loss=loss_surr, entropy=loss_entropy, kl_divergence=kl,
                               v_loss=loss_v, v_var=var_v, rel_v_loss=rel_v_loss,
                               head_similarity_loss=head_similarity_value,
                               inter_head_diversity=max(0.0, 1.0 - head_similarity_value),
                               head_diversity_coef=self.inter_head_diversity_coef, pi_update=None)
                kl_all.append(kl.abs().item())
                if self.target_kl is not None and kl.abs() > 1.5 * self.target_kl:
                    break
            self.logger.log(pi_update_step=i_update)
            self.logger.log(update=None, reward=r, value=value_old, clip=clip,
                           returns=returns, advantages=advantages_old.abs())

        self.logger.log(agent_update_time=time.time() - time_t)
        return [r.mean().item(), loss_entropy.item(), max(kl_all)]

    def checkConverged(self, ls_info):
        return False

    # ==========================================================
    # 重写 _process_traj: 使用 MetaComm 的 reduced_advantages
    # ==========================================================
    def _process_traj(self, s, a, r, s1, d, logp, perm=None):
        b, T, n, dim_s = s.shape
        s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
        if perm is not None:
            perm = torch.as_tensor(perm, dtype=torch.long, device=self.device)
            if perm.dim() == 4 and perm.size(-1) == 1:
                perm = perm.squeeze(-1)

        with torch.no_grad():
            perm_flat = None if perm is None else perm.reshape(-1, n)
            value = self._evalV(s.view(-1, n, dim_s), perm=perm_flat).view(b, T, n, -1)
            returns = torch.zeros(value.size(), device=self.device)
            deltas, advantages = torch.zeros_like(returns), torch.zeros_like(returns)
            last_perm = None if perm is None else perm.select(1, T - 1)
            prev_value = self._evalV(s1.select(1, T - 1), perm=last_perm)
            if not self.use_rtg:
                prev_return = prev_value
            else:
                prev_return = torch.zeros_like(prev_value)
            prev_advantage = torch.zeros_like(prev_return)
            d_mask = d.float()
            for t in reversed(range(T)):
                deltas[:, t] = r.select(1, t) + self.gamma * (1 - d_mask.select(1, t)) * prev_value - value.select(1, t)
                advantages[:, t] = deltas.select(1, t) + self.gamma * self.lamda * (1 - d_mask.select(1, t)) * prev_advantage
                if self.use_gae_returns:
                    returns[:, t] = value.select(1, t) + advantages.select(1, t)
                else:
                    returns[:, t] = r.select(1, t) + self.gamma * (1 - d_mask.select(1, t)) * prev_return
                prev_return = returns.select(1, t)
                prev_value = value.select(1, t)
                prev_advantage = advantages.select(1, t)
            if self.advantage_norm:
                advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (advantages.std(dim=1, keepdim=True) + 1e-5)
        return value, returns, advantages, None

    def save_nets(self, dir_name, iter=0, is_newbest=False, **kwargs):
        import os

        model_dir = os.path.join(dir_name, 'Models')
        os.makedirs(model_dir, exist_ok=True)
        prefix = 'best' if is_newbest else str(iter)

        torch.save(self.g2a.state_dict(), os.path.join(model_dir, f'{prefix}_g2a.pt'))

    def load_nets(self, dir_name, iter=0, best=False, is_newbest=False, **kwargs):
        import os

        prefix = 'best' if (best or is_newbest) else str(iter)
        candidates = [
            os.path.join(dir_name, 'Models', f'{prefix}_g2a.pt'),
            os.path.join(dir_name, f'{prefix}_g2a.pt'),
        ]

        for g2a_path in candidates:
            if os.path.exists(g2a_path):
                self.g2a.load_state_dict(torch.load(g2a_path, map_location=self.device))
                return

        raise FileNotFoundError(
            f'Cannot find MetaComm checkpoint for prefix={prefix} under {dir_name}'
        )

