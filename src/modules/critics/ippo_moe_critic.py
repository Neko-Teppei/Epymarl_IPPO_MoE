import torch as th
import torch.nn as nn


class IPPOMoECritic(nn.Module):
    """MoE Critic 物理语义与信用分配设计 (K=8 隐式价值分解)：
    由于 DTDE 范式下缺乏全局状态，本 MoE 架构旨在通过局部观测模式实现全局奖励的隐式信用分配 (Implicit Credit Assignment)。
    8 个专家在 Router 的负载均衡下，自发涌现出以下三种认知分工：
    - 2 个 Ego-Value 专家：专职评估自身局部状态（位置/速度）对全局奖励的独立贡献，不受环境遮挡影响。
    - 4 个 Alter-Value / Intent 专家：专职推断隐变量 Z（队友的隐藏策略与状态）。根据观测信噪比的不同，在清晰与严重遮挡环境下动态拟合队友行为对全局带来的正负向价值。
    - 2 个 Synergy 专家：作为协同模式检测器。专职处理自身与队友特征的非线性耦合，评估复杂互动（如包夹、避障）带来的联合协作增益。
    """

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.num_experts = args.moe_num_experts

        input_shape = self._get_input_shape(scheme)
        hidden_dim = args.rnn_hidden_dim

        # 共享编码器：先抽取共享特征，再交给路由与专家分工
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Router：输出每个专家的权重（后续 softmax）
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_experts),
            nn.Softmax(dim=-1),
        )

        # Experts：每个专家独立预测一个标量 V
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        shared_features = self.shared_encoder(inputs)

        routing_weights = self.router(shared_features)
        expert_vs = th.stack([expert(shared_features) for expert in self.experts], dim=-2)

        # V_total = sum_k w_k * V_k
        v_values = th.sum(routing_weights.unsqueeze(-1) * expert_vs, dim=-2)
        return v_values, routing_weights

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = [batch["obs"][:, ts]]

        if self.args.obs_agent_id:
            agent_ids = (
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bs, max_t, -1, -1)
            )
            inputs.append(agent_ids)

        return th.cat(inputs, dim=-1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
