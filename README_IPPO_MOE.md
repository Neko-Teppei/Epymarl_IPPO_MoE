# EPyMARL IPPO-MoE Critic 改造说明

## 1. 改造目标
本次改造在 **EPyMARL 原生 IPPO** 算法上，新增了一个严格符合 Dec-POMDP 范式的 **MoE Value Critic**：

- 输入严格来自 `scheme["obs"]["vshape"]`（可选拼接 agent id）；
- Critic 是 V 网络，不接收 actions，不计算 Q；
- 保持 On-policy PPO 主流程；
- 新增 MoE 路由负载均衡与熵正则；
- 新增 Actor warmup 机制；
- 保留原有日志体系，可继续使用 TensorBoard。

---

## 2. 主要文件变更

### 新增 1：`src/modules/critics/ippo_moe_critic.py`
新增 `IPPOMoECritic`，结构如下：

1. **Shared Encoder**（两层 MLP）
   - `input_shape -> rnn_hidden_dim -> ReLU -> rnn_hidden_dim -> ReLU`
2. **Router**
   - 输入 shared features，输出 `moe_num_experts` 权重（Softmax）
3. **Experts（ModuleList）**
   - 每个专家输出一个标量 V
4. **融合公式**
   - `V_total = sum_k w_k * V_k`

`forward` 返回：
- `v_values`: 形状 `[bs, t, n_agents, 1]`
- `routing_weights`: 形状 `[bs, t, n_agents, num_experts]`

### 新增 2：`src/learners/ippo_moe_learner.py`
新增 `IPPOMoELearner`，核心逻辑：

1. 继承原 PPO 训练流程（old policy、ratio、clip、entropy）；
2. Critic 损失由三部分组成：
   - `v_loss`：PPO 的 value MSE（含 0.5 系数）；
   - `lb_loss`：路由负载均衡项；
   - `router_entropy_loss`：路由熵正则项；
3. **Actor Warmup**
   - 前 `t_max * 5%` 仅更新 Critic，不更新 Actor；
4. 新增日志：
   - `is_warmup`
   - `critic_loss`
   - `moe_lb_loss`
   - `moe_ent_loss`
   - `expert_usage_i`
   - per-agent reward / return（try-except 保护）

### 修改 3：注册
- `src/modules/critics/__init__.py`
  - 注册 `ippo_moe_critic`
- `src/learners/__init__.py`
  - 注册 `ippo_moe_learner`

### 新增 4：算法配置
- `src/config/algs/ippo_moe.yaml`
  - 提供 IPPO-MoE 配置模板
  - 关键超参数：
    - `moe_num_experts: 8`
    - `moe_lb_coef: 0.05`
    - `moe_ent_coef: 0.005`

### 新增 5：批量实验脚本
- `run_mpe.sh`
  - 一键循环运行 `ippo_moe / ippo / mappo` 三种算法；
  - 默认 3 个随机种子（45/46/47）；
  - 环境使用 EPyMARL 原生 `gymma` + `mpe:SimpleSpread-v3`；
  - 可直接 `chmod +x run_mpe.sh && ./run_mpe.sh` 执行。

---

## 3. 快速使用

> 以下命令仅为示例，请按你的实验脚本入口调整。

### 3.1 训练
```bash
python src/main.py --config=ippo_moe --env-config=mpe
```

### 3.2 开启 TensorBoard（保持原有绘图/日志逻辑）
```bash
tensorboard --logdir results/tb_logs --port 6008
```

浏览器访问：
- <http://localhost:6008/>

### 3.3 关注新增指标
在 TensorBoard 中建议重点观察：
- `critic_loss`
- `moe_lb_loss`
- `moe_ent_loss`
- `expert_usage_0 ... expert_usage_{K-1}`
- `is_warmup`

---

## 4. 维度与实现约束说明

1. 所有输入维度动态读取，不存在环境硬编码维度；
2. 路由权重张量与专家输出张量严格按最后两个维度对齐后加权求和；
3. Critic 与 Actor 的 mask 使用方式与原 PPO 一致，避免无效时间步污染；
4. Warmup 仅影响 Actor `backward/step`，不影响 Critic 更新。

---

## 5. 依赖说明
本次改造**未新增第三方 Python 库依赖**，可直接使用项目原有依赖环境。

若你仍希望固定依赖版本，可沿用项目既有安装方式（如 `requirements.txt` / conda env）。
