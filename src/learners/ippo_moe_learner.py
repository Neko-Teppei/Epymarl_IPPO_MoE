import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_registry


class IPPOMoELearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert rewards.size(2) == 1, "Expected common reward with singleton agent dim"
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            old_mac_out.append(self.old_mac.forward(batch, t=t))
        old_pi = th.stack(old_mac_out, dim=1)
        old_pi[mask == 0] = 1.0
        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        is_warmup = t_env < self.args.t_max * 0.05

        pg_loss_value = 0.0
        grad_norm_value = 0.0
        pi_max_value = 0.0
        latest_expert_usage = None
        critic_train_stats = {}
        advantages = None

        for _ in range(self.args.epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                mac_out.append(self.mac.forward(batch, t=t))
            pi = th.stack(mac_out, dim=1)

            advantages, critic_train_stats, routing_weights = self.train_critic_sequential(
                self.critic,
                self.target_critic,
                batch,
                rewards,
                critic_mask,
            )
            advantages = advantages.detach()

            pi[mask == 0] = 1.0
            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = -(((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum())

            if not is_warmup:
                self.agent_optimiser.zero_grad()
                pg_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
                self.agent_optimiser.step()
                grad_norm_value = grad_norm.item()

            pg_loss_value = pg_loss.item()
            pi_max_value = ((pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item())

            with th.no_grad():
                # 统计专家使用率：根据 argmax 路由决策
                num_experts = routing_weights.shape[-1]
                chosen_experts = th.argmax(routing_weights, dim=-1).reshape(-1)
                usage = th.bincount(chosen_experts, minlength=num_experts).float()
                latest_expert_usage = usage / usage.sum().clamp_min(1.0)

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "moe_lb_loss",
                "moe_ent_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("is_warmup", float(is_warmup), t_env)
            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss_value, t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm_value, t_env)
            self.logger.log_stat("pi_max", pi_max_value, t_env)

            if latest_expert_usage is not None:
                num_experts = latest_expert_usage.shape[0]
                for expert_id in range(num_experts):
                    self.logger.log_stat(
                        f"expert_usage_{expert_id}",
                        latest_expert_usage[expert_id].item(),
                        t_env,
                    )

            # 尝试记录每个智能体的奖励和回报，不影响主训练流程
            try:
                per_agent_reward = rewards.mean(dim=(0, 1))
                per_agent_return = rewards.sum(dim=1).mean(dim=0)
                for agent_id in range(self.n_agents):
                    self.logger.log_stat(
                        f"agent_{agent_id}_reward_step",
                        per_agent_reward[agent_id].item(),
                        t_env,
                    )
                    self.logger.log_stat(
                        f"agent_{agent_id}_episode_return",
                        per_agent_return[agent_id].item(),
                        t_env,
                    )
            except Exception:
                pass

            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        with th.no_grad():
            target_vals, _ = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "moe_lb_loss": [],
            "moe_ent_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v_values, routing_weights = critic(batch)
        v = v_values[:, :-1].squeeze(3)
        routing_weights = routing_weights[:, :-1]

        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask

        # PPO critic 主损失：1/2 * (V - R)^2
        v_loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()

        # MoE 负载均衡损失：alpha * sum_k (w_bar_k)^2
        mean_weights = routing_weights.mean(dim=(0, 1, 2))
        lb_loss = self.args.moe_lb_coef * (mean_weights ** 2).sum()

        # Router 熵项：-beta * sum_k w_k log(w_k + eps)，并按 mask 加权
        eps = 1e-8
        router_entropy = -th.sum(routing_weights * th.log(routing_weights + eps), dim=-1)
        router_entropy_loss = -self.args.moe_ent_coef * ((router_entropy * mask).sum() / mask.sum())

        total_critic_loss = v_loss + lb_loss + router_entropy_loss

        self.critic_optimiser.zero_grad()
        total_critic_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(total_critic_loss.item())
        running_log["moe_lb_loss"].append(lb_loss.item())
        running_log["moe_ent_loss"].append(router_entropy_loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(masked_td_error.abs().sum().item() / mask_elems)
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log, routing_weights.detach()

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                if step == nsteps:
                    nstep_return_t += self.args.gamma ** step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), f"{path}/critic.th")
        th.save(self.agent_optimiser.state_dict(), f"{path}/agent_opt.th")
        th.save(self.critic_optimiser.state_dict(), f"{path}/critic_opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(f"{path}/critic.th", map_location=lambda storage, loc: storage)
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(f"{path}/agent_opt.th", map_location=lambda storage, loc: storage)
        )
        self.critic_optimiser.load_state_dict(
            th.load(f"{path}/critic_opt.th", map_location=lambda storage, loc: storage)
        )
