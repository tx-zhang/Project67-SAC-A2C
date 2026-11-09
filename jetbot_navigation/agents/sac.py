"""Continuous SAC: squashed Gaussian, twin Q, target Q and learned entropy."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from .sac_features import HistoryEncoder, RandomNetworkDistillation


@dataclass
class SACConfig:
    observation_dim: int = 23
    action_dim: int = 2
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.2
    history_length: int = 1
    history_stride: int = 1
    recurrent_hidden_dim: int = 64
    rnd_scale: float = 0.0
    rnd_decay_steps: int = 10000000

    def __post_init__(self):
        if min(self.observation_dim, self.action_dim, self.hidden_dim) < 1:
            raise ValueError('network dimensions must be positive')
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError('invalid gamma or tau')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning rate must be finite and positive')
        if not math.isfinite(self.initial_alpha) or self.initial_alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        if min(self.history_length, self.history_stride, self.recurrent_hidden_dim, self.rnd_decay_steps) < 1:
            raise ValueError('history and curiosity sizes must be positive')
        if not math.isfinite(self.rnd_scale) or self.rnd_scale < 0:
            raise ValueError('RND scale must be finite and nonnegative')

    @property
    def replay_observation_dim(self):
        return self.observation_dim * self.history_length


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network = mlp(cfg.observation_dim, cfg.hidden_dim, 2 * cfg.action_dim)
        if cfg.history_length > 1:
            self.network = nn.Sequential(
                HistoryEncoder(cfg.observation_dim, cfg.history_length, cfg.recurrent_hidden_dim),
                mlp(cfg.recurrent_hidden_dim, cfg.hidden_dim, 2 * cfg.action_dim))

    def forward(self, observations, deterministic=False, noise=None):
        mean, log_std = self.network(observations).chunk(2, dim=-1)
        if deterministic:
            return mean.tanh(), None
        distribution = torch.distributions.Normal(mean, log_std.clamp(-20, 2).exp())
        raw = distribution.rsample() if noise is None else mean + distribution.scale * noise
        # Stable log-Jacobian for the tanh transform.
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_probability = (distribution.log_prob(raw) - correction).sum(-1)
        return raw.tanh(), log_probability


class BehaviorNoise:
    """Periodically resample behavior noise; reset environments resample immediately."""
    def __init__(self, num_envs, action_dim, period, device):
        if period < 1:
            raise ValueError('exploration period must be positive')
        self.period = period
        self.values = torch.zeros(num_envs, action_dim, device=device)
        self.kind = None

    def sample(self, episode_lengths, uniform=False):
        refresh = episode_lengths.remainder(self.period) == 0
        if self.kind != uniform:
            refresh = torch.ones_like(refresh)
        count = int(refresh.sum())
        if count:
            shape = (count, self.values.shape[1])
            if uniform:
                self.values[refresh] = torch.rand(shape, device=self.values.device) * 2 - 1
            else:
                self.values[refresh] = torch.randn(shape, device=self.values.device)
        self.kind = uniform
        return self.values


class TwinQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q1 = mlp(cfg.observation_dim + cfg.action_dim, cfg.hidden_dim, 1)
        self.q2 = mlp(cfg.observation_dim + cfg.action_dim, cfg.hidden_dim, 1)
        self.recurrent = cfg.history_length > 1
        if self.recurrent:
            self.q1 = HistoryQ(cfg)
            self.q2 = HistoryQ(cfg)

    def forward(self, observations, actions):
        if self.recurrent:
            return self.q1(observations, actions), self.q2(observations, actions)
        features = torch.cat((observations, actions), dim=-1)
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)


class HistoryQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = HistoryEncoder(cfg.observation_dim, cfg.history_length, cfg.recurrent_hidden_dim)
        self.head = mlp(cfg.recurrent_hidden_dim + cfg.action_dim, cfg.hidden_dim, 1)

    def forward(self, observations, actions):
        return self.head(torch.cat((self.encoder(observations), actions), -1)).squeeze(-1)


def replay_successor(next_observations, terminated, truncated, info):
    """Replace reset observations with the terminal successors."""
    done = terminated | truncated
    return torch.where(done[:, None], info['terminal_observation'], next_observations)


def bellman_target(reward, terminated, next_q, next_log_probability, alpha, gamma):
    # Time limits bootstrap; only physical/task termination stops bootstrap.
    return reward + gamma * (~terminated).float() * (next_q - alpha * next_log_probability)


class ReplayBuffer:
    def __init__(self, capacity, cfg, device='cpu'):
        if capacity < 1:
            raise ValueError('replay capacity must be positive')
        self.capacity, self.position, self.size = capacity, 0, 0
        self.device = torch.device(device)
        self.data = {
            'observations': torch.zeros(capacity, cfg.replay_observation_dim, device=self.device),
            'actions': torch.zeros(capacity, cfg.action_dim, device=self.device),
            'rewards': torch.zeros(capacity, device=self.device),
            'next_observations': torch.zeros(capacity, cfg.replay_observation_dim, device=self.device),
            'terminated': torch.zeros(capacity, dtype=torch.bool, device=self.device),
        }

    @torch.no_grad()
    def add(self, observations, actions, rewards, next_observations, terminated):
        values = dict(observations=observations, actions=actions, rewards=rewards,
                      next_observations=next_observations, terminated=terminated)
        original_count = observations.shape[0]
        count = min(original_count, self.capacity)
        start = (self.position + original_count - count) % self.capacity
        indices = (torch.arange(count, device=self.device) + start) % self.capacity
        for name, value in values.items():
            self.data[name][indices] = value.detach()[-count:].to(self.device)
        self.position = (self.position + original_count) % self.capacity
        self.size = min(self.size + original_count, self.capacity)

    def sample(self, batch_size, device):
        if self.size < batch_size:
            raise ValueError('insufficient replay samples')
        indices = torch.randint(self.size, (batch_size,), device=self.device)
        return {key: value[indices].to(device) for key, value in self.data.items()}

    def state_dict(self):
        return dict(capacity=self.capacity, position=self.position, size=self.size,
                    data={key: value[:self.size].cpu().clone() for key, value in self.data.items()})

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity:
            raise ValueError('replay capacity differs from checkpoint')
        self.position, self.size = state['position'], state['size']
        for key, value in state['data'].items():
            self.data[key][:self.size].copy_(value)


class SAC:
    def __init__(self, cfg, device='cpu'):
        self.cfg, self.device = cfg, torch.device(device)
        self.actor = Actor(cfg).to(self.device)
        self.critic = TwinQ(cfg).to(self.device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.initial_alpha), device=self.device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)
        self.updates = 0
        self.rnd = (RandomNetworkDistillation(cfg.observation_dim, learning_rate=cfg.learning_rate, device=self.device)
                    if cfg.rnd_scale > 0 else None)
        self._compact_grus()

    def _compact_grus(self):
        for network in (self.actor, self.critic, self.target):
            for module in network.modules():
                if isinstance(module, nn.GRU):
                    module.flatten_parameters()

    @torch.no_grad()
    def act(self, observations, deterministic=False, noise=None):
        return self.actor(observations.to(self.device), deterministic, noise)[0]

    def update(self, batch):
        obs, actions = batch['observations'], batch['actions']
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            learning_reward = batch['rewards']
            rnd_bonus = torch.zeros_like(learning_reward)
            if self.rnd is not None:
                weight = self.cfg.rnd_scale * max(0., 1 - self.rnd.transitions.item() / self.cfg.rnd_decay_steps)
                rnd_bonus = weight * self.rnd.bonus(batch['next_observations'][:, -self.cfg.observation_dim:])
                learning_reward = learning_reward + rnd_bonus
            next_actions, next_logp = self.actor(batch['next_observations'])
            q1, q2 = self.target(batch['next_observations'], next_actions)
            target = bellman_target(learning_reward, batch['terminated'],
                                    torch.minimum(q1, q2), next_logp, alpha, self.cfg.gamma)
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.history_length > 1:
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1., error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        sampled_actions, logp = self.actor(obs)
        q1, q2 = self.critic(obs, sampled_actions)
        actor_loss = (alpha * logp - torch.minimum(q1, q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.history_length > 1:
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1., error_if_nonfinite=True)
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (logp.detach() - self.cfg.action_dim)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.cfg.tau)
        self.updates += 1
        metrics = dict(critic_loss=critic_loss.item(), actor_loss=actor_loss.item(),
                       alpha_loss=alpha_loss.item(), alpha=self.log_alpha.exp().item(),
                       entropy=-logp.mean().item(), mean_target=target.mean().item())
        if self.rnd is not None:
            metrics['rnd_bonus'] = rnd_bonus.mean().item()
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f'non-finite SAC update: {metrics}')
        return metrics

    def save(self, path, replay=None, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(schema_version=1, config=asdict(self.cfg), updates=self.updates,
                     actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                     target=self.target.state_dict(), log_alpha=self.log_alpha.detach(),
                     actor_optimizer=self.actor_optimizer.state_dict(),
                     critic_optimizer=self.critic_optimizer.state_dict(),
                     alpha_optimizer=self.alpha_optimizer.state_dict(), metadata=metadata,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        if replay is not None:
            state['replay'] = replay.state_dict()
        if self.rnd is not None:
            state['rnd'] = self.rnd.state_dict()
            state['rnd_optimizer'] = self.rnd.optimizer.state_dict()
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load(self, path, replay=None):
        # Only load your own project checkpoints; payload contains optimizer state.
        # Keep large CPU replay tensors off the GPU when loading a checkpoint.
        state = torch.load(path, map_location='cpu', weights_only=False)
        if asdict(SACConfig(**state['config'])) != asdict(self.cfg):
            raise ValueError('SAC configuration differs from checkpoint')
        if replay is not None and 'replay' not in state:
            raise ValueError('checkpoint has no replay; resume from latest.pt instead')
        for name in ('actor', 'critic', 'target'):
            getattr(self, name).load_state_dict(state[name])
        self.log_alpha.data.copy_(state['log_alpha'])
        for name in ('actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        self.updates = state['updates']
        if self.rnd is not None:
            self.rnd.load_state_dict(state['rnd'])
            self.rnd.optimizer.load_state_dict(state['rnd_optimizer'])
        if replay is not None:
            replay.load_state_dict(state['replay'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if self.device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
        self._compact_grus()
        return state['metadata']
