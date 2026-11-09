"""GRU models and Isaac time-limit handling for skrl's recurrent A2C."""
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from skrl.agents.torch.a2c import A2C_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


@dataclass
class A2CConfig:
    hidden_dim: int = 64
    head_dim: int = 128
    rollouts: int = 512
    sequence_length: int = 128
    learning_rate: float = 0.0003
    gamma: float = 0.999
    gae_lambda: float = 0.97
    entropy_scale: float = 0.01
    initial_log_std: float = -0.7
    gsde: bool = False
    noise_period: int = 30

    def __post_init__(self):
        if min(self.hidden_dim, self.head_dim, self.rollouts, self.sequence_length) < 1:
            raise ValueError('network and rollout sizes must be positive')
        if self.rollouts % self.sequence_length:
            raise ValueError('rollouts must be divisible by sequence_length')
        if self.noise_period < 1:
            raise ValueError('noise_period must be positive')


class RecurrentModel(Model):
    def __init__(self, config, num_envs, device, output_dim):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (23,), np.float32)
        action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
        Model.__init__(self, observation_space, action_space, device)
        self.config, self.num_envs = config, num_envs
        self.gru = nn.GRU(23, config.hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(config.hidden_dim, config.head_dim), nn.ELU(),
                                  nn.Linear(config.head_dim, config.head_dim), nn.ELU(),
                                  nn.Linear(config.head_dim, output_dim))
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01 if output_dim == 2 else 1.)

    def get_specification(self):
        return {'rnn': {'sequence_length': self.config.sequence_length,
                        'sizes': [(1, self.num_envs, self.config.hidden_dim)]}}

    def recurrent_features(self, inputs):
        states, hidden = inputs['states'], inputs['rnn'][0]
        if not self.training:
            features, hidden = self.gru(states.unsqueeze(1), hidden)
            return features[:, 0], hidden
        length = self.config.sequence_length
        sequences = states.reshape(-1, length, 23)
        hidden = hidden.reshape(1, -1, length, self.config.hidden_dim)[:, :, 0].contiguous()
        done = inputs['terminated'].reshape(-1, length)
        # A terminal observation belongs to its old episode; reset before the next frame.
        boundaries = (done[:, :-1].any(dim=0).nonzero().flatten() + 1).tolist()
        outputs, start = [], 0
        for end in boundaries + [length]:
            features, hidden = self.gru(sequences[:, start:end], hidden)
            outputs.append(features)
            hidden = hidden * (~done[:, end - 1]).reshape(1, -1, 1)
            start = end
        return torch.cat(outputs, dim=1).reshape(-1, self.config.hidden_dim), hidden


class Policy(GaussianMixin, RecurrentModel):
    def __init__(self, config, num_envs, device):
        RecurrentModel.__init__(self, config, num_envs, device, 2)
        # Store the Gaussian latent action/log probability; clip only at the simulator boundary.
        GaussianMixin.__init__(self, clip_actions=False, min_log_std=-5., max_log_std=1.)
        self.log_std = nn.Parameter(torch.full((2,), config.initial_log_std))
        self.to(device)

    def compute(self, inputs, role):
        features, hidden = self.recurrent_features(inputs)
        return self.head(features), self.log_std, {'rnn': [hidden]}


class Value(DeterministicMixin, RecurrentModel):
    def __init__(self, config, num_envs, device):
        RecurrentModel.__init__(self, config, num_envs, device, 1)
        DeterministicMixin.__init__(self)
        self.to(device)

    def compute(self, inputs, role):
        features, hidden = self.recurrent_features(inputs)
        return self.head(features), {'rnn': [hidden]}


class GsdePolicy(Policy):
    """SB3 gSDE exploration with independent noise matrices for each environment."""
    def __init__(self, config, num_envs, device):
        from stable_baselines3.common.distributions import StateDependentNoiseDistribution
        super().__init__(config, num_envs, device)
        self.noise = StateDependentNoiseDistribution(2, full_std=True, learn_features=False)
        mean_head, self.log_std = self.noise.proba_distribution_net(
            config.head_dim, log_std_init=config.initial_log_std, latent_sde_dim=23)
        nn.init.orthogonal_(mean_head.weight, gain=0.01)
        nn.init.zeros_(mean_head.bias)
        self.head[-1] = mean_head
        self.to(device)
        self.reset_noise()

    @torch.no_grad()
    def reset_noise(self, done=None):
        previous = self.noise.exploration_matrices if done is not None else None
        self.noise.sample_weights(self.log_std.clamp(-5, 1), self.num_envs)
        if done is not None:
            mask = done.flatten()
            previous[mask] = self.noise.exploration_matrices[mask]
            self.noise.exploration_matrices = previous
        # SB3's single-environment path uses exploration_mat instead of the batch.
        self.noise.exploration_mat = self.noise.exploration_matrices[0]

    def act(self, inputs, role=''):
        mean, _, outputs = self.compute(inputs, role)
        # Fixed normalized sensor features keep initial noise near the Gaussian baseline scale.
        features = torch.nn.functional.normalize(inputs['states'], dim=-1)
        distribution = self.noise.proba_distribution(mean, self.log_std.clamp(-5, 1), features)
        actions = inputs.get('taken_actions')
        if actions is None:
            actions = distribution.sample()
        return actions, distribution.log_prob(actions).unsqueeze(-1), outputs

    def distribution(self, role=''):
        return self.noise.distribution

    def get_entropy(self, role=''):
        return self.noise.distribution.entropy()


class IsaacA2C(A2C_RNN):
    def __init__(self, config, num_envs, device='cpu', training=True):
        self.config = config
        self.num_envs = num_envs
        self.updates, self.steps, self.metrics = 0, 0, {}
        policy = (GsdePolicy if config.gsde else Policy)(config, num_envs, device)
        value = Value(config, num_envs, device)
        memory = RandomMemory(config.rollouts, num_envs, device) if training else None
        cfg = dict(rollouts=config.rollouts, mini_batches=1,
                   discount_factor=config.gamma, **{'lambda': config.gae_lambda},
                   learning_rate=config.learning_rate, entropy_loss_scale=config.entropy_scale,
                   grad_norm_clip=0.5, time_limit_bootstrap=False,
                   experiment=dict(write_interval=0, checkpoint_interval=0))
        super().__init__(models={'policy': policy, 'value': value}, memory=memory,
                         observation_space=policy.observation_space,
                         action_space=policy.action_space, device=device, cfg=cfg)
        # skrl 1.4.3 reads this attribute even when no scheduler was configured.
        self.scheduler = None
        self.init({'timesteps': 0})

    @torch.no_grad()
    def act(self, states, timestep, timesteps):
        if self.config.gsde and (self._rollout % self.config.rollouts == 0
                                 or timestep % self.config.noise_period == 0):
            self.policy.reset_noise()
        return super().act(states, timestep, timesteps)

    @torch.no_grad()
    def record_transition(self, states, actions, rewards, next_states, terminated, truncated,
                          infos, timestep, timesteps):
        timeout = truncated & ~terminated
        if timeout.any():
            _, _, outputs = self.value.act(
                {'states': states, 'rnn': self._rnn_initial_states['value']}, role='value')
            terminal_value, _, _ = self.value.act(
                {'states': infos['terminal_observation'], 'rnn': outputs['rnn']}, role='value')
            # GAE cuts at reset; only timeouts receive the actual terminal successor value.
            rewards = rewards + self.config.gamma * torch.where(timeout, terminal_value, 0.)
        super().record_transition(states, actions, rewards, next_states, terminated, truncated,
                                  infos, timestep, timesteps)
        if self.config.gsde and (terminated | truncated).any():
            self.policy.reset_noise(terminated | truncated)

    def track_data(self, tag, value):
        self.metrics[tag] = float(value)

    def _update(self, timestep, timesteps):
        super()._update(timestep, timesteps)
        self.updates += 1

    @torch.no_grad()
    def evaluate(self, states):
        mean, _, outputs = self.policy.compute(
            {'states': states, 'rnn': self._rnn_initial_states['policy']}, role='policy')
        self._rnn_initial_states['policy'] = outputs['rnn']
        return mean.clamp(-1, 1)

    def reset_evaluation(self, done):
        for hidden in self._rnn_initial_states['policy']:
            hidden[:, done.flatten()] = 0

    def save(self, path, contract):
        if self._rollout % self.config.rollouts:
            raise ValueError('save training checkpoints at a completed rollout boundary')
        payload = dict(format='gru-a2c-v1', config=asdict(self.config), num_envs=self.num_envs,
                       contract=contract, policy=self.policy.state_dict(), value=self.value.state_dict(),
                       optimizer=self.optimizer.state_dict(), steps=self.steps, updates=self.updates,
                       torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                       python_rng=random.getstate(),
                       cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path, contract, training=True):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('format') != 'gru-a2c-v1' or payload['contract'] != contract:
            raise ValueError('checkpoint algorithm or original task contract does not match')
        if asdict(A2CConfig(**payload['config'])) != asdict(self.config):
            raise ValueError('checkpoint configuration does not match')
        if training and payload['num_envs'] != self.num_envs:
            raise ValueError('resume training with the saved number of environments')
        self.policy.load_state_dict(payload['policy'])
        self.value.load_state_dict(payload['value'])
        self.steps, self.updates = payload['steps'], payload['updates']
        if training:
            self.optimizer.load_state_dict(payload['optimizer'])
            torch.set_rng_state(payload['torch_rng'])
            np.random.set_state(payload['numpy_rng'])
            random.setstate(payload['python_rng'])
            if payload['cuda_rng'] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(payload['cuda_rng'])
        # Physical state restarts on resume, so rollout memory and GRU states start empty.
