"""Bounded observation history and observation-only curiosity for SAC."""
import torch
from torch import nn


class ObservationHistory:
    def __init__(self, observations, length=1, stride=1):
        if min(length, stride) < 1:
            raise ValueError('history length and stride must be positive')
        self.length, self.stride = length, stride
        self.frames = observations[:, None].repeat(1, 1 + (length - 1) * stride, 1)

    def current(self):
        return self.frames[:, ::self.stride].flatten(1).clone()

    def advance(self, successors, reset_observations, done):
        # Construct the terminal history before replacing ended episodes.
        self.frames = torch.cat((self.frames[:, 1:], successors[:, None]), dim=1)
        terminal = self.current()
        self.frames[done] = reset_observations[done, None]
        return terminal, self.current()


class HistoryEncoder(nn.Module):
    def __init__(self, observation_dim, length, hidden):
        super().__init__()
        self.observation_dim, self.length = observation_dim, length
        self.gru = nn.GRU(observation_dim, hidden, batch_first=True)

    def forward(self, observations):
        sequence = observations.reshape(-1, self.length, self.observation_dim)
        return self.gru(sequence)[1][-1]


class RandomNetworkDistillation(nn.Module):
    def __init__(self, observation_dim, hidden=64, learning_rate=3e-4, device='cpu'):
        super().__init__()

        def network():
            return nn.Sequential(nn.Linear(observation_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 32))

        self.target = network().requires_grad_(False)
        self.predictor = network()
        self.register_buffer('count', torch.tensor(1e-4, dtype=torch.float64))
        self.register_buffer('mean', torch.tensor(0., dtype=torch.float64))
        self.register_buffer('m2', torch.tensor(1e-4, dtype=torch.float64))
        self.register_buffer('transitions', torch.tensor(0, dtype=torch.long))
        self.to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

    def error(self, observations):
        with torch.no_grad():
            target = self.target(observations)
        return (self.predictor(observations) - target).square().mean(-1)

    @torch.no_grad()
    def bonus(self, observations):
        scale = (self.m2 / self.count).sqrt().clamp_min(1e-4)
        return (self.error(observations) / scale).clamp(0, 5).float()

    def observe(self, observations):
        error = self.error(observations.detach())
        with torch.no_grad():
            values = error.double()
            count = values.numel()
            delta = values.mean() - self.mean
            total = self.count + count
            self.m2.add_(((values - values.mean()).square().sum()
                          + delta.square() * self.count * count / total))
            self.mean.add_(delta * count / total)
            self.count.copy_(total)
            self.transitions.add_(count)
        loss = error.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.)
        self.optimizer.step()
        return loss.item()


class UpdateSchedule:
    """Keep fractional update credit when collecting several transitions at once."""
    def __init__(self, ratio, credit=0.):
        self.ratio, self.credit = ratio, credit

    def advance(self, transitions):
        self.credit += transitions * self.ratio
        updates = int(self.credit + 1e-9)
        self.credit -= updates
        return updates
