import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign().mul(noise.abs().sqrt())

    def reset_noise(self) -> None:
        device = self.weight_mu.device
        eps_in = self._scale_noise(self.in_features, device)
        eps_out = self._scale_noise(self.out_features, device)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions=3, scalar_dim=3):
        super().__init__()
        channels, height, width = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            flat_size = self.features(dummy).view(1, -1).size(1)

        self.fc_input = flat_size + scalar_dim

        self.val_fc1 = NoisyLinear(self.fc_input, 256)
        self.val_fc2 = NoisyLinear(256, 1)
        self.adv_fc1 = NoisyLinear(self.fc_input, 256)
        self.adv_fc2 = NoisyLinear(256, num_actions)
        self.noisy_layers = nn.ModuleList(
            [self.val_fc1, self.val_fc2, self.adv_fc1, self.adv_fc2]
        )

    def forward(self, x, scalars):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if scalars is not None:
            x = torch.cat((x, scalars), dim=1)
        val = F.relu(self.val_fc1(x))
        val = self.val_fc2(val)
        adv = F.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)
        return val + adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        for layer in self.noisy_layers:
            layer.reset_noise()
