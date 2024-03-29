from abc import abstractmethod
from functools import partial

import numpy as np
import torch

from ...modules.diffusionmodules.util import make_beta_schedule
from ...util import append_zero


def generate_roughly_equally_spaced_steps(
    num_substeps: int, max_step: int
) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


class Discretization:
    def __call__(self, n, do_append_zero=True, device="cpu", flip=False):
        sigmas = self.get_sigmas(n, device=device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))

    @abstractmethod
    def get_sigmas(self, n, device):
        pass


class EDMDiscretization(Discretization):
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, n, device="cpu"):
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas


class LegacyDDPMDiscretization(Discretization):
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(
            "linear", num_timesteps, linear_start=linear_start, linear_end=linear_end
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

    def get_sigmas(self, n, device="cpu"):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        sigmas = to_torch((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return torch.flip(sigmas, (0,))

class Img2ImgDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 1.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization: Discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        print("prune index:", max(int(self.strength * len(sigmas)), 1))
        sigmas = torch.flip(sigmas, (0,))
        print(f"sigmas after pruning: ", sigmas)
        return sigmas


class Txt2NoisyDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 0.0 means full sampling (all sigmas are returned)
    """

    def __init__(
        self, discretization: Discretization, strength: float = 0.0, original_steps=None
    ):
        self.discretization = discretization
        self.strength = strength
        self.original_steps = original_steps
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        if self.original_steps is None:
            steps = len(sigmas)
        else:
            steps = self.original_steps + 1
        prune_index = max(min(int(self.strength * steps) - 1, steps - 1), 0)
        sigmas = sigmas[prune_index:]
        print("prune index:", prune_index)
        sigmas = torch.flip(sigmas, (0,))
        print(f"sigmas after pruning: ", sigmas)
        return sigmas