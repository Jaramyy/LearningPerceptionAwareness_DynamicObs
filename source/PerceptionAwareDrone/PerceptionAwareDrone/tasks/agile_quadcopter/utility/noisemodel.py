from collections.abc import Sequence
from functools import partial

import torch


class NoiseModel:
    def __init__(self, noise_cfg: dict, device: torch.device | str, num_envs: int):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.params = {}
        self.cfg = noise_cfg
        for name, cfg in noise_cfg.items():
            dim = cfg["dim"]
            mean = torch.empty(num_envs, dim, device=self.device, dtype=torch.float32).uniform_(
                -cfg.get("mean", 0.0), cfg.get("mean", 0.0)
            )
            std = torch.empty(num_envs, dim, device=self.device, dtype=torch.float32).uniform_(0.0, cfg.get("std", 0.0))
            clip = torch.empty(num_envs, dim, device=self.device, dtype=torch.float32).fill_(
                cfg.get("clip", float("inf"))
            )
            self.params[name] = {
                "dim": dim,
                "type": cfg["type"],
                "mean": mean,
                "std": std,
                "clip": clip,
            }
            setattr(self, f"{name}_apply", partial(self._add_noise, name=name))

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for name, cfg in self.params.items():
            dim = cfg["dim"]
            mean = torch.empty((len(env_ids), dim), device=self.device, dtype=torch.float32).uniform_(
                -self.cfg[name].get("mean", 0.0), self.cfg[name].get("mean", 0.0)
            )
            std = torch.empty((len(env_ids), dim), device=self.device, dtype=torch.float32).uniform_(
                0.0, self.cfg[name].get("std", 0.0)
            )
            clip = torch.empty((len(env_ids), dim), device=self.device, dtype=torch.float32).fill_(
                self.cfg[name].get("clip", float("inf"))
            )
            cfg["mean"][env_ids] = mean
            cfg["std"][env_ids] = std
            cfg["clip"][env_ids] = clip

    def apply(self, values: torch.Tensor, name: str) -> torch.Tensor:
        return self._add_noise(values, name)

    def _add_noise(self, values: torch.Tensor, name: str) -> torch.Tensor:
        cfg = self.params[name]
        shape = (self.num_envs, cfg["mean"].size(-1))
        mean = torch.ones(shape, device=self.device) * cfg["mean"]
        std = torch.ones(shape, device=self.device) * cfg["std"]
        if cfg["type"] == "uniform":
            noise = mean + std * (torch.rand(shape, device=self.device) * 2.0 - 1.0)
        elif cfg["type"] == "gaussian":
            noise = torch.normal(mean=mean, std=std)
        else:
            raise ValueError(f"Unsupported noise type {cfg['type']}")

        if cfg["clip"] is not None:
            noise = torch.clamp(noise, min=-cfg["clip"], max=cfg["clip"])

        return values + noise


# noisemodel.py
if __name__ == "__main__":
    """Main function."""
    # run the main function
    noise_cfg = {
        "pos": {"type": "gaussian", "dim": 3, "mean": 0.1, "std": 0.1, "clip": 0.2},
        "vel": {"type": "uniform", "dim": 3, "mean": 0.1, "std": 0.05, "clip": 0.1},
    }
    num_envs = 11
    device = "cpu"
    noise_model = NoiseModel(noise_cfg, device, num_envs)

    pos = torch.zeros((num_envs, 3), device=device)
    vel = torch.zeros((num_envs, 3), device=device)

    noisy_pos = noise_model.apply(pos, "pos")
    noisy_vel = noise_model.apply(vel, "vel")
    print("Original pos:", pos)
    print("Noisy pos:", noisy_pos)
    print("Original vel:", vel)
    print("Noisy vel:", noisy_vel)
    print("Noise.mean", noise_model.params["pos"]["mean"])

    noise_model.reset()
    print("After reset noise parameters.")
    print("Noise.mean", noise_model.params["pos"]["mean"])
    noisy_pos = noise_model.apply(pos, "pos")
    noisy_vel = noise_model.apply(vel, "vel")
    print("After reset - Noisy pos:", noisy_pos)
    print("After reset - Noisy vel:", noisy_vel)