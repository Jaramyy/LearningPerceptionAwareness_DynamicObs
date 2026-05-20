import math

import torch
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz


def sampleUniformQuatwithTilt(tile: torch.Tensor, size: int) -> torch.Tensor:
    """Sample uniform quaternions with tilt angle limit.

    Args:
        tile (torch.Tensor): Tilt angle limit in radians.
        size (int): Number of quaternions to sample.

    Returns:
        torch.Tensor: Sampled quaternions of shape (size, 4).
    """
    phi = torch.rand((size), device=tile.device) * 2.0 * math.pi
    theta = torch.rand((size), device=tile.device) * tile
    axisZ = torch.zeros((size, 3), device=tile.device)
    xyL = torch.sin(theta)
    # import ipdb; ipdb.set_trace()
    axisZ[:, 2] = torch.cos(theta)
    axisZ[:, 0] = xyL * torch.cos(phi)
    axisZ[:, 1] = xyL * torch.sin(phi)

    # print("axisZ:\n", axisZ)
    # print("norm axisZ:\n", torch.linalg.norm(axisZ, dim=-1))

    X_euler = torch.asin(-axisZ[:, 0])
    Y_euler = torch.atan2(axisZ[:, 1], axisZ[:, 2])
    # print("X_euler:\n", X_euler)
    # print("Y_euler:\n", Y_euler)
    Z_euler = torch.rand((size), device=tile.device) * 2.0 * math.pi
    # print("Z_euler:\n", Z_euler)
    quats = quat_from_euler_xyz(roll=X_euler, pitch=Y_euler, yaw=Z_euler)
    # print("quats:\n", quats)
    return quats


def sampleCenterQuatwithTilt(tile: torch.Tensor, size: int) -> torch.Tensor:
    """Sample centered quaternions with tilt angle limit, more concentrated near horizontal z axis.

    Args:
        tile (torch.Tensor): Tilt angle limit in radians.
        size (int): Number of quaternions to sample.

    Returns:
        torch.Tensor: Sampled quaternions of shape (size, 4).
    """
    phi = torch.rand((size), device=tile.device) * 2.0 * math.pi
    lowCostTheta = torch.cos(tile)
    cosTheta = torch.empty((size), device=tile.device).uniform_(float(lowCostTheta), 1.0)
    axisZ = torch.zeros((size, 3), device=tile.device)
    xyL = torch.sin(torch.acos(cosTheta))
    # import ipdb; ipdb.set_trace()
    axisZ[:, 2] = cosTheta
    axisZ[:, 0] = xyL * torch.cos(phi)
    axisZ[:, 1] = xyL * torch.sin(phi)

    # print("axisZ:\n", axisZ)
    # print("norm axisZ:\n", torch.linalg.norm(axisZ, dim=-1))

    X_euler = torch.asin(-axisZ[:, 0])
    Y_euler = torch.atan2(axisZ[:, 1], axisZ[:, 2])
    # print("X_euler:\n", X_euler)
    # print("Y_euler:\n", Y_euler)
    Z_euler = torch.rand((size), device=tile.device) * 2.0 * math.pi
    # print("Z_euler:\n", Z_euler)
    quats = quat_from_euler_xyz(roll=X_euler, pitch=Y_euler, yaw=Z_euler)
    # print("quats:\n", quats)
    return quats


def sampleSymmetryQuatwithTilt(tile: torch.Tensor, size: int) -> torch.Tensor:
    """Sample centered quaternions with tilt angle limit, more concentrated near horizontal z axis.

    Args:
        tile (torch.Tensor): Tilt angle limit in radians.
        size (int): Number of quaternions to sample.

    Returns:
        torch.Tensor: Sampled quaternions of shape (size, 4).
    """
    phi = torch.rand((size), device=tile.device) * 2.0 * math.pi
    lowCostTheta = torch.cos(tile)
    cosTheta = torch.empty((size), device=tile.device).uniform_(float(lowCostTheta), 1.0)
    cosTheta[::2] *= -1.0  # every other one flip
    axisZ = torch.zeros((size, 3), device=tile.device)
    xyL = torch.sin(torch.acos(cosTheta))
    # import ipdb; ipdb.set_trace()
    axisZ[:, 2] = cosTheta
    axisZ[:, 0] = xyL * torch.cos(phi)
    axisZ[:, 1] = xyL * torch.sin(phi)

    # print("axisZ:\n", axisZ)
    # print("norm axisZ:\n", torch.linalg.norm(axisZ, dim=-1))

    X_euler = torch.asin(-axisZ[:, 0])
    Y_euler = torch.atan2(axisZ[:, 1], axisZ[:, 2])
    # print("X_euler:\n", X_euler)
    # print("Y_euler:\n", Y_euler)
    Z_euler = torch.rand((size), device=tile.device) * 2.0 * math.pi
    # print("Z_euler:\n", Z_euler)
    quats = quat_from_euler_xyz(roll=X_euler, pitch=Y_euler, yaw=Z_euler)
    # print("quats:\n", quats)
    return quats


def sampleSymmetryQuatwithTiltforEnv(tile: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample uniform quaternions with tilt angle limit for each environment.

    Args:
        tile (torch.Tensor): Tilt angle limit in radians for each environment.
        size (int): Number of quaternions to sample.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Sampled quaternions of shape (size, 4).
            - Cosine of the tilt angle of shape (size,).
    """
    phi = torch.rand((size), device=tile.device) * 2.0 * math.pi

    # sample theta in [0, tile] => cos(theta) in [cos(tile), 1.0]
    min_cos = torch.cos(tile)
    r = torch.rand((size), device=tile.device)
    cosTheta = min_cos + r * (1.0 - min_cos)

    # symmetry: inverse the range of 50% [-1.0, -cos(tile)]
    # get inverse z axis
    flip_mask = torch.rand((size), device=tile.device) < 0.5
    cosTheta[flip_mask] *= -1.0

    axisZ = torch.zeros((size, 3), device=tile.device)
    # sqrt(1-cos^2) => sin, sin(acos()) more stable
    xyL = torch.sqrt((1.0 - cosTheta.pow(2)).clamp(min=0.0))

    axisZ[:, 2] = cosTheta
    axisZ[:, 0] = xyL * torch.cos(phi)
    axisZ[:, 1] = xyL * torch.sin(phi)

    X_euler = torch.asin(-axisZ[:, 0])
    Y_euler = torch.atan2(axisZ[:, 1], axisZ[:, 2])

    Z_euler = torch.rand((size), device=tile.device) * 2.0 * math.pi

    quats = quat_from_euler_xyz(roll=X_euler, pitch=Y_euler, yaw=Z_euler)
    return quats, cosTheta


if __name__ == "__main__":
    tile = torch.tensor(math.pi * 0.1)
    size = 10
    # quats = sampleUniformQuatwithTilt(tile, size)
    # print("sampleUniformQuatwithTilt:\n", quats)
    # quats = sampleCenterQuatwithTilt(tile, size)
    # print("sampleCenterQuatwithTilt:\n", quats)
    quats, cosTheta = sampleSymmetryQuatwithTiltforEnv(tile, size)
    print("sampleSymmetryQuatwithTilt:\n", quats)
    print("cosTheta:\n", cosTheta)