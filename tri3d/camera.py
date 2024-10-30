import numpy as np


def project_pinhole(
    xyz: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    k3: float = 0.0,
):
    z = xyz[..., 2]
    x = np.sign(z) * xyz[..., 0] / z
    y = np.sign(z) * xyz[..., 1] / z

    r2 = np.square(x) + np.square(y)
    r4 = np.square(r2)
    r6 = r2 * r4

    radial_factor = 1 + k1 * r2 + k2 * r4 + k3 * r6
    x_ = x * radial_factor + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_ = y * radial_factor + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    u = fx * x_ + cx
    v = fy * y_ + cy

    return np.stack([u, v, z], axis=-1)


def project_kannala(
    xyz: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
) -> np.ndarray:
    """original code from https://github.com/zenseact/zod/blob/main/zod/utils/geometry.py"""
    norm_data = np.hypot(xyz[..., 0], xyz[..., 1])
    radial = np.arctan2(norm_data, xyz[..., 2])
    radial2 = radial**2
    radial4 = radial2**2
    radial6 = radial4 * radial2
    radial8 = radial4**2
    distortion_angle = radial * (
        1 + k1 * radial2 + k2 * radial4 + k3 * radial6 + k4 * radial8
    )
    u_dist = distortion_angle * xyz[..., 0] / norm_data
    v_dist = distortion_angle * xyz[..., 1] / norm_data
    pos_u = fx * u_dist + cx
    pos_v = fy * v_dist + cy
    return np.stack((pos_u, pos_v, xyz[..., 2]), axis=-1)
