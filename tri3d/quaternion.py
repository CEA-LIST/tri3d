import numpy as np


def from_euler(seq, angles, degrees=False):
    angles = np.asarray(angles)
    
    if len(seq) == 1:
        angles = angles[..., None]

    if degrees:
        angles = angles / 180 * np.pi

    quat = np.zeros(angles.shape[:-1] + (4,))
    quat[..., 0] = 1

    for i in range(len(seq)):
        c = np.cos(0.5 * angles[..., i])
        s = np.sin(0.5 * angles[..., i])

        if seq.islower():
            x = seq[i] == "x"
            y = seq[i] == "y"
            z = seq[i] == "z"

        else:
            extrinsic_axes = quaternion_rotation_matrix(quat)
            i = 0 if seq[i] == "X" else 1 if seq[i] == "Y" else 2
            x = extrinsic_axes[..., 0, i]
            y = extrinsic_axes[..., 1, i]
            z = extrinsic_axes[..., 2, i]

        q = np.stack([c, x * s, y * s, z * s], axis=-1)
        quat = quaternion_multiply(q, quat)

    return quat


def as_euler(seq: str, quat: np.ndarray, degrees: bool = False):
    """Return the rotation encoded as euler angles.

    straight from https://pmc.ncbi.nlm.nih.gov/articles/PMC9648712/
    """
    quat = np.asarray(quat)

    if seq.isupper():
        return as_euler(seq.lower()[::-1], quat, degrees)[..., ::-1]

    if degrees:
        return as_euler(seq, quat) * (180 / np.pi)

    if not seq.islower():
        raise ValueError("Cannot mix intrinsic and extrinsic rotations")

    i, j, k = ["xyz".index(ax) + 1 for ax in seq]
    if i == j or j == k:
        raise ValueError("invalid rotation axes")

    if i == k:
        not_proper = False
        k = 6 - i - j
    else:
        not_proper = True

    eps = (i - j) * (j - k) * (k - i) // 2

    if not_proper:
        a = quat[..., 0] - quat[..., j]
        b = quat[..., i] + quat[..., k] * eps
        c = quat[..., j] + quat[..., 0]
        d = quat[..., k] * eps - quat[..., i]
    else:
        a = quat[..., 0]
        b = quat[..., i]
        c = quat[..., j]
        d = quat[..., k] * eps

    theta2 = np.arccos(2 * (a**2 + b**2) / (a**2 + b**2 + c**2 + d**2) - 1)
    thetap = np.arctan2(b, a)
    thetam = np.arctan2(d, c)

    case1 = theta2 < 1e-6
    case2 = np.abs(theta2 - np.pi / 2) < 1e-6
    out1 = 0
    out2 = 0
    out3 = thetap - thetam
    theta1 = np.where(case1, out1, np.where(case2, out2, out3))

    out1 = 2 * thetap - theta1
    out2 = 2 * thetam + theta1
    out3 = thetap + thetam
    theta3 = np.where(case1, out1, np.where(case2, out2, out3))

    if not_proper:
        theta3 = eps * theta3
        theta2 = theta2 - np.pi / 2

    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta3 = (theta3 + np.pi) % (2 * np.pi) - np.pi

    return np.stack([theta1, theta2, theta3], axis=-1)


def slerp(q0, q1, t):
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
    t = np.asarray(t)

    dot = (q0 * q1).sum(axis=-1).clip(-1, 1)

    # Simple linear approximation
    out_linear = q0 + np.expand_dims(t, -1) * (q1 - q0)
    out_linear /= np.linalg.norm(out_linear, axis=-1, keepdims=True)

    # Slerp
    theta_0 = np.arccos(dot)  # theta_0 = angle between input vectors
    theta = theta_0 * t  # theta = angle between v0 and result

    q2 = q1 - q0 * dot[..., None]
    q2 /= np.linalg.norm(q2, axis=-1, keepdims=True).clip(min=1e-6)

    out_slerp = q0 * np.cos(theta[..., None]) + q2 * np.sin(theta[..., None])

    return np.where(dot[..., None] > 0.9995, out_linear, out_slerp)


def quaternion_multiply(q0, q1):
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)

    if q0.ndim < q1.ndim:
        q0 = np.broadcast_to(q0, q1.shape)
    else:
        q1 = np.broadcast_to(q1, q0.shape)

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    out = np.empty_like(q0)
    out[..., 0] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    out[..., 1] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    out[..., 2] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    out[..., 3] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    out = out / np.linalg.norm(out, axis=-1, keepdims=True)

    return out


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    :param Q:
        (Nx)4 element array representing the quaternion (q0,q1,q1,q3)
    :return:
        A (Nx)3x3 element matrix representing the 3D rotation.
    """
    Q = np.asarray(Q)

    w = Q[..., 0]
    x = Q[..., 1]
    y = Q[..., 2]
    z = Q[..., 3]

    mat = np.zeros(Q.shape[:-1] + (3, 3))

    mat[..., 0, 0] = 1.0 - 2.0 * y * y - 2.0 * z * z
    mat[..., 0, 1] = 2.0 * x * y - 2.0 * z * w
    mat[..., 0, 2] = 2.0 * x * z + 2.0 * y * w

    mat[..., 1, 0] = 2.0 * x * y + 2.0 * z * w
    mat[..., 1, 1] = 1.0 - 2.0 * x * x - 2.0 * z * z
    mat[..., 1, 2] = 2.0 * y * z - 2.0 * x * w

    mat[..., 2, 0] = 2.0 * x * z - 2.0 * y * w
    mat[..., 2, 1] = 2.0 * y * z + 2.0 * x * w
    mat[..., 2, 2] = 1.0 - 2.0 * x * x - 2.0 * y * y

    # mat[..., 3, 3] = 1

    return mat
