import itertools
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from scipy.spatial.transform import Rotation as SPRotation
import shapely.geometry


__all__ = [
    "Transformation",
    "Pipeline",
    "AffineTransform",
    "Rotation",
    "Translation",
    "RigidTransform",
    "CameraProjection",
    "where_in_box",
    "test_box_in_frame",
    "bbox_2d",
    "approx_kitti_bbox2d",
]


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

    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2, axis=-1, keepdims=True).clip(min=1e-6)

    out_slerp = q0 * np.cos(theta) + q2 * np.sin(theta)

    return np.where(dot > 0.9995, out_linear, out_slerp)


def quaternion_multiply(q0, q1):
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    out = np.empty_like(q0)
    out[..., 0] = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    out[..., 1] = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
    out[..., 2] = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
    out[..., 3] = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0

    return out


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q1,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    Q = np.asarray(Q)

    w = Q[..., 0]
    x = Q[..., 1]
    y = Q[..., 2]
    z = Q[..., 3]

    mat = np.zeros(Q.shape[:-1] + (4, 4))

    mat[..., 0, 0] = 1.0 - 2.0 * y * y - 2.0 * z * z
    mat[..., 0, 1] = 2.0 * x * y - 2.0 * z * w
    mat[..., 0, 2] = 2.0 * x * z + 2.0 * y * w

    mat[..., 1, 0] = 2.0 * x * y + 2.0 * z * w
    mat[..., 1, 1] = 1.0 - 2.0 * x * x - 2.0 * z * z
    mat[..., 1, 2] = 2.0 * y * z - 2.0 * x * w

    mat[..., 2, 0] = 2.0 * x * z - 2.0 * y * w
    mat[..., 2, 1] = 2.0 * y * z + 2.0 * x * w
    mat[..., 2, 2] = 1.0 - 2.0 * x * x - 2.0 * y * y

    mat[..., 3, 3] = 1

    return mat


def project_pinhole(
    xyz,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
):
    z = xyz[..., 2]
    x = np.sign(z) * xyz[..., 0] / z
    y = np.sign(z) * xyz[..., 1] / z

    r2 = np.square(x) + np.square(y)

    radial_factor = 1 + k1 * r2
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


cube_edges = np.array(
    [
        (-0.5, -0.5, -0.5),  #    5------6
        (+0.5, -0.5, -0.5),  #   /|     /|
        (+0.5, +0.5, -0.5),  #  / |    / |
        (-0.5, +0.5, -0.5),  # 4------7  |
        (-0.5, -0.5, +0.5),  # |  1---|--2
        (+0.5, -0.5, +0.5),  # | /    | /
        (+0.5, +0.5, +0.5),  # |/     |/
        (-0.5, +0.5, +0.5),  # 0------3
        (+0.5, +0.0, -0.5),
    ]
)


class Transformation(ABC):
    """Base class for geometric transformations.

    A transformation can can host a single or a batch of transformations as
    indicated by the the :attr:`single` attribute. Batched transformation
    support len and indexing.

    Transformations can be chained together with the matrix multiplication (`@`)
    operator.
    """

    @property
    @abstractmethod
    def single(self) -> bool:
        """Whether this is a single transformation or a batch."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self):
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __matmul__(self, other: Self) -> Self:
        """Compose transformations together.

        The rightmost operation is applied first.
        """
        if isinstance(other, Pipeline):
            return Pipeline(*other.operations, self)
        else:
            return Pipeline(other, self)

    @abstractmethod
    def apply(self, x) -> np.ndarray:
        """Apply transformation to given point or array of points.

        The broadcasting rules between transformations and their inputs are as follows:

        +-------------+-----------+------------+---------------------+
        | transform   | input     | output     | brodcast            |
        +-------------+-----------+------------+---------------------+
        | ``[]``      | ``[3]``   | ``[3]``    | single mapping      |
        +-------------+-----------+------------+---------------------+
        | ``[]``      | ``[n,3]`` | ``[n,3]``  | broadcast transform |
        +-------------+-----------+------------+---------------------+
        | ``[n]``     | ``[3]``   | ``[n,3]``  | broadcst input      |
        +-------------+-----------+------------+---------------------+
        | ``[n]``     | ``[n]``   | ``[n,3]``  | one-to-one mapping  |
        +-------------+-----------+------------+---------------------+
        """
        raise NotImplementedError

    @abstractmethod
    def inv(self) -> Self:
        """Return inverse transformation."""
        raise NotImplementedError


class Pipeline(Transformation):
    """A series of operations.

    Operations are applied in the provided order.
    """

    def __init__(self, *operations):
        self.operations = list(operations)

        max_len = max(1 if op.single else len(op) for op in operations)
        if not all(op.single or len(op) == max_len for op in operations):
            raise ValueError("Batched operations must have the same length")

    @property
    def single(self) -> bool:
        return all(p.single for p in self.operations)

    def __len__(self):
        if self.single:
            raise TypeError("Single pipeline has no len().")

        return max([len(op) for op in self.operations if not op.single])

    def __getitem__(self, item) -> Self:
        return self.__class__(
            *[op if op.single else op[item] for op in self.operations]
        )

    def __matmul__(self, other: Transformation) -> Transformation:
        if isinstance(other, Pipeline):
            return Pipeline(*other.operations, *self.operations)
        else:
            return Pipeline(other, *self.operations)

    def apply(self, x: np.ndarray):
        out = x
        for op in self.operations:
            out = op.apply(out)

        return out

    def inv(self) -> Self:
        return Pipeline(*[op.inv() for op in self.operations[::-1]])


class AffineTransform(Transformation):
    """Linear transformation defined by a matrix.

    :math:`x \\mapsto M \\begin{bmatrix}x \\\\ 1\\end{bmatrix}`
    """

    def __init__(self, mat: np.ndarray):
        mat = np.asarray(mat)
        mat = np.pad(mat, (4 - mat.shape[-2], 4 - mat.shape[-1]))
        mat[..., 3:, :] = np.eye(1, 4, k=3)

        self.mat = mat

    @property
    def single(self) -> bool:
        return self.mat.ndim == 2

    def __len__(self):
        if self.single:
            raise TypeError("Single AffineTransform has no length.")

        return self.mat.shape[0]

    def __getitem__(self, item) -> Self:
        if self.single:
            raise TypeError("cannot index single transform")

        return self.__class__(self.mat[item])

    def __matmul__(self, other: Transformation) -> Transformation:
        if isinstance(other, AffineTransform):
            mat = self.mat
            other_mat = other.mat

            if mat.ndim < other_mat.ndim:
                mat = np.broadcast_to(mat, other_mat.shape)
            else:
                other_mat = np.broadcast_to(other_mat, mat.shape)

            return self.__class__(mat @ other_mat)

        else:
            return super().__matmul__(other)

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x)

        return (
            np.linalg.vecdot(self.mat[..., :3, :3], np.expand_dims(x, -2))
            + self.mat[..., :3, 3]
        )

    def inv(self) -> Self:
        return self.__class__(np.linalg.inv(self.mat))


class Rotation(Transformation):
    """A Rotation defined by a quaternion (w, x, y, z)."""

    def __init__(self, quat):
        quat = np.asarray(quat)

        self.rot = SPRotation.from_quat(quat[..., [1, 2, 3, 0]])
        self.mat = self.rot.as_matrix()

    def __repr__(self):
        if self.single:
            yaw, pitch, roll = self.rot.as_euler("ZYX", degrees=True)
            return f"Rotation([{yaw:2.0f}°, {pitch:2.0f}°, {roll:2.0f}°])"
        else:
            yaw, pitch, roll = self.rot.as_euler("ZYX", degrees=True)[0]
            return f"Rotation([[{yaw:2.0f}°, {pitch:2.0f}°, {roll:2.0f}°], ...])"

    def __matmul__(self, other: Transformation) -> Transformation:
        if isinstance(other, Rotation):
            return self.__class__((self.rot * other.rot).as_quat()[..., [3, 0, 1, 2]])

        elif isinstance(other, Translation):
            return RigidTransform(self, self.apply(other.vec))

        elif isinstance(other, RigidTransform):
            return RigidTransform(
                self @ other.rotation, self.apply(other.translation.vec)
            )

        else:
            return super().__matmul__(other)

    def __len__(self):
        return len(self.rot)

    def __getitem__(self, item) -> Self:
        if self.single:
            raise TypeError("cannot index single transform")

        obj = Rotation([0, 0, 0, 1])
        obj.rot = self.rot[item]
        obj.mat = self.mat[item]
        return obj

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x)

        return np.matmul(
            np.expand_dims(x, -2), np.moveaxis(self.mat, -1, -2)[..., :3, :3]
        ).squeeze(-2)

    def inv(self) -> Self:
        return Rotation(self.rot.inv().as_quat()[..., [3, 0, 1, 2]])

    @property
    def single(self) -> bool:
        return self.rot.single

    @classmethod
    def from_attitude(cls, yaw, pitch, roll):
        """Create rotation using aeronautical interpretation of euler angles.

        The returned rotation sequentially applies yaw, pitch then roll,
        which is sometimes expressed as a rotation along :math:`zy'x"`.
        """
        scipy_rot = SPRotation.from_euler("ZYX", (yaw, pitch, roll))
        return Rotation(scipy_rot.as_quat()[..., [3, 0, 1, 2]])

    def as_quat(self):
        """Return the quaternion representation."""
        return self.rot.as_quat()[..., [3, 0, 1, 2]]

    @classmethod
    def from_matrix(cls, mat):
        """Create a rotation from 3x3 or 4x4 rotation matrices."""
        mat = np.asarray(mat)

        if mat.shape[1] == 4 and np.abs(mat[:, 3]).max() > 1e-6:
            raise ValueError()

        return Rotation(SPRotation.from_matrix(mat[:, :3]).as_quat()[..., [3, 0, 1, 2]])

    def as_matrix(self) -> np.ndarray:
        """Return the rotation as a 4x4 rotation matrix."""
        out = np.zeros(self.mat.shape[:-2] + (4, 4), dtype=self.mat.dtype)
        out[:] = np.eye(4)
        out[..., :3, :3] = self.mat

        return out

    def as_euler(self, seq: str, degrees: bool = False):
        """Return the rotation encoded as euler angles.

        This is an alias of :meth:`scipy.spatial.transform.Rotation.as_euler`
        """
        return self.rot.as_euler(seq, degrees)

    @classmethod
    def from_euler(cls, seq: str, degrees: bool = False):
        rot = SPRotation.from_euler(seq, degrees)
        return Rotation(rot.as_quat()[..., [3, 0, 1, 2]])


class Translation(Transformation):
    """Translatation"""

    def __init__(self, vec):
        self.vec = np.asarray(vec)

    @property
    def single(self) -> bool:
        return self.vec.ndim == 1

    def __repr__(self):
        if self.vec.ndim == 1:
            return "Translation([{:.2e}, {:.2e}, {:.2e}])".format(*self.vec)
        else:
            return "Translation([({:.2e}, {:.2e}, {:.2e}), ...])".format(*self.vec[0])

    def __len__(self):
        if self.single:
            raise TypeError("Single Translation has no len.")

        return len(self.vec)

    def __getitem__(self, item) -> Self:
        if self.single:
            raise TypeError("Single Translation has no len.")

        return self.__class__(self.vec.reshape((-1, 3))[item])

    def __matmul__(self, other: Transformation) -> Transformation:
        if isinstance(other, Translation):
            return self.__class__(self.vec + other.vec)
        elif isinstance(other, RigidTransform):
            return RigidTransform(other.rotation, other.translation.vec + self.vec)
        elif isinstance(other, Rotation):
            return RigidTransform(other, self)
        else:
            return super().__matmul__(other)

    def __add__(self, other):
        return self.__class__(self.vec + other.vec)

    def __neg__(self):
        return self.__class__(-self.vec)

    def __sub__(self, other):
        return self.__class__(self.vec - other.vec)

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x)
        return np.add(x, self.vec, dtype=x.dtype)

    def inv(self) -> Self:
        return self.__class__(-self.vec)


class RigidTransform(Transformation):
    """A rotation followed by a translation."""

    def __init__(self, rotation, translation):
        if not isinstance(rotation, Rotation):
            rotation = np.asarray(rotation)
            if rotation.shape[-2:] == (3, 3):
                # check orthonormality
                # inv = np.transpose(rotation, (0, 2, 1) if rotation.ndim == 3 else (1, 0))
                # if rotation.size > 0 and np.max(np.abs(rotation @ inv - np.eye(3))) > 1e-5:
                #     raise ValueError('matrix does not define a rotation')
                rotation = Rotation.from_matrix(rotation)
            elif 0 < rotation.ndim < 3 and rotation.shape[-1] == 4:
                rotation = Rotation(rotation)
            else:
                raise ValueError("invalid rotation value")

        if not isinstance(translation, Translation):
            translation = Translation(translation)

        self.rotation = rotation
        self.translation = translation

    @classmethod
    def interpolate(cls, p1: Self, p2: Self, w: float):
        q = slerp(p1.rotation.as_quat(), p2.rotation.as_quat(), w)
        t = w * p1.translation.vec + (1 - w) * p2.translation.vec
        return cls(q, t)

    @property
    def single(self) -> bool:
        return self.rotation.single and self.translation.single

    def __repr__(self):
        if self.translation.single:
            t = "[{:.1f}, {:.1f}, {:.1f}]".format(*self.translation.vec)
        else:
            t = "[({:.1f}, {:.1f}, {:.1f}), ...]".format(*self.translation.vec[0])
        if self.rotation.single:
            r = "[{:.0f}, {:.0f}, {:.0f}]".format(
                *self.rotation.as_euler("ZYX", degrees=True)
            )
        else:
            r = "[({:.0f}, {:.0f}, {:.0f}), ...]".format(
                *self.rotation[0].as_euler("ZYX", degrees=True)
            )

        return f"RigidTransform({r}, {t})"

    def __len__(self):
        if self.single:
            raise TypeError("Single RigidTranform has no len.")

        return max(
            1 if self.translation.single else len(self.translation),
            1 if self.rotation.single else len(self.rotation),
        )

    def __getitem__(self, item) -> Self:
        if self.single:
            raise TypeError("Single RigidTranform has no len.")

        return self.__class__(
            self.rotation if self.rotation.single else self.rotation[item],
            self.translation if self.translation.single else self.translation[item],
        )

    def __iter__(self):
        return itertools.starmap(self.__class__, zip(self.rotation, self.translation))

    def __matmul__(self, other: Transformation) -> Transformation:
        if isinstance(other, Translation):
            return RigidTransform(
                self.rotation, self.translation.vec + self.rotation.apply(other.vec)
            )
        elif isinstance(other, Rotation):
            return RigidTransform(self.rotation @ other, self.translation)
        elif isinstance(other, RigidTransform):
            return RigidTransform(
                self.rotation @ other.rotation,
                self.translation.vec + self.rotation.apply(other.translation.vec),
            )
        else:
            return super().__matmul__(other)

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x)
        return self.rotation.apply(x) + self.translation.vec

    def inv(self) -> Self:
        inv = self.rotation.inv()
        return RigidTransform(inv, -inv.apply(self.translation.vec))

    @staticmethod
    def from_matrix(mat):
        mat = np.asarray(mat)
        return RigidTransform(mat[..., :3, :3], mat[..., :3, 3])


class CameraProjection(Transformation):
    def __init__(self, model, intrinsics):
        self.model = model
        self.intrinsics = intrinsics

    @property
    def single(self) -> bool:
        return True

    def __len__(self) -> int:
        raise ValueError("single transform has no len")

    def __getitem__(self) -> int:
        raise ValueError("single transform has no items")

    def apply(self, xyz):
        if self.model == "pinhole":
            return project_pinhole(xyz, *self.intrinsics)
        elif self.model == "kannala":
            return project_kannala(xyz, *self.intrinsics)
        else:
            raise ValueError("")

    def as_matrix(self) -> np.ndarray:
        # TODO: implement at least for rectified camera
        raise NotImplementedError

    def inv(self) -> Self:
        # TODO: implement at least for rectified camera
        raise NotImplementedError


def where_in_box(pts, size, box2sensor: Transformation):
    """Return the points that lie inside a bounding box.

    :param pts: N by 3 array of point coordinates
    :param size: (l, w, h) triplet size of the box
    :param box2sensor: transformation from box local to sensor coordinates
    """
    # credit to https://math.stackexchange.com/a/1552579 and nuscene-devkit for
    # the method.
    pts = np.asarray(pts)
    size = np.asarray(size)

    # get box axes
    box_axes = box2sensor.apply([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    origin = box_axes[None, 0]
    box_axes = box_axes[1:] - origin

    # project points into box coordinates and mark points inside the box
    inside = np.abs((pts - origin) @ box_axes.T) < size[None, :] / 2

    return np.where(inside[:, 0] & inside[:, 1] & inside[:, 2])[0]


def test_box_in_frame(obj2img, obj_size, img_size):
    """Return whether any of a box corner points lands within an image."""
    pts_2d = obj2img.apply(cube_edges * obj_size)
    in_frame = (
        (pts_2d[:, 2] > 0)
        & np.all(pts_2d > 0, axis=1)
        & np.all(pts_2d[:, :2] < [img_size], axis=1)
    )
    return any(in_frame)


def bbox_2d(obj2img, size, imsize):
    """Return the 2D bounding box for a 3D box projected onto an image."""
    size = np.asarray(size)
    imsize = tuple(imsize)

    edges_25d = obj2img.apply(cube_edges * size[None, :])
    zmax = max(edges_25d[:, 2])

    # filter in frame
    box_25d = shapely.geometry.MultiPoint(edges_25d).convex_hull
    visible_space = shapely.geometry.MultiPoint(
        [(0, 0, 0), imsize + (0,), (0, 0, zmax)]
    ).envelope
    box_visible = box_25d.intersection(visible_space).envelope
    edges_2d = np.array(box_visible.exterior.coords)

    return tuple(edges_2d.min(0)) + tuple(edges_2d.max(0))


def approx_kitti_bbox2d(position, size, heading):
    """Approximate 2D object height as if kitti cameras were used.

    Returns fake 2D coordinates (xmin, ymin, xmax, ymax) such that
    ymax - ymin is the estimated 2D height in pixels.
    """
    l, w, h = size

    # find closest edge to camera
    obj2cam = Translation(position) @ Rotation.from_euler("Y", heading)
    edges = obj2cam.apply(
        0.5 * np.array([[-l, -w, 0], [-l, w, 0], [l, -w, 0], [l, w, 0]])
    )
    dist = np.linalg.norm(edges, axis=-1).min()

    # infer approximate pixel height
    height_2d = h * 740 / dist  # approximate vertical camera intrinsic

    return -1, 0, -1, height_2d
