import itertools
from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from .camera import project_kannala, project_pinhole
from .quaternion import (
    quaternion_multiply,
    quaternion_rotation_matrix,
    slerp,
    as_euler,
    from_euler,
)


__all__ = [
    "Transformation",
    "Pipeline",
    "AffineTransform",
    "Rotation",
    "Translation",
    "RigidTransform",
    "CameraProjection",
    "where_in_box",
]


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
        if self.single:
            raise TypeError("cannot index single transform")

        return Pipeline(
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
        self.mat = np.tile(np.eye(4), mat.shape[:-2] + (1, 1))
        self.mat[..., :mat.shape[-2], :mat.shape[-1]] = mat

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

        return AffineTransform(self.mat[item])

    def __matmul__(self, other: Transformation) -> Transformation:
        if isinstance(other, AffineTransform):
            mat = self.mat
            other_mat = other.mat

            if mat.ndim < other_mat.ndim:
                mat = np.broadcast_to(mat, other_mat.shape)
            else:
                other_mat = np.broadcast_to(other_mat, mat.shape)

            return AffineTransform(mat @ other_mat)

        else:
            return super().__matmul__(other)

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x)

        return (
            np.linalg.vecdot(self.mat[..., :3, :3], np.expand_dims(x, -2))
            + self.mat[..., :3, 3]
        )

    def inv(self) -> Self:
        return AffineTransform(np.linalg.inv(self.mat))


class Rotation(Transformation):
    """A Rotation defined by a quaternion (w, x, y, z)."""

    def __init__(self, quat):
        self.quat = np.asarray(quat)
        self.mat = quaternion_rotation_matrix(self.quat)

    def __repr__(self):
        if self.single:
            yaw, pitch, roll = self.as_euler("ZYX", degrees=True)
            return f"Rotation([{yaw:2.0f}°, {pitch:2.0f}°, {roll:2.0f}°])"
        else:
            yaw, pitch, roll = self.as_euler("ZYX", degrees=True)[0]
            return f"Rotation([[{yaw:2.0f}°, {pitch:2.0f}°, {roll:2.0f}°], ...])"

    def __matmul__(self, other: Transformation) -> Transformation:
        if isinstance(other, Rotation):
            return Rotation(quaternion_multiply(self.quat, other.quat))

        elif isinstance(other, Translation):
            return RigidTransform(self, self.apply(other.vec))

        elif isinstance(other, RigidTransform):
            return RigidTransform(
                self @ other.rotation, self.apply(other.translation.apply(np.zeros(3)))
            )

        else:
            return super().__matmul__(other)

    def __len__(self):
        if self.single:
            raise TypeError("single transform has no len")

        return len(self.quat)

    def __getitem__(self, item) -> Self:
        if self.single:
            raise TypeError("cannot index single transform")

        obj = Rotation.__new__(Rotation)
        obj.quat = self.quat[item]
        obj.mat = self.mat[item]
        return obj

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x)
        return np.linalg.vecdot(np.expand_dims(x, -2), self.mat)

    def inv(self) -> Self:
        return Rotation(self.quat * [1, -1, -1, -1])

    @property
    def single(self) -> bool:
        return self.quat.ndim == 1

    def as_quat(self):
        """Return the quaternion representation."""
        return np.copy(self.quat)

    @classmethod
    def from_matrix(cls, mat):
        """Create a rotation from 3x3 or 4x4 rotation matrices."""
        mat = np.asarray(mat)

        w = 0.5 * np.sqrt(1 + mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2])
        x = (mat[..., 2, 1] - mat[..., 1, 2]) * 0.25 / w
        y = (mat[..., 0, 2] - mat[..., 2, 0]) * 0.25 / w
        z = (mat[..., 1, 0] - mat[..., 0, 1]) * 0.25 / w
        quat = np.stack([w, x, y, z], axis=-1)
        return cls(quat)

    def as_matrix(self) -> np.ndarray:
        """Return the rotation as a 4x4 rotation matrix."""
        out = np.zeros(self.mat.shape[:-2] + (4, 4), dtype=self.mat.dtype)
        out[:] = np.eye(4)
        out[..., :3, :3] = self.mat

        return out

    def as_euler(self, seq: str, degrees: bool = False):
        return as_euler(seq, self.quat, degrees)

    @classmethod
    def from_euler(cls, seq: str, angles, degrees: bool = False):
        return cls(from_euler(seq, angles, degrees))


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

        return Translation(self.vec[item])

    def __matmul__(self, other: Transformation) -> Transformation:
        if isinstance(other, Translation):
            return Translation(self.vec + other.vec)
        elif isinstance(other, RigidTransform):
            return RigidTransform(other.rotation, other.translation.vec + self.vec)
        elif isinstance(other, Rotation):
            return RigidTransform(other, self)
        else:
            return super().__matmul__(other)

    def __add__(self, other):
        return Translation(self.vec + other.vec)

    def __neg__(self):
        return Translation(-self.vec)

    def __sub__(self, other):
        return Translation(self.vec - other.vec)

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x)
        return np.add(x, self.vec, dtype=x.dtype)

    def inv(self) -> Self:
        return Translation(-self.vec)


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
        """Linearly interpolate between two transformations.

        :param p1:
            Left transformation
        :param p2:
            Right transformations
        :param w:
            Interpolation ratio such that 0. -> p1, 1. -> p2.
        """
        w = np.asarray(w)
        q = slerp(p1.rotation.quat, p2.rotation.quat, w)
        t = (1 - w[..., None]) * p1.translation.vec + w[..., None] * p2.translation.vec
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

        return RigidTransform(
            self.rotation if self.rotation.single else self.rotation[item],
            self.translation if self.translation.single else self.translation[item],
        )

    def __iter__(self):
        return itertools.starmap(RigidTransform, zip(self.rotation, self.translation))

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


def where_in_box(pts, size, box2sensor: Transformation) -> np.ndarray:
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


def test_box_in_frame(obj2img: Transformation, obj_size, img_size) -> bool:
    """Return whether any of a box corner points lands within an image."""
    pts_2d = obj2img.apply(cube_edges * obj_size)
    in_frame = (
        (pts_2d[:, 2] > 0)
        & np.all(pts_2d > 0, axis=1)
        & np.all(pts_2d[:, :2] < [img_size], axis=1)
    )
    return any(in_frame)


# def bbox_2d(obj2img, size, imsize):
#     """Return the 2D bounding box for a 3D box projected onto an image."""
#     size = np.asarray(size)
#     imsize = tuple(imsize)

#     edges_25d = obj2img.apply(cube_edges * size[None, :])
#     zmax = max(edges_25d[:, 2])

#     # filter in frame
#     box_25d = shapely.geometry.MultiPoint(edges_25d).convex_hull
#     visible_space = shapely.geometry.MultiPoint(
#         [(0, 0, 0), imsize + (0,), (0, 0, zmax)]
#     ).envelope
#     box_visible = box_25d.intersection(visible_space).envelope
#     edges_2d = np.array(box_visible.exterior.coords)

#     return tuple(edges_2d.min(0)) + tuple(edges_2d.max(0))
