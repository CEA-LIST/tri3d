import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SPRotation

from tri3d.geometry import (
    AffineTransform,
    CameraProjection,
    Pipeline,
    RigidTransform,
    Rotation,
    Translation,
)

rnd_seed = 0
transform_classes = (
    AffineTransform,
    CameraProjection,
    Pipeline,
    RigidTransform,
    Rotation,
    Translation,
)


def equal_quat(q1, q2):
    return np.allclose(q1 * np.sign(q1[..., 0, None]), q2 * np.sign(q2[..., 0, None]))


@pytest.fixture(autouse=True)
def reset_rnd_seed():
    np.random.seed(rnd_seed)


def gen_args(cls, shape):
    # WARNING: must return equivalent args when called sequentially from the
    # same random seed.
    if issubclass(cls, AffineTransform):
        mat = np.random.randn(*shape, 4, 4)
        mat[..., 3, :] = 0
        mat[..., 3, 3] = 1
        return (mat,)
    elif issubclass(cls, CameraProjection):
        model = ["pinhole", "kannala"][np.random.randint(0, 1)]
        if model == "pinhole":
            intrinsics = np.random.rand(*shape, 9)
        else:
            intrinsics = np.random.rand(*shape, 8)
        model = ["pinhole", "kannala"][np.random.randint(0, 1)]
        return model, intrinsics
    elif issubclass(cls, Pipeline):
        quat, vec = np.split(np.random.randn(*shape, 7), [4], axis=-1)
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        return Translation(vec), Rotation(quat)
    elif issubclass(cls, RigidTransform):
        quat, vec = np.split(np.random.randn(*shape, 7), [4], axis=-1)
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        return quat, vec
    elif issubclass(cls, Rotation):
        quat = np.random.randn(*shape, 4)
        quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
        return (quat,)
    elif issubclass(cls, Translation):
        vec = np.random.randn(*shape, 3)
        return (vec,)
    else:
        raise ValueError()


def test_affine():
    mat, = gen_args(AffineTransform, [])
    transform = AffineTransform(mat)
    pts = np.random.randn(3)

    actual = transform.apply(pts)
    expected = mat[:3, :3] @ pts + mat[:3, 3]

    assert actual == pytest.approx(expected)


def test_camera_projection():
    transform = CameraProjection("pinhole", ())


def test_rotation():
    (quat,) = gen_args(Rotation, [])
    transform = Rotation(quat)
    point = np.random.randn(3)

    # apply
    actual = transform.apply(point)
    w, x, y, z = quat
    expected = SPRotation([x, y, z, w]).apply(point)
    assert actual == pytest.approx(expected)


def test_rotation_conversion():
    (quat,) = gen_args(Rotation, [10])
    refrot = SPRotation.from_quat(quat[:, [1, 2, 3, 0]])

    # from_matrix
    actual = Rotation.from_matrix(refrot.as_matrix()).quat
    assert equal_quat(actual, quat)

    for seq in ["XYZ", "XYX", "xyz"]:
        actual = Rotation.from_euler(seq, refrot.as_euler(seq)).quat
        assert equal_quat(actual, quat)

        actual = Rotation(quat).as_euler(seq)
        expected = refrot.as_euler(seq)
        assert actual == pytest.approx(expected)


def test_translation():
    (vec,) = gen_args(Translation, [])
    transform = Translation(vec)
    point = np.random.randn(3)

    actual = transform.apply(point)
    expected = point + vec

    assert actual == pytest.approx(expected)


def test_rigid():
    quat, vec = gen_args(RigidTransform, [])
    transform = RigidTransform(quat, vec)
    point = np.random.randn(3)

    actual = transform.apply(point)
    expected = Rotation(quat).apply(point) + vec

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("t1_shape", [(), (10,)])
@pytest.mark.parametrize("t2_shape", [(), (10,)])
def test_pipeline(t1_shape, t2_shape):
    for T1, T2 in itertools.product(transform_classes, transform_classes):
        t1 = T1(*gen_args(T1, t1_shape))
        t2 = T2(*gen_args(T2, t2_shape))
        tp = Pipeline(t1, t2)
        point = np.random.randn(10, 3)
        predicted = tp.apply(point)
        expected = t2.apply(t1.apply(point))
        assert predicted == pytest.approx(expected)


@pytest.mark.parametrize("transform_cls", transform_classes)
@pytest.mark.parametrize("t_shape", [(), (10,)])
def test_inv(transform_cls, t_shape):
    point = np.random.randn(10, 3)
    args = gen_args(transform_cls, t_shape)
    transform = transform_cls(*args)
    try:
        transform_inv = transform.inv()
    except NotImplementedError:
        pytest.skip(f"{transform_cls} does not implement inv()")
        return

    actual = transform_inv.apply(transform.apply(point))
    assert actual == pytest.approx(point)


@pytest.mark.parametrize("transform_cls", transform_classes)
def test_indexing(transform_cls):
    args = gen_args(transform_cls, [10])
    batch_transform = transform_cls(*args)

    single_transforms = []
    np.random.seed(rnd_seed)
    for i in range(10):
        args = gen_args(transform_cls, [])
        single_transforms.append(transform_cls(*args))

    pts = np.random.randn(3)

    predicted = batch_transform.apply(pts)
    expected = np.stack([t.apply(pts) for t in single_transforms])

    assert predicted == pytest.approx(expected)

    with pytest.raises(TypeError):
        single_transforms[0][0]

    with pytest.raises(TypeError):
        len(single_transforms[0])

    assert len(batch_transform) == 10

    assert len(list(batch_transform)) == 10


@pytest.mark.parametrize("transform_cls", transform_classes)
def test_broadcast(transform_cls):
    # single transform, batch points
    args = gen_args(transform_cls, [])
    transform = transform_cls(*args)
    pts = np.random.randn(10, 3)

    predicted = transform.apply(pts)
    expected = np.stack([transform.apply(p) for p in pts])
    assert expected == pytest.approx(predicted)

    # batch transform, single points
    args = gen_args(transform_cls, [10])
    transform = transform_cls(*args)
    pts = np.random.randn(3)

    predicted = transform.apply(pts)
    expected = np.stack([t.apply(pts) for t in transform])
    assert expected == pytest.approx(predicted)

    # batch transform, batch points
    args = gen_args(transform_cls, [10])
    transform = transform_cls(*args)
    pts = np.random.randn(10, 3)

    predicted = transform.apply(pts)
    expected = np.stack([t.apply(p) for t, p in zip(transform, pts)])
    assert expected == pytest.approx(predicted)


@pytest.mark.parametrize("T1", transform_classes)
@pytest.mark.parametrize("T2", transform_classes)
def test_chain(T1, T2):
    t1 = T1(*gen_args(T1, []))
    t2 = T2(*gen_args(T2, []))
    pts = np.random.randn(3)

    predicted = (t2 @ t1).apply(pts)
    expected = t2.apply(t1.apply(pts))

    assert predicted == pytest.approx(expected)

    t1 = T1(*gen_args(T1, [10]))
    t2 = T2(*gen_args(T2, []))

    predicted = (t2 @ t1).apply(pts)
    expected = t2.apply(t1.apply(pts))

    assert predicted == pytest.approx(expected)

    t1 = T1(*gen_args(T1, []))
    t2 = T2(*gen_args(T2, [10]))

    predicted = (t2 @ t1).apply(pts)
    expected = t2.apply(t1.apply(pts))

    assert predicted == pytest.approx(expected)
