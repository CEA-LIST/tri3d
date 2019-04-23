import numpy as np
import pytest
from tri3d.geometry import AffineTransform, Rotation, Translation, RigidTransform
from scipy.spatial.transform import Rotation as SPRotation


rnd_seed = 0
transform_classes = (AffineTransform, Rotation, Translation, RigidTransform)


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
    elif issubclass(cls, Rotation):
        quat = np.random.randn(*shape, 4)
        quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
        return (quat,)
    elif issubclass(cls, Translation):
        vec = np.random.randn(*shape, 3)
        return (vec,)
    elif issubclass(cls, RigidTransform):
        quat, vec = np.split(np.random.randn(*shape, 7), [4], axis=-1)
        return quat, vec


def test_affine():
    (mat,) = gen_args(AffineTransform, [])
    transform = AffineTransform(mat)
    pts = np.random.randn(3)

    actual = transform.apply(pts)
    expected = mat[:3, :3] @ pts + mat[:3, 3]

    assert actual == pytest.approx(expected)


def test_rotation():
    (quat,) = gen_args(Rotation, [])
    transform = Rotation(quat)
    point = np.random.randn(3)

    actual = transform.apply(point)
    w, x, y, z = quat
    expected = SPRotation.from_quat([x, y, z, w]).apply(point)

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


@pytest.mark.parametrize("transform_cls", transform_classes)
@pytest.mark.parametrize("t_shape", [(), (10,)])
def test_inv(transform_cls, t_shape):
    point = np.random.randn(10, 3)
    args = gen_args(transform_cls, t_shape)
    transform = transform_cls(*args)
    transform_inv = transform.inv()

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
