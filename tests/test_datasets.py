import pathlib

import numpy as np
import pytest

from tri3d.datasets import (
    AbstractDataset,
    Argoverse2,
    NuScenes,
    Once,
    SemanticKITTI,
    Waymo,
    ZODFrames,
)


@pytest.fixture(
    scope="module", params=[Argoverse2, NuScenes, Once, SemanticKITTI, Waymo, ZODFrames]
)
def dataset(request) -> AbstractDataset:
    cls = request.param
    return cls(pathlib.Path(__file__).parent.parent / "datasets" / cls.__name__.lower())


def test_frames(dataset):
    # frames
    sequences = dataset.sequences()
    assert len(sequences) > 0


def test_data(dataset):
    sequences = dataset.sequences()
    seq = sequences[-1]

    # Sensors have frames and frames have data
    for s in dataset.pcl_sensors:
        frames = dataset.frames(seq, sensor=s)
        if len(frames) > 0:
            pcl = dataset.points(seq, frames[0], sensor=s)
            assert pcl.shape[0] > 0 and pcl.shape[1] >= 3

    for s in dataset.img_sensors:
        frames = dataset.frames(seq, sensor=s)
        if len(frames) > 0:
            img = dataset.image(seq, frames[-1], sensor=s)
            assert img.size > (0, 0)


def test_timestamps(dataset):
    sequences = dataset.sequences()

    # All timelines are available and sorted
    for s in dataset.pcl_sensors + dataset.cam_sensors + dataset.img_sensors:
        frames = dataset.frames(sequences[0], sensor=s)
        timestamps = dataset.timestamps(sequences[0], sensor=s)
        assert isinstance(timestamps, np.ndarray)
        assert len(frames) == len(timestamps)
        assert np.all(timestamps == np.sort(timestamps))


def test_poses(dataset):
    sequences = dataset.sequences()

    for s in dataset.pcl_sensors + dataset.cam_sensors + dataset.img_sensors:
        frames = dataset.frames(sequences[0], sensor=s)
        if len(frames) > 0:  # skip missing sensor
            poses = dataset.poses(sequences[0], sensor=s)
            assert len(frames) == len(poses)

    if len(dataset.cam_sensors) == 0:
        return

    frames = dataset.frames(sequences[0], sensor=dataset.pcl_sensors[0])
    poses = dataset.poses(
        sequences[0], sensor=dataset.cam_sensors[0], timeline=dataset.pcl_sensors[0]
    )

    assert len(poses) == len(frames)


def test_alignment(dataset):
    if len(dataset.cam_sensors) == 0:
        pytest.skip("no camera sensors")
        return

    sequences = dataset.sequences()
    seq = sequences[0]
    s1 = dataset.pcl_sensors[0]
    s2 = dataset.cam_sensors[0]
    f1 = 0
    f2 = len(dataset.frames(seq, sensor=s2)) - 1

    p1 = dataset.poses(seq, sensor=s1)[f1]
    p2 = dataset.poses(seq, sensor=s2)[f2]

    t = dataset.alignment(seq, (f1, f2), (s1, s2))

    x = np.random.randn(3)

    assert t.apply(x) == pytest.approx((p2.inv() @ p1).apply(x))
