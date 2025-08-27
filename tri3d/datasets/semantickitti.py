import pathlib

import numpy as np

from ..geometry import RigidTransform, Translation
from .dataset import Dataset

label_names = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}


train_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
val_sequences = ["08"]
test_sequences = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]


class SemanticKITTI(Dataset):
    """`SemanticKITTI <https://semantic-kitti.org>`_ dataset.
    
    .. run::

        import matplotlib.pyplot as plt
        import numpy as np
        from tri3d.datasets import SemanticKITTI

        plt.switch_backend("Agg")

        dataset = SemanticKITTI("datasets/semantickitti")
        name = "tri3d.datasets.SemanticKITTI"
        camera, imgcoords, lidar = None, None, "velodyne"
        seq, frame, cam_frame = 1, 0, None
    """
    cam_sensors = []
    img_sensors = []
    pcl_sensors = ["velodyne"]
    det_labels = []
    sem_labels = [
        "unlabeled",
        "outlier",
        "car",
        "bicycle",
        "bus",
        "motorcycle",
        "on-rails",
        "truck",
        "other-vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
        "road",
        "parking",
        "sidewalk",
        "other-ground",
        "building",
        "fence",
        "other-structure",
        "lane-marking",
        "vegetation",
        "trunk",
        "terrain",
        "pole",
        "traffic-sign",
        "other-object",
        "moving-car",
        "moving-bicyclist",
        "moving-person",
        "moving-motorcyclist",
        "moving-on-rails",
        "moving-bus",
        "moving-truck",
        "moving-other-vehicle",
    ]

    _default_pcl_sensor = "velodyne"

    def __init__(self, root, sequences: list[str] = train_sequences):
        self.root = pathlib.Path(root)
        self._sequences = sequences

        # Remap labels to 0, 1, ...
        self.label_lut = np.empty([max(label_names.keys()) + 1], dtype=np.int64)
        for idx, old_idx in enumerate(sorted(label_names.keys())):
            self.label_lut[old_idx] = idx

        # load timestamps
        self._timestamps = [
            np.loadtxt(self.root / seq / "times.txt") for seq in sequences
        ]

        # load poses
        self.__poses = []
        for seq in sequences:
            calibs = {}
            with open(self.root / seq / "calib.txt") as f:
                for line in f.readlines():
                    if len(line) == 0:
                        continue

                    what, *values = line.split()
                    calibs[what] = np.eye(4)
                    calibs[what][:3, :] = np.array([float(x) for x in values]).reshape(
                        3, 4
                    )

            path = self.root / seq / "poses.txt"
            poses_ = np.loadtxt(path).reshape(-1, 3, 4)
            poses = np.tile(np.eye(4), [poses_.shape[0], 1, 1])
            poses[:, :3, :] = poses_
            poses = np.linalg.inv(calibs["Tr:"]) @ poses @ calibs["Tr:"]
            poses = RigidTransform.from_matrix(poses)

            self.__poses.append(poses)

    def _calibration(self, seq, src_sensor, dst_sensor):
        if src_sensor == dst_sensor:
            return Translation([0.0, 0.0, 0.0])

        else:
            raise ValueError()

    def _poses(self, seq, sensor):
        if sensor != self.pcl_sensors[0]:
            raise ValueError()

        return self.__poses[seq]

    def _points(self, seq, frame, sensor):
        path = self.root / self._sequences[seq] / "velodyne" / f"{frame:06d}.bin"
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    def _boxes(self, seq):
        raise NotImplementedError

    def sequences(self):
        return list(range(len(self._sequences)))

    def timestamps(self, seq, sensor="velodyne"):
        return self._timestamps[seq]

    def _labels(self, seq, frame):
        path = self.root / self._sequences[seq] / "labels" / f"{frame:06d}.label"
        return np.fromfile(path, dtype=np.int16).reshape(-1, 2)

    def semantic(self, seq, frame, sensor="velodyne"):
        return self.label_lut[self._labels(seq, frame)[:, 0]]

    def instances(self, seq, frame, sensor="velodyne"):
        return self._labels(seq, frame)[:, 1]
