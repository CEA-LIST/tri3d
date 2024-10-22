import dataclasses
from os import path
import pathlib
from typing import List, Tuple

import numpy as np

import PIL.Image

from ..geometry import (
    AffineTransform,
    CameraProjection,
    RigidTransform,
    Rotation,
    Translation,
)

from .dataset import Box, Dataset


@dataclasses.dataclass(frozen=True)
class KITTIObjectBox(Box):
    truncated: int
    occluded: int
    bbox: Tuple[float, float, float, float]


class KITTIObject(Dataset):
    """KITTI 3D object detection dataset.

    The sensors labels are `cam` and `img` for the left camera in 3D and homogeneous
    coordinates resp., and `velo` for the lidar.

    .. note::
       To match tri3D conventions, box annotation are modified as follows:

       - center is at the center of the box, not bottom
       - transform converts from tri3D object coordinates (x forward, z up)
         not kitti (x right-ward, z forward)
       - size is (length, width, height), not (height, length, width)

    `KITTI objects website <https://www.cvlibs.net/datasets/kitti/eval_3dobject.php>`_
    """

    cam_sensors = ["cam"]
    img_sensors = ["img2"]
    pcl_sensors = ["velo"]
    det_labels = [
        "Car",
        "Van",
        "Truck",
        "Pedestrian",
        "Person_sitting",
        "Cyclist",
        "Tram",
        "Misc",
        "Person",
        "DontCare",
    ]

    _default_cam_sensor = "cam"
    _default_pcl_sensor = "velo"
    _default_box_coords = "cam"

    def __init__(self, data_dir, split="training", label_map=None):
        if not path.exists(data_dir):
            raise ValueError("{} does not exist".format(data_dir))

        self.root = pathlib.Path(data_dir) / split
        self.label_map = {k: k for k in self.det_labels}
        if label_map is not None:
            self.label_map = {k: label_map[v] for k, v in label_map}

        self._annotation_cache = {}
        self._calib_cache = {}

        self.filenames = sorted(
            f.name.removesuffix(".bin") for f in (self.root / "velodyne").iterdir()
        )

    def _calibration(self, seq, src_sensor, dst_sensor):
        if src_sensor == "boxes":
            src_sensor = "cam"

        if src_sensor == dst_sensor:
            return RigidTransform([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        file = self.root / "calib" / (self.filenames[seq] + ".txt")
        calib = {}
        with open(file) as f:
            for line in f:
                if not line.strip():  # skip blank line
                    continue
                k, v_str = line.split(":", 1)
                calib[k] = np.fromstring(v_str, sep=" ").reshape((3, -1))

        velo2cam = Rotation.from_matrix(calib["R0_rect"]) @ RigidTransform.from_matrix(
            calib["Tr_velo_to_cam"]
        )
        cam2img2 = CameraProjection("pinhole", [1.0, 1.0, 0.0, 0.0]) @ AffineTransform(
            calib["P2"]
        )

        if src_sensor != "velo":
            return (
                self._calibration(seq, "velo", dst_sensor)
                @ self._calibration(seq, "velo", src_sensor).inv()
            )
        elif dst_sensor == "cam":
            return velo2cam
        elif dst_sensor == "img2":
            return cam2img2 @ velo2cam
        else:
            raise ValueError("unsupported sensor combination")

    def _points(self, seq, frame, sensor="velo"):
        if sensor != "velo":
            raise ValueError()

        file = self.root / "velodyne" / f"{self.filenames[seq]}.bin"
        return np.fromfile(file, dtype=np.float32).reshape(-1, 4)

    def _boxes(self, seq) -> List[KITTIObjectBox]:
        data = np.loadtxt(
            self.root / "label_2" / f"{self.filenames[seq]}.txt",
            dtype=[
                ("type", "U64"),
                ("truncated", "f4"),
                ("occluded", "i4"),
                ("bbox", "f4", 4),
                ("dimensions", "f4", 3),
                ("location", "f4", 3),
                ("rotation_y", "f4"),
            ],
            usecols=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ).reshape((-1,))

        annotations = []
        for i, a in enumerate(data):
            # convert to our conventions
            height, width, length = a["dimensions"]
            position = a["location"]
            position[1] -= height / 2  # center position vertically
            heading = a["rotation_y"]
            label = self.label_map[a["type"]]

            # switch to desired coordinate system
            obj2cam = (
                Translation(position)
                @ Rotation.from_euler("Y", heading)
                @ Rotation.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            )

            annotations.append(
                KITTIObjectBox(
                    frame=0,
                    uid=i,
                    center=obj2cam.apply([0, 0, 0]),
                    size=np.array([length, width, height]),
                    heading=heading,
                    transform=obj2cam,
                    label=label,
                    truncated=a["truncated"],
                    occluded=a["occluded"],
                    bbox=a["bbox"],
                )
            )

        return annotations

    def sequences(self):
        return list(range(len(self.filenames)))

    def frames(self, seq=None, sensor=None):
        if seq is None:
            return [(i, 0) for i in range(len(self.filenames))]
        else:
            return [(seq, 0)]

    def timestamps(self, seq, sensor):
        return np.zeros(1)

    def poses(self, seq, sensor, timeline=None):
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        pose = self._calibration(seq, sensor, "velo")
        return RigidTransform(pose.rotation.as_quat()[None], pose.translation.vec[None])

    def image(self, seq, frame, sensor="img2"):
        if sensor == "img2":
            return PIL.Image.open(
                path.join(self.root, "image_2", self.filenames[seq] + ".png")
            )
        else:
            raise ValueError()

    def rectangles(self, seq: int, frame: int):
        raise NotImplementedError
