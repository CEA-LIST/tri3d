import json
import pathlib

import numpy as np

import PIL.Image

from ..geometry import (
    CameraProjection,
    Rotation,
    RigidTransform,
    Translation,
)
from .dataset import Box, Dataset


class Once(Dataset):
    """`Once <https://once-for-auto-driving.github.io/index.html>`_ dataset.

    .. note::
       The lidar is rotated 90° clockwise so that x points toward the front of
       the ego car.
    """

    cam_sensors = [
        "cam01",
        "cam03",
        "cam05",
        "cam06",
        "cam07",
        "cam08",
        "cam09",
    ]
    img_sensors = [
        "img01",
        "img03",
        "img05",
        "img06",
        "img07",
        "img08",
        "img09",
    ]
    pcl_sensors = ["lidar_roof"]
    det_labels = ["Car", "Truck", "Bus", "Pedestrian", "Cyclist"]

    _default_img_sensor = "cam01"
    _default_pcl_sensor = "lidar_roof"
    _default_box_coords = "lidar_roof"

    def __init__(self, root, subset="train"):
        self.root = pathlib.Path(root) / subset / "data"
        self.scenes = sorted(d.name for d in self.root.iterdir())

        self._scene_calibs = []
        self._scene_frameids = []
        self._scene_timestamps = []
        self._scene_annotations = []
        self._scene_poses = []

        for seq in self.scenes:
            with open(self.root / seq / (seq + ".json"), "rb") as f:
                metadata = json.load(f)

            calibs = {}
            for cam, cam_calib in metadata["calib"].items():
                cam2velo = Rotation.from_euler(
                    "Z", np.pi / 2
                ) @ RigidTransform.from_matrix(cam_calib["cam_to_velo"])
                fx, _, cx, _, fy, cy, *_ = np.array(cam_calib["cam_intrinsic"]).ravel()
                k1, k2, p1, p2, k3 = cam_calib["distortion"]
                cam2img = CameraProjection(
                    "pinhole", (fx, fy, cx, cy, k1, k2, p1, p2, k3)
                )

                calibs[(cam, "lidar_roof")] = cam2velo
                calibs[("lidar_roof", cam)] = cam2velo.inv()
                calibs[(cam, cam.replace("cam", "img"))] = cam2img

            self._scene_calibs.append(calibs)

            pose_quats = []
            pose_vecs = []
            frameids = []
            timestamps = []
            box_frames = []
            box_labels = []
            box_centers = []
            box_sizes = []
            box_headings = []
            for i, frame in enumerate(metadata["frames"]):
                qx, qy, qz, qw = frame["pose"][:4]
                x, y, z = frame["pose"][4:]
                pose_quats.append([qw, qx, qy, qz])
                pose_vecs.append([x, y, z])
                frameids.append(frame["frame_id"])
                timestamps.append(int(frame["frame_id"]) / 1000)

                if "annos" not in frame or "names" not in frame["annos"]:
                    continue

                for label, box in zip(
                    frame["annos"]["names"], frame["annos"]["boxes_3d"]
                ):
                    box_frames.append(i)
                    box_labels.append(label)
                    box_centers.append(box[:3])
                    box_sizes.append(box[3:6])
                    box_headings.append(box[6])

            self._scene_poses.append(
                RigidTransform(pose_quats, pose_vecs)
                @ Rotation.from_euler("Z", -np.pi / 2)
            )
            self._scene_frameids.append(frameids)
            self._scene_timestamps.append(np.array(timestamps))
            box_transforms = (
                Rotation.from_euler("Z", np.pi / 2)
                @ Translation(box_centers)
                @ Rotation.from_euler("Z", box_headings)
            )
            self._scene_annotations.append(
                [
                    Box(
                        frame=frame,
                        uid=None,
                        center=center,
                        size=size,
                        heading=heading,
                        transform=transform,
                        label=label,
                    )
                    for frame, center, size, heading, transform, label in zip(
                        box_frames,
                        np.array(box_centers),
                        np.array(box_sizes),
                        box_headings,
                        box_transforms,
                        box_labels,
                    )
                ]
            )

    def _calibration(self, seq: int, src_sensor: str, dst_sensor: str):
        if src_sensor == "ego" or src_sensor == "boxes":
            src_sensor = "lidar_roof"

        if dst_sensor == "ego" or src_sensor == "boxes":
            dst_sensor = "lidar_roof"

        if src_sensor == dst_sensor:
            return Translation([0.0, 0.0, 0.0])

        if dst_sensor in self.img_sensors:
            cam = self.cam_sensors[self.img_sensors.index(dst_sensor)]
            src2cam = self._calibration(seq, src_sensor, cam)
            cam2img = self._scene_calibs[seq][(cam, dst_sensor)]
            return cam2img @ src2cam

        if src_sensor == "lidar_roof" or dst_sensor == "lidar_roof":
            return self._scene_calibs[seq][(src_sensor, dst_sensor)]

        velo2dst = self._calibration(seq, "lidar_roof", dst_sensor)
        src2velo = self._calibration(seq, src_sensor, "lidar_roof")
        return velo2dst @ src2velo

    def _poses(self, seq, sensor):
        if sensor == "ego":
            return self._scene_poses[seq]

        raise ValueError("Use imu pose to infer sensor poses.")

    def _points(self, seq: int, frame: int, sensor: str) -> np.ndarray:
        pcl = np.fromfile(
            self.root
            / self.scenes[seq]
            / sensor
            / (self._scene_frameids[seq][frame] + ".bin"),
            dtype=np.float32,
        ).reshape(-1, 4)
        pcl[:, 0], pcl[:, 1] = -pcl[:, 1], np.copy(pcl[:, 0])
        return pcl

    def _boxes(self, seq):
        return self._scene_annotations[seq]

    def sequences(self):
        return list(range(len(self.scenes)))

    def timestamps(self, seq: int, sensor: str):
        return self._scene_timestamps[seq]

    def image(self, seq: int, frame: int, sensor: str):
        return PIL.Image.open(
            self.root
            / self.scenes[seq]
            / self.cam_sensors[self.img_sensors.index(sensor)]
            / (self._scene_frameids[seq][frame] + ".jpg")
        )
