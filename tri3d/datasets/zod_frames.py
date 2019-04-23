import itertools
import json
import pathlib
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
from PIL import Image

from .. import geometry, misc
from .dataset import Box, Dataset


class ZODFrames(Dataset):
    """Zenseact Open dataset."""

    cam_sensors = ["front"]
    img_sensors = ["img_front"]
    pcl_sensors = ["velodyne"]
    det_labels = [
        "Animal",
        "DynamicBarrier",
        "Pedestrian",
        "PoleObject",
        "TrafficBeacon",
        "TrafficGuide",
        "TrafficSign",
        "TrafficSignal",
        "Vehicle",
        "VulnerableVehicle",
    ]
    sem_labels = None
    _default_cam_sensor = "front"
    _default_pcl_sensor = "velodyne"
    _default_box_coords = "velodyne"

    def __init__(self, root, metadata, split, anon_method="dnat"):
        self.root = pathlib.Path(root)
        self.anon_method = anon_method

        with open(metadata, "rb") as f:
            self.metadata = json.load(f)[split]

        self._timelines = []
        self._ego_poses = []

        for seq in self.metadata:
            seq_timelines = {}

            for name, frames in seq["camera_frames"].items():
                if not name.endswith(self.anon_method):
                    continue

                name = name.removesuffix("_" + self.anon_method)

                seq_timelines[name] = np.array(
                    [datetime.fromisoformat(f["time"]).timestamp() for f in frames]
                )
                seq_timelines["img_" + name] = seq_timelines[name]

            for name, frames in seq["lidar_frames"].items():
                seq_timelines[name] = np.array(
                    [datetime.fromisoformat(f["time"]).timestamp() for f in frames]
                )

            with open(
                self.root / "single_frames" / seq["id"] / "ego_motion.json", "rb"
            ) as f:
                ego_motion = json.load(f)

            self._ego_poses.append(
                geometry.RigidTransform.from_matrix(ego_motion["poses"])
            )

            seq_timelines["ego"] = np.array(ego_motion["timestamps"])
            self._timelines.append(seq_timelines)

        self._keyframes = []
        for i, seq in enumerate(self.metadata):
            self._timelines[i]["keyframes"] = datetime.fromisoformat(
                seq["keyframe_time"]
            ).timestamp()

    def _calibration(self, seq, src_sensor, dst_sensor):
        if src_sensor == dst_sensor:
            return geometry.Translation([0.0, 0.0, 0.0])

        prefix = self.root / "single_frames" / self.metadata[seq]["id"]
        with open(prefix / "calibration.json", "rb") as f:
            calibration = json.load(f)

        if dst_sensor in self.img_sensors:
            intrinsics = (
                calibration["FC"]["intrinsics"][0][0],
                calibration["FC"]["intrinsics"][1][1],
                calibration["FC"]["intrinsics"][0][2],
                calibration["FC"]["intrinsics"][1][2],
                *calibration["FC"]["distortion"],
            )
            cam2img = geometry.CameraProjection("kannala", intrinsics)

            cam = self.cam_sensors[self.img_sensors.index(dst_sensor)]
            src2cam = self._calibration(seq, src_sensor, cam)

            return cam2img @ src2cam

        elif src_sensor == "ego":
            return self._calibration(seq, dst_sensor, src_sensor).inv()

        elif dst_sensor != "ego":  # break into src -> ego -> dst
            return self._calibration(seq, dst_sensor, "ego").inv() @ self._calibration(
                seq, src_sensor, "ego"
            )

        elif src_sensor == "front":
            return geometry.RigidTransform.from_matrix(calibration["FC"]["extrinsics"])

        elif src_sensor == "velodyne":
            return geometry.RigidTransform.from_matrix(
                calibration["FC"]["lidar_extrinsics"]
            )

        else:
            raise ValueError()

    def _poses(self, seq, sensor):
        if sensor == "velodyne" or sensor == "boxes":
            return self._ego_poses[seq]

        # interp to timestamps
        ego2world = self._ego_poses[seq]
        ego_ts = self._timelines[seq]["ego"]
        sensor2ego = self._calibration(seq, sensor, "velodyne")
        sensor_ts = self._timelines[seq][sensor]

        i1, i2 = misc.lr_bisect(ego_ts, sensor_ts)
        t1 = ego_ts[i1]
        t2 = ego_ts[i2]
        alpha = (t2 - sensor_ts) / (t2 - t1).clip(min=1e-6)
        return (
            geometry.RigidTransform.interpolate(ego2world[i1], ego2world[i2], alpha)
            @ sensor2ego
        )

    def _points(self, seq, frame, sensor):
        filepath = self.metadata[seq]["lidar_frames"][sensor][frame]["filepath"]
        points = np.load(self.root / filepath)
        xyz = np.stack([points["x"], points["y"], points["z"]], axis=-1)
        return xyz

    def _boxes(self, seq):
        filepath = self.metadata[seq]["annotations"]["object_detection"]["filepath"]
        with open(self.root / filepath, "rb") as f:
            annotations = json.load(f)

        out = []
        for ann in annotations:
            props = ann["properties"]

            label = props["class"]
            if label not in self.det_labels:
                print(label + " not in det_labels")
                continue

            if "location_3d" not in props:
                continue

            uuid = props["annotation_uuid"]
            pos = np.array(props["location_3d"]["coordinates"])
            rot = np.array(
                [
                    props["orientation_3d_qw"],
                    props["orientation_3d_qx"],
                    props["orientation_3d_qy"],
                    props["orientation_3d_qz"],
                ]
            )
            size = np.array(
                [
                    props["size_3d_length"],
                    props["size_3d_width"],
                    props["size_3d_height"],
                ]
            )

            transform = geometry.RigidTransform(rot, pos)

            out.append(
                Box(
                    frame=10,
                    uid=uuid,
                    center=transform.apply([0, 0, 0]),
                    size=size,
                    heading=transform.rotation.as_euler("ZYX")[0],
                    transform=transform,
                    label=label,
                )
            )

        return out

    def sequences(self) -> List[int]:
        """Return the list of sequences/recordings indices (0..num_sequences)."""
        return list(range(len(self.meta)))

    def timestamps(self, seq, sensor):
        if sensor == "boxes":
            sensor = "velodyne"

        return self._timelines[seq][sensor]

    def image(self, seq: int, frame: int = 0, sensor: str = "front") -> Image:
        sensor = sensor + "_" + self.anon_method
        filepath = self.metadata[seq]["camera_frames"][sensor][frame]["filepath"]
        return Image.open(self.root / filepath)