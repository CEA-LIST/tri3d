import dataclasses
import io
import pathlib
from typing import Sequence

import numpy as np
import pyarrow.feather as pf
from PIL import Image

from .. import geometry
from .dataset import Box, Dataset


@dataclasses.dataclass(frozen=True)
class ArgoverseBox(Box):
    num_interior_pts: int


class Argoverse2(Dataset):
    """`Argoverse 2 dataset <https://www.argoverse.org/av2.html>`_.

    .. run::

        import matplotlib.pyplot as plt
        import numpy as np
        from tri3d.datasets import Argoverse

        plt.switch_backend("Agg")

        dataset = Argoverse("datasets/argoverse2")
        name = "tri3d.datasets.Argoverse"
        camera, imgcoords, lidar = "cam_ring_front_center", "img_ring_front_center", "lidar"
        seq, frame, cam_frame = 0, 100, 200
    """

    _default_cam_sensor = "cam_ring_front_center"
    _default_pcl_sensor = "lidar"
    _default_box_coords = "lidar"

    cam_sensors = [
        "cam_ring_front_center",
        "cam_ring_front_left",
        "cam_ring_front_right",
        "cam_ring_rear_left",
        "cam_ring_rear_right",
        "cam_ring_side_left",
        "cam_ring_side_right",
        # "cam_stereo_front_left",
        # "cam_stereo_front_right",
    ]

    img_sensors = [
        "img_ring_front_center",
        "img_ring_front_left",
        "img_ring_front_right",
        "img_ring_rear_left",
        "img_ring_rear_right",
        "img_ring_side_left",
        "img_ring_side_right",
        # "img_stereo_front_left",
        # "img_stereo_front_right",
    ]

    pcl_sensors = [
        "lidar",
    ]

    sensors = cam_sensors + pcl_sensors
    det_labels = [
        "ANIMAL",
        "ARTICULATED_BUS",
        "BICYCLE",
        "BICYCLIST",
        "BOLLARD",
        "BOX_TRUCK",
        "BUS",
        "CONSTRUCTION_BARREL",
        "CONSTRUCTION_CONE",
        "DOG",
        "LARGE_VEHICLE",
        "MESSAGE_BOARD_TRAILER",
        "MOBILE_PEDESTRIAN_SIGN",
        "MOTORCYCLE",
        "MOTORCYCLIST",
        "OFFICIAL_SIGNALER",
        "PEDESTRIAN",
        "RAILED_VEHICLE",
        "REGULAR_VEHICLE",
        "SCHOOL_BUS",
        "SIGN",
        "STOP_SIGN",
        "STROLLER",
        "TRAFFIC_LIGHT_TRAILER",
        "TRUCK",
        "TRUCK_CAB",
        "VEHICULAR_TRAILER",
        "WHEELCHAIR",
        "WHEELED_DEVICE",
        "WHEELED_RIDER",
    ]
    sem_labels = []
    sem2d_labels = []

    def __init__(self, root, split="train") -> None:
        self.root = pathlib.Path(root)
        self.split = split
        self.records = sorted((self.root / "sensor" / split).iterdir())

        self.timelines = []

        for r in self.records:
            record_timelines = {}

            record_timelines["lidar"] = np.sort(
                [int(f.stem) for f in (r / "sensors" / "lidar").iterdir()]
            )

            for c in self.cam_sensors:
                record_timelines[c] = np.sort(
                    [
                        int(f.stem)
                        for f in (
                            r / "sensors" / "cameras" / c.removeprefix("cam_")
                        ).iterdir()
                    ]
                )
                record_timelines[c.replace("cam_", "img_")] = record_timelines[c]

            ego_pose_ts = pf.read_table(
                r / "city_SE3_egovehicle.feather", columns=["timestamp_ns"]
            )["timestamp_ns"].to_numpy()
            record_timelines["ego_pose"] = ego_pose_ts

            self.timelines.append(record_timelines)

    # @misc.memoize_method()
    def _calibration(self, seq, src_sensor, dst_sensor):
        if src_sensor in self.pcl_sensors or src_sensor == "boxes":
            src_sensor = "ego_pose"
        if dst_sensor in self.pcl_sensors or src_sensor == "boxes":
            dst_sensor = "ego_pose"

        # easy case
        if src_sensor == dst_sensor:
            return geometry.Translation([0.0, 0.0, 0.0])

        # decompose src -> img into src -> cam -> img
        elif dst_sensor in self.img_sensors:
            record = self.records[seq]
            camera_calibration = pf.read_table(
                record / "calibration" / "intrinsics.feather"
            ).to_pydict()
            idx = camera_calibration["sensor_name"].index(
                dst_sensor.removeprefix("img_")
            )
            fx = camera_calibration["fx_px"][idx]
            fy = camera_calibration["fy_px"][idx]
            cx = camera_calibration["cx_px"][idx]
            cy = camera_calibration["cy_px"][idx]
            k1 = camera_calibration["k1"][idx]
            k2 = camera_calibration["k2"][idx]
            k3 = camera_calibration["k3"][idx]
            w = camera_calibration["width_px"][idx]
            h = camera_calibration["height_px"][idx]

            cam2img = geometry.CameraProjection(
                "pinhole", (fx, fy, cx, cy, k1, k2, 0.0, 0.0, k3), w, h
            )

            cam_sensor = self.cam_sensors[self.img_sensors.index(dst_sensor)]
            src2cam = self._calibration(seq, src_sensor, cam_sensor)

            return cam2img @ src2cam

        # decompose src -> dst into src -> ego_pose -> dst
        elif dst_sensor != "ego_pose":
            src2ego = self._calibration(seq, src_sensor, "ego_pose")
            ego2dst = self._calibration(seq, dst_sensor, "ego_pose").inv()
            return ego2dst @ src2ego

        # src -> ego
        record = self.records[seq]
        extrinsics = pf.read_table(
            record / "calibration" / "egovehicle_SE3_sensor.feather"
        ).to_pydict()

        idx = extrinsics["sensor_name"].index(src_sensor.removeprefix("cam_"))
        qw = extrinsics["qw"][idx]
        qx = extrinsics["qx"][idx]
        qy = extrinsics["qy"][idx]
        qz = extrinsics["qz"][idx]
        tx = extrinsics["tx_m"][idx]
        ty = extrinsics["ty_m"][idx]
        tz = extrinsics["tz_m"][idx]
        return geometry.RigidTransform([qw, qx, qy, qz], [tx, ty, tz])

    def _poses(self, seq, sensor) -> geometry.RigidTransform:
        if sensor == "boxes":
            sensor = "lidar"

        record = self.records[seq]
        data = pf.read_table(record / "city_SE3_egovehicle.feather")

        frames = np.searchsorted(data["timestamp_ns"], self.timelines[seq][sensor])

        rot = np.stack(
            [
                data["qw"].to_numpy(),
                data["qx"].to_numpy(),
                data["qy"].to_numpy(),
                data["qz"].to_numpy(),
            ],
            axis=1,
        )[frames]
        pos = np.stack(
            [
                data["tx_m"].to_numpy(),
                data["ty_m"].to_numpy(),
                data["tz_m"].to_numpy(),
            ],
            axis=1,
        )[frames]
        ego_poses = geometry.RigidTransform(rot, pos)
        sensor2ego: geometry.RigidTransform = self._calibration(seq, sensor, "ego_pose")  # type: ignore

        return ego_poses @ sensor2ego

    def _points(self, seq, frame, sensor):
        record = self.records[seq]
        ts = self.timelines[seq]["lidar"][frame]

        data = pf.read_feather(record / "sensors" / "lidar" / f"{ts}.feather")

        xyz = np.stack(
            [
                data["x"].to_numpy(),
                data["y"].to_numpy(),
                data["z"].to_numpy(),
                data["intensity"].to_numpy().astype(np.float32),
            ],
            axis=1,
        )

        return xyz

    def _boxes(self, seq) -> Sequence[ArgoverseBox]:
        record = self.records[seq]

        data = pf.read_table(record / "annotations.feather")

        timestamps = data["timestamp_ns"].to_numpy()
        frames = np.searchsorted(self.timelines[seq]["lidar"], timestamps)
        uids = [
            int(u.replace("-", "")[:16], 16) for u in data["track_uuid"].to_pylist()
        ]
        labels = data["category"].to_pylist()
        sizes = np.stack(
            [
                data["length_m"].to_numpy(),
                data["width_m"].to_numpy(),
                data["height_m"].to_numpy(),
            ],
            axis=1,
        )
        centers = np.stack(
            [data["tx_m"].to_numpy(), data["ty_m"].to_numpy(), data["tz_m"].to_numpy()],
            axis=1,
        )
        rotations = np.stack(
            [
                data["qw"].to_numpy(),
                data["qx"].to_numpy(),
                data["qy"].to_numpy(),
                data["qz"].to_numpy(),
            ],
            axis=1,
        )
        transforms = geometry.RigidTransform(rotations, centers)
        headings = transforms.rotation.as_euler("ZYX")[:, 0]
        num_interior_pts = data["num_interior_pts"].to_numpy()

        return list(
            map(
                ArgoverseBox,
                frames,
                uids,
                centers,
                sizes,
                headings,
                transforms,
                labels,
                num_interior_pts,
            )
        )

    def sequences(self):
        return list(range(len(self.records)))

    def timestamps(self, seq, sensor):
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]
        elif sensor == "boxes":
            sensor = "lidar"

        return self.timelines[seq][sensor]

    def image(self, seq, frame, sensor):
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        record = self.records[seq]
        ts = self.timelines[seq][sensor][frame]

        return Image.open(
            record / "sensors" / "cameras" / sensor.removeprefix("cam_") / f"{ts}.jpg"
        )
