import dataclasses
import io
import pathlib
from typing import List, Optional

import numpy as np
import pyarrow.parquet as pq

from PIL import Image

from .. import geometry

from .dataset import Box, Dataset


def lidar_pose_to_mat(lidar_pose):
    roll = lidar_pose[..., 0]
    pitch = lidar_pose[..., 1]
    yaw = lidar_pose[..., 2]
    translation = lidar_pose[..., 3:]

    ca = np.cos(yaw)
    sa = np.sin(yaw)
    cb = np.cos(pitch)
    sb = np.sin(pitch)
    cc = np.cos(roll)
    sc = np.sin(roll)

    rotation = np.moveaxis(np.zeros((9,) + ca.shape), 0, -1).reshape(ca.shape + (3, 3))

    rotation[..., 0, 0] = ca * cb
    rotation[..., 0, 1] = ca * sb * sc - sa * cc
    rotation[..., 0, 2] = ca * sb * cc - ca * sc
    rotation[..., 1, 0] = sa * cb
    rotation[..., 1, 1] = sa * sb * sc + ca * cc
    rotation[..., 1, 2] = sa * sb * cc - ca * sc
    rotation[..., 2, 0] = -sb
    rotation[..., 2, 1] = cb * sc
    rotation[..., 2, 2] = cb * cc

    return rotation, translation


@dataclasses.dataclass(frozen=True)
class WaymoBox(Box):
    num_lidar_points_in_box: int
    num_top_lidar_points_in_box: int
    difficulty_level_det: int
    difficulty_level_trk: int
    speed: np.ndarray
    acceleration: np.ndarray


class Waymo(Dataset):
    """`Waymo Open dataset <https://www.waymo.com/open>`_ (parquet file format).

    .. note::
       LiDAR timestamps are increased by a +0.05 so that they correspond to the
       middle of the sweep instead of the beginning.

    The timestamps for cameras refer to the pose timestamp in raw waymo data.
    Other temporal information such as trigger timestamps are not exposed.

    :ref:`instances2d` returns global ids[1]_ which are consistent between 
    frames. They are not common with the tracking and 3D annotations.

    :meth:`Waymo.frames` supports an extra sensor name `'SEG_LIDAR_TOP'` where
    it returns lidar frames for which 3D segmentation is available.

    .. [1] https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L285
    """

    _default_cam_sensor = "CAM_FRONT"
    _default_pcl_sensor = "LIDAR_TOP"
    _default_box_coords = "LIDAR_TOP"

    cam_sensors = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_SIDE_LEFT",
        "CAM_SIDE_RIGHT",
        "CAM_REAR_LEFT",
        "CAM_REAR",
        "CAM_REAR_RIGHT",
    ]

    img_sensors = [
        "IMG_FRONT",
        "IMG_FRONT_LEFT",
        "IMG_FRONT_RIGHT",
        "IMG_SIDE_LEFT",
        "IMG_SIDE_RIGHT",
        "IMG_REAR_LEFT",
        "IMG_REAR",
        "IMG_REAR_RIGHT",
    ]

    pcl_sensors = [
        "LIDAR_TOP",
        "LIDAR_FRONT",
        "LIDAR_SIDE_LEFT",
        "LIDAR_SIDE_RIGHT",
        "LIDAR_REAR",
    ]

    sensors = cam_sensors + pcl_sensors
    det_labels = ["UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST"]
    sem_labels = [
        "UNDEFINED",
        "CAR",
        "TRUCK",
        "BUS",
        "OTHER_VEHICLE",
        "MOTORCYCLIST",
        "BICYCLIST",
        "PEDESTRIAN",
        "SIGN",
        "TRAFFIC_LIGHT",
        "POLE",
        "CONSTRUCTION_CONE",
        "BICYCLE",
        "MOTORCYCLE",
        "BUILDING",
        "VEGETATION",
        "TREE_TRUNK",
        "CURB",
        "ROAD",
        "LANE_MARKER",
        "OTHER_GROUND",
        "WALKABLE",
        "SIDEWALK",
    ]
    sem2d_labels = [
        "UNDEFINED",
        "EGO_VEHICLE",
        "CAR",
        "TRUCK",
        "BUS",
        "OTHER_LARGE_VEHICLE",
        "BICYCLE",
        "MOTORCYCLE",
        "TRAILER",
        "PEDESTRIAN",
        "CYCLIST",
        "MOTORCYCLIST",
        "BIRD",
        "GROUND_ANIMAL",
        "CONSTRUCTION_CONE_POLE",
        "POLE",
        "PEDESTRIAN_OBJECT",
        "SIGN",
        "TRAFFIC_LIGHT",
        "BUILDING",
        "ROAD",
        "LANE_MARKER",
        "ROAD_MARKER",
        "SIDEWALK",
        "VEGETATION",
        "SKY",
        "GROUND",
        "DYNAMIC",
        "STATIC",
    ]

    def __init__(self, root, split="training") -> None:
        self.root = pathlib.Path(root)
        self.split = split
        self.records = sorted(
            a.name.removesuffix(".parquet")
            for a in (self.root / split / "camera_box").iterdir()
        )

        self.timelines = []

        for r in self.records:
            record_timelines = {}

            for i, sensor in enumerate(self.pcl_sensors, start=1):
                record_timelines[sensor] = (
                    pq.read_table(
                        self.root / self.split / "lidar" / (r + ".parquet"),
                        filters=[("key.laser_name", "=", i)],
                        columns=["key.frame_timestamp_micros"],
                    )["key.frame_timestamp_micros"]
                    .sort()
                    .to_numpy()
                    / 1e6
                    + 0.05
                )

                record_timelines["SEG_" + sensor] = (
                    pq.read_table(
                        self.root
                        / self.split
                        / "lidar_segmentation"
                        / (r + ".parquet"),
                        filters=[("key.laser_name", "=", i)],
                        columns=["key.frame_timestamp_micros"],
                    )["key.frame_timestamp_micros"]
                    .sort()
                    .to_numpy()
                    / 1e6
                    + 0.05
                )

            for i, c in enumerate(self.cam_sensors, start=1):
                record_timelines[c] = (
                    pq.read_table(
                        self.root / self.split / "camera_image" / (r + ".parquet"),
                        filters=[("key.camera_name", "=", i)],
                        columns=["[CameraImageComponent].pose_timestamp"],
                    )["[CameraImageComponent].pose_timestamp"]
                    .sort()
                    .to_numpy()
                )

                record_timelines[c.replace("CAM_", "IMG_")] = record_timelines[c]

                record_timelines["SEG_" + c.replace("CAM_", "IMG_")] = (
                    pq.read_table(
                        self.root
                        / self.split
                        / "camera_segmentation"
                        / (r + ".parquet"),
                        filters=[("key.camera_name", "=", 1)],
                        columns=["key.frame_timestamp_micros"],
                    )["key.frame_timestamp_micros"]
                    .sort()
                    .to_numpy()
                    / 1e6
                )

            record_timelines["boxes"] = record_timelines["LIDAR_TOP"]

            self.timelines.append(record_timelines)

    # @misc.memoize_method()
    def _calibration(self, seq, src_sensor, dst_sensor):
        if src_sensor in self.pcl_sensors or src_sensor == "boxes":
            src_sensor = "LIDAR_TOP"
        if dst_sensor in self.pcl_sensors or src_sensor == "boxes":
            dst_sensor = "LIDAR_TOP"

        # easy case
        if src_sensor == dst_sensor:
            return geometry.Translation([0, 0, 0])

        # decompose src -> img into src -> cam -> img
        elif dst_sensor in self.img_sensors:
            record = self.records[seq]
            camera_calibration = pq.read_table(
                self.root / self.split / "camera_calibration" / (record + ".parquet"),
                filters=[
                    ("key.camera_name", "=", 1 + self.cam_sensors.index(src_sensor))
                ],
            )
            calib = camera_calibration.to_pylist()[0]
            fx, fy, cx, cy, k1, p1, p2 = [
                calib["[CameraCalibrationComponent].intrinsic." + field]
                for field in ["f_u", "f_v", "c_u", "c_v", "k1", "p1", "p2"]
            ]
            cam2img = geometry.CameraProjection(
                "pinhole", (fx, fy, cx, cy, k1, 0.0, p1, p2)
            )

            cam_sensor = self.cam_sensors[self.img_sensors.index(dst_sensor)]
            src2cam = self._calibration(seq, src_sensor, cam_sensor)

            return cam2img @ src2cam

        # decompose cam -> dst into cam -> lidar -> dst
        elif src_sensor in self.cam_sensors:
            record = self.records[seq]
            camera_calibration = pq.read_table(
                self.root / self.split / "camera_calibration" / (record + ".parquet"),
                filters=[
                    ("key.camera_name", "=", 1 + self.cam_sensors.index(src_sensor))
                ],
            )
            cam2lidar = (
                camera_calibration["[CameraCalibrationComponent].extrinsic.transform"]
                .to_numpy()[0]
                .reshape(4, 4)
            )
            # waymo -> kitti axis convention for cameras
            permute = np.array(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
            cam2lidar = cam2lidar @ permute
            cam2lidar = geometry.RigidTransform.from_matrix(cam2lidar)

            lidar2dst = self._calibration(seq, "LIDAR_TOP", dst_sensor)
            return lidar2dst @ cam2lidar

        # src -> cam
        elif dst_sensor in self.cam_sensors:
            return self._calibration(seq, dst_sensor, src_sensor).inv()

        else:
            raise ValueError("invalid or unsupported coords combination")

    # @misc.memoize_method(maxsize=1000)
    def _poses(self, seq, sensor):
        if sensor == "boxes":
            sensor = "LIDAR_TOP"

        if sensor in self.pcl_sensors:
            # LIDARs use same frame timestamp as vehicle_pose pose
            record = self.records[seq]
            data = pq.read_table(
                self.root / self.split / "vehicle_pose" / (record + ".parquet"),
                columns=["[VehiclePoseComponent].world_from_vehicle.transform"],
            )
            poses = (
                data["[VehiclePoseComponent].world_from_vehicle.transform"]
                .combine_chunks()
                .flatten()
                .to_numpy()
                .reshape(-1, 4, 4)
            )
            return geometry.RigidTransform.from_matrix(poses)

        if sensor in self.cam_sensors:
            record = self.records[seq]
            data = pq.read_table(
                self.root / self.split / "camera_image" / (record + ".parquet"),
                filters=[("key.camera_name", "=", self.cam_sensors.index(sensor) + 1)],
                columns=["[CameraImageComponent].pose.transform"],
            )
            poses = (
                data["[CameraImageComponent].pose.transform"]
                .combine_chunks()
                .flatten()
                .to_numpy()
                .reshape(-1, 4, 4)
            )
            return geometry.RigidTransform.from_matrix(poses) @ self._calibration(
                seq, sensor, "LIDAR_TOP"
            )

        raise ValueError()

    def _lidar_returns(self, seq, frame, sensor="LIDAR_TOP"):
        record = self.records[seq]
        timestamp = self.timelines[seq][sensor][frame] - 0.05  # sweep middle -> start
        laser_name = 1 + self.pcl_sensors.index(sensor)

        lidar = pq.read_table(
            self.root / self.split / "lidar" / (record + ".parquet"),
            filters=[
                ("key.laser_name", "=", laser_name),
                ("key.frame_timestamp_micros", "=", int(timestamp * 1e6)),
            ],
        )

        h, w, d = lidar["[LiDARComponent].range_image_return1.shape"][0].as_py()
        r1 = (
            lidar["[LiDARComponent].range_image_return1.values"]
            .to_numpy()[0]
            .reshape((h, w, d))
        )
        r2 = (
            lidar["[LiDARComponent].range_image_return2.values"]
            .to_numpy()[0]
            .reshape((h, w, d))
        )

        return r1, r2

    def _points(self, seq, frame, sensor):
        record = self.records[seq]
        timestamp = self.timelines[seq][sensor][frame] - 0.05  # sweep middle -> start
        laser_name = 1 + self.pcl_sensors.index(sensor)

        # Laser data
        return1, return2 = self._lidar_returns(seq, frame, sensor)
        h, w, _ = return1.shape

        # Calibration data
        lidar_calibration = pq.read_table(
            self.root / self.split / "lidar_calibration" / (record + ".parquet"),
            filters=[("key.laser_name", "=", laser_name)],
        )

        extrinsic = (
            lidar_calibration["[LiDARCalibrationComponent].extrinsic.transform"]
            .to_numpy()[0]
            .reshape(4, 4)
            .astype(np.float32)
        )

        elevation = lidar_calibration[
            "[LiDARCalibrationComponent].beam_inclination.values"
        ]
        if elevation.is_null()[0].as_py():
            a = lidar_calibration["[LiDARCalibrationComponent].beam_inclination.min"][
                0
            ].as_py()
            b = lidar_calibration["[LiDARCalibrationComponent].beam_inclination.max"][
                0
            ].as_py()
            step = (b - a) / h
            elevation = np.linspace(a + step / 2, b - step / 2, h, dtype=np.float32)
        else:
            elevation = elevation.to_numpy()[0].astype(np.float32)

        elevation = np.flip(elevation).reshape(-1, 1)

        step = 2 * np.pi / w
        azimuth = np.linspace(
            np.pi - step / 2, -np.pi + step / 2, w, dtype=np.float32
        ).reshape(1, w)
        az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
        azimuth -= az_correction

        # Range image -> XYZ
        x1 = return1[:, :, 0] * np.cos(elevation) * np.cos(azimuth)
        y1 = return1[:, :, 0] * np.cos(elevation) * np.sin(azimuth)
        z1 = return1[:, :, 0] * np.sin(elevation)
        xyz1 = np.stack([x1, y1, z1], axis=-1)

        x2 = return2[:, :, 0] * np.cos(elevation) * np.cos(azimuth)
        y2 = return2[:, :, 0] * np.cos(elevation) * np.sin(azimuth)
        z2 = return2[:, :, 0] * np.sin(elevation)
        xyz2 = np.stack([x2, y2, z2], axis=-1)

        # Apply extrinsic transform sensor -> pose
        xyz1 = xyz1 @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        xyz2 = xyz2 @ extrinsic[:3, :3].T + extrinsic[:3, 3]

        # Pixel pose (=compensation for pose movement during sweep)
        if sensor == "LIDAR_TOP":
            lidar_pose = pq.read_table(
                self.root / self.split / "lidar_pose" / (record + ".parquet"),
                filters=[("key.frame_timestamp_micros", "=", int(timestamp * 1e6))],
            )
            shape = lidar_pose["[LiDARPoseComponent].range_image_return1.shape"][
                0
            ].as_py()
            pixel_pose = (
                lidar_pose["[LiDARPoseComponent].range_image_return1.values"]
                .to_numpy()[0]
                .reshape(shape)
            )
            rot, vec = lidar_pose_to_mat(pixel_pose)

            xyz1 = np.einsum("...j,...ij->...i", xyz1, rot) + vec
            xyz2 = np.einsum("...j,...ij->...i", xyz2, rot) + vec

            # Pixel pose includes world -> LIDAR_TOP, so we undo that
            vehicle_pose_inv = self._poses(seq, "LIDAR_TOP")[frame].inv()
            xyz1 = vehicle_pose_inv.apply(xyz1)
            xyz2 = vehicle_pose_inv.apply(xyz2)

        m1i, m1j = np.nonzero(return1[:, :, 0] > 0)
        m2i, m2j = np.nonzero(return2[:, :, 0] > 0)
        xyz1 = xyz1[m1i, m1j]
        xyz2 = xyz2[m2i, m2j]

        xyz = np.concatenate([xyz1, xyz2])

        return xyz

    def _boxes(self, seq) -> List[WaymoBox]:
        record = self.records[seq]

        lidar_box = pq.read_table(
            self.root / self.split / "lidar_box" / (record + ".parquet"),
        )

        keyframe_ts = self.timelines[seq]["boxes"]
        box_ts = lidar_box["key.frame_timestamp_micros"].to_numpy() / 1e6 + 0.05
        box_frames = np.searchsorted(keyframe_ts, box_ts, side="left")

        laser_object_id = lidar_box["key.laser_object_id"].to_numpy()

        center = np.stack(
            [
                lidar_box["[LiDARBoxComponent].box.center.x"].to_numpy(),
                lidar_box["[LiDARBoxComponent].box.center.y"].to_numpy(),
                lidar_box["[LiDARBoxComponent].box.center.z"].to_numpy(),
            ],
            axis=1,
        )

        size = np.stack(
            [
                lidar_box["[LiDARBoxComponent].box.size.x"].to_numpy(),
                lidar_box["[LiDARBoxComponent].box.size.y"].to_numpy(),
                lidar_box["[LiDARBoxComponent].box.size.z"].to_numpy(),
            ],
            axis=1,
        )

        heading = lidar_box["[LiDARBoxComponent].box.heading"].to_numpy()

        speed = np.stack(
            [
                lidar_box["[LiDARBoxComponent].speed.x"].to_numpy(),
                lidar_box["[LiDARBoxComponent].speed.y"].to_numpy(),
                lidar_box["[LiDARBoxComponent].speed.z"].to_numpy(),
            ],
            axis=1,
        )

        acceleration = np.stack(
            [
                lidar_box["[LiDARBoxComponent].acceleration.x"].to_numpy(),
                lidar_box["[LiDARBoxComponent].acceleration.y"].to_numpy(),
                lidar_box["[LiDARBoxComponent].acceleration.z"].to_numpy(),
            ],
            axis=1,
        )

        label = lidar_box["[LiDARBoxComponent].type"].to_numpy()
        label = [self.det_labels[i] for i in label]

        num_lidar_points_in_box = lidar_box[
            "[LiDARBoxComponent].num_lidar_points_in_box"
        ].to_numpy()
        num_top_lidar_points_in_box = lidar_box[
            "[LiDARBoxComponent].num_top_lidar_points_in_box"
        ].to_numpy()
        difficulty_level_detection = lidar_box[
            "[LiDARBoxComponent].difficulty_level.detection"
        ].to_numpy()
        difficulty_level_tracking = lidar_box[
            "[LiDARBoxComponent].difficulty_level.tracking"
        ].to_numpy()

        transform = geometry.RigidTransform(
            geometry.Rotation.from_euler("Z", heading), center
        )

        out = []
        for i in range(len(label)):
            out.append(
                WaymoBox(
                    frame=box_frames[i],
                    uid=laser_object_id[i],
                    center=center[i],
                    size=size[i],
                    heading=heading[i],
                    transform=transform[i],
                    label=label[i],
                    num_lidar_points_in_box=num_lidar_points_in_box[i],
                    num_top_lidar_points_in_box=num_top_lidar_points_in_box[i],
                    difficulty_level_det=difficulty_level_detection[i],
                    difficulty_level_trk=difficulty_level_tracking[i],
                    speed=speed[i],
                    acceleration=acceleration[i],
                )
            )

        return out

    def sequences(self):
        return list(range(len(self.records)))

    def timestamps(self, seq: Optional[int] = None, sensor="LIDAR_TOP"):
        if sensor == "boxes":
            sensor = "LIDAR_TOP"
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        return self.timelines[seq][sensor]

    def image(self, seq, frame, sensor="CAM_FRONT"):
        """Return image from given camera at given frame.

        A default sensor (for instance a front facing camera) should be
        provided for convenience.
        """
        record = self.records[seq]
        pose_timestamp = self.timelines[seq][sensor][frame]
        camera_name = 1 + self.cam_sensors.index(sensor.replace("IMG", "CAM"))

        data = pq.read_table(
            self.root / self.split / "camera_image" / (record + ".parquet"),
            filters=[
                ("key.camera_name", "=", camera_name),
                ("[CameraImageComponent].pose_timestamp", "=", pose_timestamp),
            ],
        )

        data = data["[CameraImageComponent].image"][0].as_buffer()

        return Image.open(io.BytesIO(data))

    def _panoptic(self, seq: int, frame: int, sensor="LIDAR_TOP"):
        record = self.records[seq]
        timestamp = self.timelines[seq][sensor][frame] - 0.05
        laser_name = self.pcl_sensors.index(sensor) + 1

        return1, return2 = self._lidar_returns(seq, frame, sensor)
        h, w, _ = return1.shape

        segmentation = pq.read_table(
            self.root / self.split / "lidar_segmentation" / (record + ".parquet"),
            filters=[
                ("key.frame_timestamp_micros", "=", int(timestamp * 1e6)),
                ("key.laser_name", "=", laser_name),
            ],
        )

        segmentation1 = (
            segmentation["[LiDARSegmentationLabelComponent].range_image_return1.values"]
            .to_numpy()[0]
            .reshape(h, w, 2)
        )
        segmentation2 = (
            segmentation["[LiDARSegmentationLabelComponent].range_image_return2.values"]
            .to_numpy()[0]
            .reshape(h, w, 2)
        )

        return segmentation1[return1[:, :, 0] > 0], segmentation2[return2[:, :, 0] > 0]

    def semantic(self, seq: int, frame: int, sensor="LIDAR_TOP"):
        return1, return2 = self._panoptic(seq, frame, sensor)
        return np.concatenate([return1[:, 1], return2[:, 1]])

    def instances(self, seq: int, frame: int, sensor="LIDAR_TOP"):
        return1, return2 = self._panoptic(seq, frame, sensor)
        return np.concatenate([return1[:, 0], return2[:, 0]])

    def _panoptic2d(self, seq: int, frame: int, sensor="IMG_FRONT"):
        record = self.records[seq]
        timestamp = self.timelines[seq]["LIDAR_TOP"][frame] - 0.05  # frame timestamp
        camera_name = self.img_sensors.index(sensor.replace("CAM_", "IMG_")) + 1

        camera_segmentation = pq.read_table(
            self.root / self.split / "camera_segmentation" / (record + ".parquet"),
            filters=[
                ("key.frame_timestamp_micros", "=", int(timestamp * 1e6)),
                ("key.camera_name", "=", camera_name),
            ],
        )

        divisor = camera_segmentation[
            "[CameraSegmentationLabelComponent].panoptic_label_divisor"
        ][0].as_py()
        instance2uid = camera_segmentation[
            "[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.global_instance_ids"
        ][0].as_py()
        instance2uid = np.array([-1] + instance2uid)

        ca = camera_segmentation[
            "[CameraSegmentationLabelComponent].panoptic_label"
        ].combine_chunks()[0]
        f = io.BytesIO(ca.as_buffer())
        data = np.asarray(Image.open(f))

        segmentation = data // divisor
        instances = instance2uid[data % divisor]

        return segmentation, instances

    def semantic2d(self, seq: int, frame: int, sensor="IMG_FRONT"):
        segmentation, _ = self._panoptic2d(seq, frame, sensor)
        return segmentation

    def instances2d(self, seq: int, frame: int, sensor="IMG_FRONT"):
        _, instances = self._panoptic2d(seq, frame, sensor)
        return instances

    def frames(self, seq=None, sensor=None):
        if sensor.startswith("SEG_"):
            seg_ts = self.timestamps(seq, sensor)
            sensor_ts = self.timestamps(seq, sensor.removeprefix("SEG_"))
            frames = np.searchsorted(sensor_ts, seg_ts, side="left")
            return [(seq, f) for f in frames.tolist()]

        else:
            return super(Waymo, self).frames(seq, sensor)
