import binascii
import json
import os
import struct
from collections import namedtuple
import dataclasses

import numpy as np

import PIL.Image

from ..geometry import (
    CameraProjection,
    Rotation,
    RigidTransform,
    Translation,
)
from .dataset import Box, Dataset

Scene = namedtuple(
    "Scene",
    [
        "data",
        "calibration",
        "ego_poses",
        "keyframes",
        "sample_tokens",
        "sample_timestamps",
        "boxes",
    ],
)
DataFile = namedtuple("DataFile", ["token", "sample_token", "timestamp", "filename"])
ImgDataFile = namedtuple(
    "ImgDataFile", ["token", "sample_token", "timestamp", "filename", "width", "height"]
)
RawAnnotation = namedtuple(
    "RawAnnotation",
    [
        "frame",
        "timestamp",
        "instance",
        "category",
        "visibility",
        "attributes",
        "position",
        "size",
        "rotation",
        "num_lidar_pts",
        "num_radar_pts",
    ],
)
Pose = namedtuple("Pose", ["timestamp", "rotation", "position"])


@dataclasses.dataclass(frozen=True)
class NuScenesBox(Box):
    visibility: int
    attributes: int


class NuScenes(Dataset):
    """`NuScenes <https://www.nuscenes.org>`_ dataset.

    .. note::

       Notable differences with original NuScenes data:

       * Size encoded as length, width, height instead of width, length, height.
       * Lidar pcl is rotated by 90° so x axis points forward.
       * Annotations are automatically interpolated between keyframes.

    The :meth:`keyframes` method returns the indices of the keyframes for each 
    sensor. Keyframes aggregate a sample for each sensor around a timestamps
    at around 2Hz.
    """

    _default_cam_sensor = "CAM_FRONT"
    _default_pcl_sensor = "LIDAR_TOP"
    _default_box_coords = "LIDAR_TOP"

    def __init__(
        self, root, subset="v1.0-mini", det_label_map=None, sem_label_map=None
    ):
        self.root_dir = root
        self.det_label_map = det_label_map
        self.sem_label_map = sem_label_map

        # load original data
        with open(os.path.join(root, subset, "attribute.json")) as f:
            attribute = json.load(f)
        with open(os.path.join(root, subset, "calibrated_sensor.json")) as f:
            calibrated_sensor = json.load(f)
        with open(os.path.join(root, subset, "category.json")) as f:
            category = json.load(f)
        with open(os.path.join(root, subset, "ego_pose.json")) as f:
            ego_pose = json.load(f)
        with open(os.path.join(root, subset, "instance.json")) as f:
            instance = json.load(f)
        with open(os.path.join(root, subset, "sample.json")) as f:
            sample = json.load(f)
        with open(os.path.join(root, subset, "sample_annotation.json")) as f:
            sample_annotation = json.load(f)
        with open(os.path.join(root, subset, "sample_data.json")) as f:
            sample_data = json.load(f)
        with open(os.path.join(root, subset, "scene.json")) as f:
            scene = json.load(f)
        with open(os.path.join(root, subset, "sensor.json")) as f:
            sensor = json.load(f)
        if os.path.exists(os.path.join(root, subset, "lidarseg.json")):
            with open(os.path.join(root, subset, "lidarseg.json")) as f:
                lidarseg = json.load(f)
        else:
            lidarseg = []
        if os.path.exists(os.path.join(root, subset, "panoptic.json")):
            with open(os.path.join(root, subset, "panoptic.json")) as f:
                panoptic = json.load(f)
        else:
            panoptic = []

        # convert into dictionaries
        attribute = {v["token"]: v for v in attribute}
        calibrated_sensor = {v["token"]: v for v in calibrated_sensor}
        category = {v["token"]: v for v in category}
        ego_pose = {v["token"]: v for v in ego_pose}
        instance = {v["token"]: v for v in instance}
        sample = {v["token"]: v for v in sample}
        sample_annotation = {v["token"]: v for v in sample_annotation}
        sample_data = {v["token"]: v for v in sample_data}
        scene = {v["token"]: v for v in scene}
        sensor = {v["token"]: v for v in sensor}
        lidarseg = {v["token"]: v for v in lidarseg}
        panoptic = {v["token"]: v for v in panoptic}

        # extract sensor names
        self.cam_sensors = []
        self.img_sensors = []
        self.pcl_sensors = []
        for s in sensor.values():
            if s["modality"] == "camera":
                self.cam_sensors.append(s["channel"])
                self.img_sensors.append(s["channel"].replace("CAM", "IMG"))
            elif s["modality"] == "lidar":
                self.pcl_sensors.append(s["channel"])

        # extract label names
        self.det_labels = [c["name"] for c in category.values()]
        self.sem_labels = self.det_labels

        # group samples by scenes
        scene_samples = {scene_t: {} for scene_t in scene.keys()}
        for sample_t, sample_v in sample.items():
            scene_samples[sample_v["scene_token"]][sample_t] = sample_v

        # assign incremental sample indices within scenes
        sample_indices = {}
        scene_sample_tokens = {scene_t: [] for scene_t in scene_samples.keys()}
        for scene_samples_t, scene_samples_v in scene_samples.items():
            sample_idx = 0
            sample_t = scene[scene_samples_t]["first_sample_token"]
            while sample_t != "":
                sample_indices[sample_t] = sample_idx
                scene_sample_tokens[scene_samples_t].append(sample_t)
                sample_t = scene_samples_v[sample_t]["next"]
                sample_idx += 1

        # group sensor data by scene
        scene_data = {
            scene_t: {sensor_v["channel"]: [] for sensor_v in sensor.values()}
            for scene_t in scene.keys()
        }

        for sample_data_v in sample_data.values():
            sensor_token = calibrated_sensor[sample_data_v["calibrated_sensor_token"]][
                "sensor_token"
            ]
            sensor_name = sensor[sensor_token]["channel"]
            scene_token = sample[sample_data_v["sample_token"]]["scene_token"]
            if "CAM" in sensor_name:
                scene_data[scene_token][sensor_name].append(
                    (
                        ImgDataFile(
                            token=sample_data_v["token"],
                            sample_token=sample_data_v["sample_token"],
                            timestamp=sample_data_v["timestamp"],
                            filename=sample_data_v["filename"],
                            width=sample_data_v["width"],
                            height=sample_data_v["height"],
                        ),
                        sample_data_v["is_key_frame"],
                        sample_data_v["sample_token"],
                    )
                )
            else:
                scene_data[scene_token][sensor_name].append(
                    (
                        DataFile(
                            token=sample_data_v["token"],
                            sample_token=sample_data_v["sample_token"],
                            timestamp=sample_data_v["timestamp"],
                            filename=sample_data_v["filename"],
                        ),
                        sample_data_v["is_key_frame"],
                        sample_data_v["sample_token"],
                    )
                )

        # add semantic segmentation data
        for d in scene_data.values():
            d["lidarseg"] = []
        for lidarseg_v in lidarseg.values():
            sample_data_v = sample_data[lidarseg_v["sample_data_token"]]
            scene_token = sample[sample_data_v["sample_token"]]["scene_token"]
            scene_data[scene_token]["lidarseg"].append(
                (
                    DataFile(
                        token=sample_data_v["token"],
                        sample_token=sample_data_v["sample_token"],
                        timestamp=sample_data_v["timestamp"],
                        filename=lidarseg_v["filename"],
                    ),
                    sample_data_v["is_key_frame"],
                    sample_data_v["sample_token"],
                )
            )

        # add panoptic segmentation data
        for d in scene_data.values():
            d["panoptic"] = []
        for panoptic_v in panoptic.values():
            sample_data_v = sample_data[panoptic_v["sample_data_token"]]
            scene_token = sample[sample_data_v["sample_token"]]["scene_token"]
            scene_data[scene_token]["panoptic"].append(
                (
                    DataFile(
                        token=sample_data_v["token"],
                        sample_token=sample_data_v["sample_token"],
                        timestamp=sample_data_v["timestamp"],
                        filename=panoptic_v["filename"],
                    ),
                    sample_data_v["is_key_frame"],
                    sample_data_v["sample_token"],
                )
            )

        # order scene data by timestamp
        for scene_data_v in scene_data.values():
            for k, v in scene_data_v.items():
                scene_data_v[k] = sorted(v, key=lambda s: s[0].timestamp)

        # index data keyframes
        scene_keyframes = {scene_t: {} for scene_t in scene.keys()}

        for scene_token, scene_data_v in scene_data.items():
            n_samples = len(scene_samples[scene_token])
            for sensor_name, sample_data_v in scene_data_v.items():
                kf = np.full([n_samples], -1, dtype=np.int64)
                for i, (_, is_key_frame, sample_token) in enumerate(sample_data_v):
                    if is_key_frame:  # is_keyframe
                        kf[sample_indices[sample_token]] = i

                scene_keyframes[scene_token][sensor_name] = kf

        # drop extra scene_data items
        scene_data = {
            scene_token: {
                modality_name: [d for d, _, _ in modality_data]
                for modality_name, modality_data in scene_data_v.items()
            }
            for scene_token, scene_data_v in scene_data.items()
        }

        # group sensor calibration by scene
        scene_calibration = {scene_t: {} for scene_t in scene.keys()}
        for scene_t, scene_data_v in scene_data.items():
            for sensor_name, sample_data_v in scene_data_v.items():
                if len(sample_data_v) > 0:
                    calibrated_sensor_token = sample_data[sample_data_v[0].token][
                        "calibrated_sensor_token"
                    ]
                    scene_calibration[scene_t][sensor_name] = calibrated_sensor[
                        calibrated_sensor_token
                    ]

        # duplicate 'CAM_*' with 'IMG_*' sensor field for convenience
        for sd in scene_data.values():
            for sensor_name, data in list(sd.items()):
                if sensor_name.startswith("CAM_"):
                    sd["IMG_" + sensor_name[4:]] = data

        for sd in scene_calibration.values():
            for sensor_name, data in list(sd.items()):
                if sensor_name.startswith("CAM_"):
                    sd["IMG_" + sensor_name[4:]] = data

        for sd in scene_keyframes.values():
            for sensor_name, data in list(sd.items()):
                if sensor_name.startswith("CAM_"):
                    sd["IMG_" + sensor_name[4:]] = data

        # group sample timestamps by scene
        scene_sample_ts = {scene_token: [] for scene_token in scene.keys()}
        for scene_t, scene_samples_v in scene_samples.items():
            scene_sample_ts[scene_t] = np.sort(
                [sample["timestamp"] for sample in scene_samples_v.values()]
            )

        # group ego pose by scene, indexed by timestamp
        scene_ego_poses = {scene_token: {} for scene_token in scene.keys()}
        for sample_data_v in sample_data.values():
            scene_token = sample[sample_data_v["sample_token"]]["scene_token"]
            ego_pose_v = ego_pose[sample_data_v["ego_pose_token"]]
            scene_ego_poses[scene_token][sample_data_v["timestamp"]] = Pose(
                timestamp=ego_pose_v["timestamp"],
                rotation=ego_pose_v["rotation"],
                position=ego_pose_v["translation"],
            )

        # add object annotation data
        scene_boxes = {scene_token: [] for scene_token in scene.keys()}

        for sample_annotation_v in sample_annotation.values():
            scene_token = sample[sample_annotation_v["sample_token"]]["scene_token"]
            category_token = instance[sample_annotation_v["instance_token"]][
                "category_token"
            ]
            annotation_category = category[category_token]["name"]
            frame = sample_indices[sample_annotation_v["sample_token"]]
            timestamp = sample[sample_annotation_v["sample_token"]]["timestamp"]
            annotation_attributes = [
                attribute[t]["name"] for t in sample_annotation_v["attribute_tokens"]
            ]
            width, length, height = sample_annotation_v["size"]
            scene_boxes[scene_token].append(
                RawAnnotation(
                    frame=frame,
                    timestamp=timestamp,
                    instance=sample_annotation_v["instance_token"],
                    category=annotation_category,
                    visibility=sample_annotation_v["visibility_token"],
                    attributes=annotation_attributes,
                    position=sample_annotation_v["translation"],
                    size=(length, width, height),
                    rotation=sample_annotation_v["rotation"],
                    num_lidar_pts=sample_annotation_v["num_lidar_pts"],
                    num_radar_pts=sample_annotation_v["num_radar_pts"],
                )
            )
            assert len(scene_boxes[scene_token][-1].rotation) == 4

        # sort by frame
        scene_boxes = {
            t: sorted(o, key=lambda a: (a.frame, a.instance))
            for t, o in scene_boxes.items()
        }

        if "index" in next(iter(category.values())):
            categories = [None] * (max(c["index"] for c in category.values()) + 1)
            for c in category.values():
                categories[c["index"]] = c["name"]
        else:
            categories = [c["name"] for c in category.values()]

        # rotate lidars to make x point forward
        for scene_calibs in scene_calibration.values():
            for sensor, sensor_calibs in scene_calibs.items():
                if sensor in self.pcl_sensors:
                    sensor_calibs["rotation"] = (
                        Rotation(sensor_calibs["rotation"])
                        @ Rotation.from_euler("Z", np.pi / 2)
                    ).as_quat()

        # set attributes
        self.categories = categories

        self.scenes = [
            Scene(
                scene_data[t],
                scene_calibration[t],
                scene_ego_poses[t],
                scene_keyframes[t],
                scene_sample_tokens[t],
                scene_sample_ts[t],
                scene_boxes[t],
            )
            for t in sorted(scene.keys())
        ]

        self.annotations_cache = {}

    def _calibration(self, seq, src_sensor, dst_sensor):
        if src_sensor == dst_sensor:
            return Translation([0.0, 0.0, 0.0])

        if dst_sensor in self.img_sensors:
            intrinsic = self.scenes[seq].calibration[dst_sensor]["camera_intrinsic"]
            intrinsic = (
                intrinsic[0][0],
                intrinsic[1][1],
                intrinsic[0][2],
                intrinsic[1][2],
            )
            cam2img = CameraProjection("pinhole", intrinsic)

            cam = self.cam_sensors[self.img_sensors.index(dst_sensor)]
            src2cam = self._calibration(seq, src_sensor, cam)

            return cam2img @ src2cam

        if src_sensor not in (self.cam_sensors + self.pcl_sensors):
            raise ValueError()

        if src_sensor == "ego":
            src_calib = Translation([0, 0, 0])
        else:
            src_calib = self.scenes[seq].calibration[src_sensor]
            src_calib = RigidTransform(src_calib["rotation"], src_calib["translation"])

        if dst_sensor == "ego":
            dst_calib = Translation([0, 0, 0])
        else:
            dst_calib = self.scenes[seq].calibration[dst_sensor]
            dst_calib = RigidTransform(dst_calib["rotation"], dst_calib["translation"])

        return dst_calib.inv() @ src_calib

    def _poses(self, seq, sensor):
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        if sensor == "boxes":
            num_frames = len(self.scenes[seq].keyframes["LIDAR_TOP"])
            return RigidTransform.from_matrix(np.tile(np.eye(4), (num_frames, 1, 1)))

        ego_poses = [
            self.scenes[seq].ego_poses[t] for t in self.timestamps(seq, sensor)
        ]
        ego_poses = RigidTransform(
            [p.rotation for p in ego_poses], [p.position for p in ego_poses]
        )

        return ego_poses @ self._calibration(seq, sensor, "ego")

    def _points(self, seq, frame, sensor):
        filename = os.path.join(
            self.root_dir, self.scenes[seq].data[sensor][frame].filename
        )
        pcl = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)
        pcl[:, 0], pcl[:, 1] = pcl[:, 1], -pcl[:, 0]
        return pcl

    def _boxes(self, seq):
        out = []

        for b in self.scenes[seq].boxes:
            obj2coords = RigidTransform(b.rotation, b.position)
            out.append(
                NuScenesBox(
                    frame=b.frame,
                    uid=struct.unpack("Q", binascii.unhexlify(b.instance)[:8])[0],
                    center=obj2coords.apply([0, 0, 0]),
                    size=b.size,
                    heading=obj2coords.rotation.as_euler("ZYX")[0],
                    transform=obj2coords,
                    label=b.category,
                    visibility=b.visibility,
                    attributes=b.attributes,
                    # num_lidar_pts=b.num_lidar_pts,
                    # num_radar_pts=b.num_radar_pts,
                )
            )

        return out

    def sequences(self):
        return list(range(len(self.scenes)))

    def timestamps(self, seq, sensor):
        if sensor == "boxes":
            return self.scenes[seq].sample_timestamps
        else:
            return np.array(
                [sample.timestamp for sample in self.scenes[seq].data[sensor]]
            )

    def image(self, seq, frame, sensor="CAM_FRONT", keyframe=True):
        return PIL.Image.open(
            os.path.join(self.root_dir, self.scenes[seq].data[sensor][frame].filename)
        )

    def rectangles(self, seq: int, frame: int):
        raise NotImplementedError

    def semantic(self, seq, frame, sensor="LIDAR_TOP"):
        seg_frame = np.searchsorted(self.scenes[seq].keyframes[sensor], frame, "left")
        if self.scenes[seq].keyframes[sensor][seg_frame] != frame:
            raise ValueError(f"frame {frame} is not a keyframe")

        filename = self.scenes[seq].data["lidarseg"][seg_frame].filename
        semantic = np.fromfile(os.path.join(self.root_dir, filename), dtype=np.uint8)

        if self.sem_label_map is not None:
            semantic = self.sem_label_map[semantic]

        return semantic

    def instances(self, seq, frame, sensor="LIDAR_TOP"):
        seg_frame = np.searchsorted(self.scenes[seq].keyframes[sensor], frame, "left")
        if self.scenes[seq].keyframes[sensor][seg_frame] != frame:
            raise ValueError(f"frame {frame} is not a keyframe")

        filename = self.scenes[seq].data["panoptic"][seg_frame].filename
        panoptic = np.load(os.path.join(self.root_dir, filename))["data"] % 1000

        return panoptic

    def keyframes(self, seq, sensor):
        return self.scenes[seq].keyframes[sensor]
