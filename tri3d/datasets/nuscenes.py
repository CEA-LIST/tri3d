import dataclasses
import json
import os
import pathlib
import uuid

import numpy as np
import PIL.Image

from ..geometry import (
    CameraProjection,
    RigidTransform,
    Rotation,
    Transformation,
    Translation,
)
from ..misc import memoize_method
from .dataset import Box, Dataset


@dataclasses.dataclass(frozen=True)
class NuScenesBox(Box):
    attributes: list[str]
    visibility: str | None
    num_lidar_pts: int
    num_radar_pts: int


@dataclasses.dataclass()
class Scene:
    data: dict[str, list[str]]
    calibration: dict[str, np.ndarray]
    ego_poses: dict[str, RigidTransform]
    keyframes: dict[str, np.ndarray]
    sample_tokens: list[str]
    timestamps: dict[str, np.ndarray]
    boxes: list[NuScenesBox]


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

    .. run::

        import matplotlib.pyplot as plt
        import numpy as np
        from tri3d.datasets import NuScenes

        plt.switch_backend("Agg")

        dataset = NuScenes("datasets/nuscenes", "v1.0-mini")
        name = "tri3d.datasets.NuScenes"
        camera, imgcoords, lidar = "CAM_FRONT", "IMG_FRONT", "LIDAR_TOP"
        seq, frame, cam_frame = 5, 130, 77
    """

    scenes: list[Scene]
    _default_cam_sensor = "CAM_FRONT"
    _default_pcl_sensor = "LIDAR_TOP"
    _default_box_coords = "LIDAR_TOP"

    def __init__(
        self,
        root,
        subset="v1.0-mini",
        det_label_map=None,
        sem_label_map=None,
    ):
        root = pathlib.Path(root)
        self.root = root
        self.subset = subset
        self.det_label_map = det_label_map
        self.sem_label_map = sem_label_map

        # load original data
        with open(root / subset / "attribute.json", "rb") as f:
            attribute = json.load(f)
        with open(root / subset / "calibrated_sensor.json", "rb") as f:
            calibrated_sensor = json.load(f)
        with open(root / subset / "category.json", "rb") as f:
            category = json.load(f)
        with open(root / subset / "ego_pose.json", "rb") as f:
            ego_pose = json.load(f)
        with open(root / subset / "instance.json", "rb") as f:
            instance = json.load(f)
        with open(root / subset / "sample.json", "rb") as f:
            sample = json.load(f)
        with open(root / subset / "sample_annotation.json", "rb") as f:
            sample_annotation = json.load(f)
        with open(root / subset / "sample_data.json", "rb") as f:
            sample_data = json.load(f)
        with open(root / subset / "scene.json", "rb") as f:
            scene = json.load(f)
        with open(root / subset / "sensor.json", "rb") as f:
            sensor = json.load(f)
        with open(root / subset / "visibility.json", "rb") as f:
            visibility = json.load(f)
        if os.path.exists(root / subset / "lidarseg.json"):
            with open(root / subset / "lidarseg.json", "rb") as f:
                lidarseg = json.load(f)
        else:
            lidarseg = []
        if os.path.exists(root / subset / "panoptic.json"):
            with open(root / subset / "panoptic.json", "rb") as f:
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
        sample_data = {v["token"]: v for v in sample_data}
        scene = {v["token"]: v for v in scene}
        sensor = {v["token"]: v for v in sensor}

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

        # categories
        if "index" in next(iter(category.values())):
            self.categories = [None] * (max(c["index"] for c in category.values()) + 1)
            for c in category.values():
                self.categories[c["index"]] = c["name"]
        else:
            self.categories = [c["name"] for c in category.values()]

        # merge channel name into sample_data
        for sample_data_v in sample_data.values():
            calibrated_sensor_t = sample_data_v["calibrated_sensor_token"]
            calibrated_sensor_v = calibrated_sensor[calibrated_sensor_t]
            sensor_v = sensor[calibrated_sensor_v["sensor_token"]]
            sample_data_v["channel"] = sensor_v["channel"]

        # merge ego pose into sample_data
        for sample_data_v in sample_data.values():
            sample_data_v["ego_pose"] = ego_pose[sample_data_v["ego_pose_token"]]

        # merge lidarseg and panoptic into sample_data
        for lidarseg_v in lidarseg:
            sample_data_t = lidarseg_v["sample_data_token"]
            sample_data[sample_data_t]["lidarseg_filename"] = lidarseg_v["filename"]

        for panoptic_v in panoptic:
            sample_data_t = panoptic_v["sample_data_token"]
            sample_data[sample_data_t]["panoptic_filename"] = panoptic_v["filename"]

        # group sample tokens by scenes
        scene_samples = {scene_t: [] for scene_t in scene.keys()}
        sample_idx = {}
        for scene_t, scene_v in scene.items():
            sample_t = scene_v["first_sample_token"]
            i = 0
            while sample_t != "":
                scene_samples[scene_t].append(sample_t)
                sample_idx[sample_t] = i
                sample_t = sample[sample_t]["next"]
                i += 1

        # Group sample data by scene
        scene_data = {scene_t: {} for scene_t in scene.keys()}

        for sample_data_v in sample_data.values():
            sample_t = sample_data_v["sample_token"]
            sample_v = sample[sample_t]
            scene_t = sample_v["scene_token"]
            channel = sample_data_v["channel"]

            if channel not in scene_data[scene_t]:
                scene_data[scene_t][channel] = []

            scene_data[scene_t][channel].append(sample_data_v)

        # sort sample data by timestamps
        for scene_data_v in scene_data.values():
            for channel in list(scene_data_v.keys()):
                scene_data_v[channel] = sorted(
                    scene_data_v[channel], key=lambda d: d["timestamp"]
                )

        # Group box annotations by scene
        scene_annotations = {scene_t: [] for scene_t in scene.keys()}
        for sample_annotation_v in sample_annotation:
            sample_t = sample_annotation_v["sample_token"]
            scene_t = sample[sample_t]["scene_token"]
            instance_t = sample_annotation_v["instance_token"]
            center = sample_annotation_v["translation"]
            rotation = sample_annotation_v["rotation"]
            transform = RigidTransform(rotation, center)
            label = category[instance[instance_t]["category_token"]]["name"]
            annotation_attributes = [
                attribute[t]["name"] for t in sample_annotation_v["attribute_tokens"]
            ]
            if "visibility_t" in sample_annotation_v is None:
                box_visibility = visibility[sample_annotation_v["visibility_t"]][
                    "level"
                ]
            else:
                box_visibility = None
            width, length, height = sample_annotation_v["size"]

            scene_annotations[scene_t].append(
                NuScenesBox(
                    frame=sample_idx[sample_t],
                    uid=instance_t,
                    center=center,
                    size=np.array([length, width, height]),
                    heading=transform.rotation.as_euler("ZYX")[0],
                    transform=transform,
                    label=label,
                    attributes=annotation_attributes,
                    visibility=box_visibility,
                    num_lidar_pts=sample_annotation_v["num_lidar_pts"],
                    num_radar_pts=sample_annotation_v["num_radar_pts"],
                )
            )

        scene_annotations = {  # sort by frame and uid
            scene_t: sorted(boxes, key=lambda b: (b.frame, b.uid))
            for scene_t, boxes in scene_annotations.items()
        }

        self.scenes = []
        for scene_t in sorted(scene.keys()):
            # calib
            scene_calibration = {}
            scene_ego_poses = {}
            scene_keyframes = {}
            scene_timestamps = {}
            scene_data_ = {}

            for channel, channel_data_v in scene_data[scene_t].items():
                if channel not in self.pcl_sensors and channel not in self.cam_sensors:
                    continue

                calib = calibrated_sensor[channel_data_v[0]["calibrated_sensor_token"]]

                if channel in self.pcl_sensors:
                    calib["rotation"] = (
                        Rotation(calib["rotation"])
                        @ Rotation.from_euler("Z", np.pi / 2)
                    ).as_quat()  # type: ignore

                scene_calibration[channel] = calib

                scene_ego_poses[channel] = RigidTransform(
                    [d["ego_pose"]["rotation"] for d in channel_data_v],
                    [d["ego_pose"]["translation"] for d in channel_data_v],
                )

                scene_keyframes[channel] = np.array(
                    [i for i, d in enumerate(channel_data_v) if d["is_key_frame"]]
                )

                scene_timestamps[channel] = np.array(
                    [d["timestamp"] for d in channel_data_v]
                )

                scene_data_[channel] = [d["filename"] for d in channel_data_v]

                if channel in self.pcl_sensors:
                    scene_data_["lidarseg"] = [
                        d.get("lidarseg_filename", None) for d in channel_data_v
                    ]
                    scene_data_["panoptic"] = [
                        d.get("panoptic_filename", None) for d in channel_data_v
                    ]

            scene_timestamps["sample"] = np.array(
                [sample[sample_t]["timestamp"] for sample_t in scene_samples[scene_t]]
            )

            self.scenes.append(
                Scene(
                    data=scene_data_,
                    calibration=scene_calibration,
                    ego_poses=scene_ego_poses,
                    keyframes=scene_keyframes,
                    sample_tokens=scene_samples[scene_t],
                    timestamps=scene_timestamps,
                    boxes=scene_annotations[scene_t],
                )
            )

        self.annotations_cache = {}

    @memoize_method(maxsize=100)
    def _calibration(self, seq, src_sensor, dst_sensor) -> Transformation:
        if src_sensor == dst_sensor:
            return Translation([0.0, 0.0, 0.0])

        if dst_sensor in self.img_sensors:
            cam_sensor = self.cam_sensors[self.img_sensors.index(dst_sensor)]
            intrinsic = self.scenes[seq].calibration[cam_sensor]["camera_intrinsic"]
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

        if src_sensor == "ego":
            return self._calibration(seq, dst_sensor, src_sensor).inv()

        if src_sensor not in self.cam_sensors and src_sensor not in self.pcl_sensors:
            raise ValueError()

        src_calib = self.scenes[seq].calibration[src_sensor]
        src_calib = RigidTransform(src_calib["rotation"], src_calib["translation"])

        if dst_sensor == "ego":
            dst_calib = Translation([0, 0, 0])
        else:
            dst_calib = self.scenes[seq].calibration[dst_sensor]
            dst_calib = RigidTransform(dst_calib["rotation"], dst_calib["translation"])

        return dst_calib.inv() @ src_calib

    @memoize_method(maxsize=10)
    def _poses(self, seq, sensor):
        if sensor == "boxes":
            N = len(self.scenes[seq].sample_tokens)
            return RigidTransform.from_matrix(np.tile(np.eye(4), (N, 1, 1)))

        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        sensor2ego: RigidTransform = self._calibration(seq, sensor, "ego")  # type: ignore
        ego2world = self.scenes[seq].ego_poses[sensor]
        return ego2world @ sensor2ego

    def _points(self, seq, frame, sensor):
        filename = self.root / self.scenes[seq].data[sensor][frame]
        pcl = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)

        # Rotate pcl to make x point forward
        pcl[:, 0], pcl[:, 1] = pcl[:, 1], -pcl[:, 0]
        return pcl

    def _boxes(self, seq):
        return self.scenes[seq].boxes

    def sequences(self):
        return list(range(len(self.scenes)))

    def timestamps(self, seq, sensor):
        if sensor == "boxes":
            sensor = "sample"
        elif sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        return self.scenes[seq].timestamps[sensor]

    def image(self, seq, frame, sensor="CAM_FRONT"):
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        filename = self.root / self.scenes[seq].data[sensor][frame]
        return PIL.Image.open(filename)

    def rectangles(self, seq: int, frame: int):
        raise NotImplementedError

    def semantic(self, seq, frame, sensor="LIDAR_TOP"):
        filename = self.scenes[seq].data["lidarseg"][frame]
        if filename is None:
            raise ValueError(f"frame {frame} has no segmentation")

        semantic = np.fromfile(self.root / filename, dtype=np.uint8)

        if self.sem_label_map is not None:
            semantic = self.sem_label_map[semantic]

        return semantic

    def instances(self, seq, frame, sensor="LIDAR_TOP"):
        filename = self.scenes[seq].data["panoptic"][frame]
        if filename is None:
            raise ValueError(f"frame {frame} has no panoptic")

        panoptic = np.load(self.root / filename)["data"] % 1000

        return panoptic

    def keyframes(self, seq, sensor):
        """The indices of the keyframes within the frames of each sensor."""
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        return self.scenes[seq].keyframes[sensor]

    def sample_tokens(self, seq):
        """The sample token for each keyframe."""
        return self.scenes[seq].sample_tokens


def dump_nuscene_boxes(
    dataset: NuScenes,
    seq_indices: list[int],
    sensor: str,
    boxes: list[NuScenesBox],
    keyframes: bool = False,
):
    """Convert boxes to the NuScene format (`sample_annotation.json`).

    Args:
        dataset: the nuscene dataset
        seq_indices: *for each box* the index of its sequence
        sensor: the coordinate system in which box poses are expressed
        boxes: the boxes to export
        keyframes:
            Whether boxes[*].frame indexes the keyframes or the sensor timeline.

    Note: Boxes on frames other than keyframes are skipped.

    Warning:
        New annotation token are generated so the `{first,last}_annotation_token`
        foreign keys in `instance.json` won't work with the exported boxes.
        These fields are not used in tri3D.
    """
    out = []
    prev = None
    token = uuid.uuid4().hex
    next = uuid.uuid4().hex

    assert len(seq_indices) == len(boxes)

    for i, s, b in zip(range(len(boxes)), seq_indices, boxes):
        if keyframes:
            frame = dataset.keyframes(s, sensor)[b.frame]
            sample_token = dataset.sample_tokens(s)[b.frame]
        else:
            frame = b.frame
            keyframe_idx = np.searchsorted(dataset.keyframes(s, sensor), b.frame)
            if dataset.keyframes(s, sensor)[keyframe_idx] != b.frame:
                continue
            sample_token = dataset.sample_tokens(s)[keyframe_idx]

        ego_pose = dataset.poses(s, sensor)[frame]
        box_pose: RigidTransform = ego_pose @ b.transform  # type: ignore
        length, width, height = b.size

        out.append(
            {
                "token": token,
                "sample_token": sample_token,
                "instance_token": b.uid,
                "attribute_token": [],  # TODO
                "visibility_token": None,  # TODO
                "translation": box_pose.translation.vec,
                "size": [width, length, height],
                "rotation": box_pose.rotation.quat,
                "num_lidar_pts": b.num_lidar_pts,
                "num_radar_pts": b.num_radar_pts,
                "prev": prev,
                "next": next,
            }
        )

        prev = token
        token = next
        next = uuid.uuid4().hex

    out[-1]["next"] = None

    return out
