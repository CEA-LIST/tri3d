import dataclasses
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from numbers import Integral

import numpy as np
from PIL.Image import Image

from ..geometry import Transformation, RigidTransform
from .. import misc


@dataclasses.dataclass(frozen=True)
class Box:
    """Base class for 3D object bounding box."""

    frame: int
    "The index of the frame in the requested timeline"

    uid: int
    "A unique instance id for tracking"

    center: np.ndarray
    "xyz coordinates of the center of the bounding box"

    size: np.ndarray
    "length, width, height"

    heading: float
    "Rotation along z axis in radian"

    transform: Transformation
    "Geometric transformation from object local to requested coordinate system"

    label: str
    "Annotated object class"

    def __iter__(self):
        yield from dataclasses.astuple(self)


class AbstractDataset(ABC):
    """Abstract class for driving datasets.

    Each implementation should document availables sensors and the label
    by which they are refered to. They are used to query data and to specify
    the timeline and spatial coordinate systems.
    """

    cam_sensors: List[str]
    """Camera names."""

    img_sensors: List[str]
    """Camera names (image plane coordinate)."""

    pcl_sensors: List[str]
    """Point cloud sensor names."""

    det_labels: List[str]
    """Detection labels."""

    sem_labels: List[str]
    """Segmentation labels."""

    @abstractmethod
    def sequences(self) -> List[int]:
        """Return the list of sequences/recordings indices (0..num_sequences)."""
        raise NotImplementedError

    @abstractmethod
    def frames(
        self, seq: int | None = None, sensor: str | None = None
    ) -> List[Tuple[int, int]]:
        """Return the frames in the dataset or a particular sequence.

        :param seq:
            Sequence index.
        :param seq:
            Sequence index.
        :return:
            A list of (sequence, frame) index tuples sorted by sequence and
            frame.
        """
        raise NotImplementedError

    @abstractmethod
    def timestamps(self, seq: int, sensor: str | None = None) -> np.ndarray:
        """Return the frame timestamps for a given sensor .

        :param seq:
            Sequence index.
        :param sensor:
            Sensor name.
        :return:
            An array of timestamps.

        .. note:: frames are guarenteed to be sorted.
        """
        raise NotImplementedError

    @abstractmethod
    def poses(
        self, seq: int, sensor: str, timeline: str | None = None
    ) -> RigidTransform:
        """Return all sensor to world transforms for a sensor.

        *World* references an arbitrary coordinate system for a sequence, not
        all datasets provide an actual global coordinate system.

        :param seq:
            Sequence index.
        :param sensor:
            Sensor name.
        :param timeline:
            When specified, the sensor poses will be interpolated to the
            timestamps of that timeline if necessary.
        :return:
            Sensor poses as a batched transform.
        """
        raise NotImplementedError

    @abstractmethod
    def alignment(
        self,
        seq: int,
        frame: Union[int, Tuple[int, int]],
        coords: Union[str, Tuple[str, str]],
    ):
        """Return the transformation from one coordinate system and timestamp to another.

        :param seq:
            Sequence index.
        :param frame:
            Either a single frame or a (src, dst) tuple. The frame is respective
            to the sensor timeline as specified by `coords`.
        :param coords:
            Either a single sensor/coordinate system or a (src, dst) tuple.
            The transformation also accounts for mismatches in sensor timelines
            and movement of the ego-car.
        :return:
            A transformation that projects points from one coordinate
            system at one frame to another.
        """
        raise NotImplementedError

    @abstractmethod
    def image(self, seq: int, frame: int, sensor: str | None = None) -> Image:
        """Return image from given camera at given frame.

        A default sensor (for instance a front facing camera) should be
        provided for convenience.

        :param seq:
            Sequence index.
        :param frame:
            Frame index.
        :param sensor:
            The image sensor to use.
        """
        raise NotImplementedError

    @abstractmethod
    def points(
        self, seq: int, frame: int, sensor: str | None = None, coords: str | None = None
    ) -> np.ndarray:
        """Return an array of 3D point coordinates from lidars.

        The first three columns contains xyz coordinates, additional columns
        are dataset-specific.

        For convenience, the point cloud can be returned in the coordinate
        system of another sensor. In that case, `frame` is understood as the
        frame for that sensor and the point cloud which has the nearest
        timestamp is retrieved and aligned.

        :param seq:
            Sequence index.
        :param frame:
            Frame index.
        :param sensor:
            The 3D sensor (generally a LiDAR) to use.
        :param coords:
            The coordinate system and timeline to use. Defaults to the sensor.
        :return:
            A NxD array where the first 3 columns are X, Y, Z point coordinates
            and the remaining ones are dataset-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def boxes(self, seq: int, frame: int, coords: str | None = None) -> List[Box]:
        """Return the 3D box annotations.

        This function will interpolate and transform annotations if necessary in order
        to match the requested coordinate system and timeline.

        :param seq:
            Sequence index.
        :param frame:
            Frame index.
        :param coords:
            The coordinate system and timeline to use.
        :return:
            A list of box annotations.
        """
        raise NotImplementedError

    @abstractmethod
    def rectangles(self, seq: int, frame: int):
        """Return a list of 2D rectangle annotations.

        .. note:: The default coordinate system should be documented.

        :param seq:
            Sequence index.
        :param frame:
            Frame index or `None` to request annotations for the whole sequence
        :return:
            A list of 2D annotations.
        """
        raise NotImplementedError

    @abstractmethod
    def semantic(self, seq: int, frame: int, sensor: str):
        """Return pointwise class annotations.

        :param seq:
            Sequence index.
        :param frame:
            Frame index.
        :param sensor:
            The camera sensor for which annotations are returned.
        :return:
            array of pointwise class label
        """
        raise NotImplementedError

    @abstractmethod
    def semantic2d(self, seq: int, frame: int):
        """Return pixelwise class annotations.

        :param seq:
            Sequence index.
        :param frame:
            Frame index.
        :return:
            array of pointwise class label
        """
        raise NotImplementedError

    @abstractmethod
    def instances(self, seq: int, frame: int):
        """Return pointwise instance ids.

        :param seq:
            Sequence index.
        :param frame:
            Frame index.
        :return:
            array of pointwise instance label
        """
        raise NotImplementedError

    @abstractmethod
    def instances2d(self, seq: int, frame: int):
        """Return pixelwise instance annotations.

        Background label pixels will contain -1. Other instance ids will follow
        dataset-specific rules. 

        :param seq:
            Sequence index.
        :param frame:
            Frame index.
        :return:
            array of pointwise instance label
        """
        raise NotImplementedError


class Dataset(AbstractDataset):
    # Base dataset with generic implementation of common methods
    # You should probably add cache (misc.memoize_method) over frequently called
    # methods.

    cam_sensors: List[str]
    img_sensors: List[str]
    pcl_sensors: List[str]
    det_labels: List[str]
    sem_labels: List[str]

    _nn_interp_thres = 0.05
    _default_cam_sensor: str
    _default_pcl_sensor: str
    _default_box_coords: str

    def _calibration(self, seq: int, src_sensor: str, dst_sensor: str):
        raise NotImplementedError

    def _poses(self, seq: int, sensor: str) -> RigidTransform:
        raise NotImplementedError

    def _points(self, seq: int, frame: int, sensor: str) -> np.ndarray:
        raise NotImplementedError

    def _boxes(self, seq: int) -> List[Box]:
        raise NotImplementedError

    def sequences(self):
        raise NotImplementedError

    def frames(self, seq: int | None = None, sensor: str | None = None):
        if seq is None:
            out = []
            for seq in range(len(self.sequences())):
                out += self.frames(seq, sensor)

            return out

        else:
            return [(seq, i) for i in range(len(self.timestamps(seq, sensor)))]

    def timestamps(self, seq: int, sensor: str):
        # Must support all sensors and a virtual one for "boxes"
        raise NotImplementedError

    def poses(self, seq: int, sensor: str, timeline: str | None = None):
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        if timeline is None:
            timeline = sensor

        if timeline == sensor:  # simple case
            try:
                return self._poses(seq, sensor)
            except ValueError:
                pass

        else:  # use timeline sensor poses and calibration when available
            try:
                tim_sensor_poses = self._calibration(seq, sensor, timeline)
                return self._poses(seq, timeline) @ tim_sensor_poses
            except ValueError:
                pass

        # interpolate sensor poses to timeline
        try:  # use sensor poses when available
            poses = self._poses(seq, sensor)
            poses_t = self.timestamps(seq, sensor)
        except ValueError:  # use imu and imu->sensor calib
            poses = self._poses(seq, "ego") @ self._calibration(seq, sensor, "ego")
            poses_t = self.timestamps(seq, "ego")

        dst_t = self.timestamps(seq, timeline)

        i1, i2 = misc.lr_bisect(poses_t, dst_t)
        t1 = poses_t[i1]
        t2 = poses_t[i2]
        alpha = (dst_t - t1) / (t2 - t1).clip(min=1e-6)

        return RigidTransform.interpolate(poses[i1], poses[i2], alpha)

    def alignment(
        self, seq: int, frame: int | tuple[int, int], coords: str | tuple[str, str]
    ):
        # normalize arguments
        if isinstance(frame, Integral):
            src_frame, dst_frame = frame, frame
        else:
            src_frame, dst_frame = frame

        if isinstance(coords, str):
            src_coords, dst_coords = coords, coords
        else:
            src_coords, dst_coords = coords

        # split image plane projection into src -> cam -> img
        if dst_coords in self.img_sensors:
            cam_coords = self.cam_sensors[self.img_sensors.index(dst_coords)]
            src2cam = self.alignment(
                seq, (src_frame, dst_frame), (src_coords, cam_coords)
            )
            cam2img = self._calibration(seq, cam_coords, dst_coords)
            return cam2img @ src2cam

        return (
            self.poses(seq, dst_coords)[dst_frame].inv()
            @ self.poses(seq, src_coords)[src_frame]
        )

    def image(self, seq: int, frame: int, sensor: str):
        raise NotImplementedError

    def points(
        self, seq: int, frame: int, sensor: str | None = None, coords: str | None = None
    ):
        if sensor is None:
            sensor = self._default_pcl_sensor

        if coords is None:
            coords = sensor

        if coords == sensor:
            return self._points(seq, frame, sensor)

        else:
            lidar_frame = misc.nearest_sorted(
                self.timestamps(seq, sensor), self.timestamps(seq, coords)[frame]
            )
            transform = self.alignment(seq, (lidar_frame, frame), (sensor, coords))
            points = self.points(seq, lidar_frame, sensor=sensor)
            points[:, :3] = transform.apply(points[:, :3])
            return points

    def boxes(self, seq: int, frame: int, coords: str | None = None):
        if coords is None:
            coords = self._default_box_coords

        # decompose ann coords -> cam -> img
        if coords in self.img_sensors:
            cam = self.cam_sensors[self.img_sensors.index(coords)]
            boxes = self.boxes(seq, frame, cam)
            cam2img = self._calibration(seq, cam, coords)
            out = []
            for b in boxes:
                transform = cam2img @ b.transform
                out.append(
                    dataclasses.replace(
                        b, center=transform.apply([0, 0, 0]), transform=transform
                    )
                )

            return out

        boxes = self._boxes(seq)

        # find nearest sensor frame
        sensor_ts = self.timestamps(seq, coords)[frame]
        boxes_timestamps = self.timestamps(seq, "boxes")
        ann_frame = misc.nearest_sorted(boxes_timestamps, sensor_ts)

        # use nearest frame when box and sensor timestamps are close
        if abs(sensor_ts - boxes_timestamps[ann_frame]) < self._nn_interp_thres:
            boxes = [b for b in boxes if b.frame == ann_frame]

            ann2coords = self.alignment(seq, (ann_frame, frame), ("boxes", coords))

            out = []
            for b in boxes:
                obj2coords = ann2coords @ b.transform
                if coords in self.cam_sensors:
                    heading = -obj2coords.rotation.as_euler("YZX")[0] - np.pi / 2
                else:
                    heading = obj2coords.rotation.as_euler("ZYX")[0]
                out.append(
                    dataclasses.replace(
                        b,
                        center=obj2coords.apply([0, 0, 0]),
                        heading=heading,
                        transform=obj2coords,
                    )
                )

            return out

        # filter on boxes visible at i1 or i2
        i1, i2 = misc.lr_bisect(boxes_timestamps, sensor_ts)

        boxes = [b for b in boxes if b.frame == i1 or b.frame == i2]
        uids = {b.uid for b in boxes}
        tracks = [[b for b in boxes if b.uid == u] for u in uids]
        tracks = [t if len(t) == 2 else t * 2 for t in tracks]

        # interpolate
        t1 = boxes_timestamps[i1]
        t2 = boxes_timestamps[i2]
        w = (sensor_ts - t1) / max(t2 - t1, 1e-6)  # TODO

        ann2coords = RigidTransform.interpolate(
            self.alignment(seq, (i1, frame), ("boxes", coords)),
            self.alignment(seq, (i2, frame), ("boxes", coords)),
            w,
        )

        out = []
        for b1, b2 in tracks:
            obj2coords = ann2coords @ RigidTransform.interpolate(
                b1.transform, b2.transform, w
            )
            out.append(
                dataclasses.replace(
                    b1,
                    center=obj2coords.apply([0, 0, 0]),
                    heading=obj2coords.rotation.as_euler("ZYX")[0],
                    transform=obj2coords,
                )
            )

        return out

    def rectangles(self, seq: int, frame: int):
        raise NotImplementedError

    def semantic(self, seq: int, frame: int):
        raise NotImplementedError

    def semantic2d(self, seq: int, frame: int):
        raise NotImplementedError

    def instances(self, seq: int, frame: int):
        raise NotImplementedError

    def instances2d(self, seq: int, frame: int):
        raise NotImplementedError
