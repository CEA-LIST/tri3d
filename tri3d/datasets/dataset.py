import dataclasses
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from PIL.Image import Image

from .. import misc
from ..geometry import RigidTransform, Transformation


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


class AbstractDataset(ABC):
    """Abstract class for driving datasets.

    Each implementation should document availables sensors and the label
    by which they are refered to. They are used to query data and to specify
    the timeline and spatial coordinate systems.

    To implement a new dataset, you might want to derive instead from
    :code:`tri3d.datasets.Dataset` as it already provides boilerplate code for most methods.

    .. run::

        import matplotlib.pyplot as plt
        import numpy as np
        from tri3d.datasets import Waymo

        plt.switch_backend("Agg")

        dataset = Waymo("datasets/waymo")
        name = "tri3d.datasets.Waymo"
        camera, imgcoords, lidar = "CAM_FRONT", "IMG_FRONT", "LIDAR_TOP"
        seq, frame, cam_frame = 0, 24, 24
    """

    cam_sensors: list[str]
    """Camera names."""

    img_sensors: list[str]
    """Camera names (image plane coordinate)."""

    pcl_sensors: list[str]
    """Point cloud sensor names."""

    det_labels: list[str]
    """Detection labels."""

    sem_labels: list[str]
    """Segmentation labels."""

    @abstractmethod
    def sequences(self) -> list[int]:
        """Return the list of sequences/recordings indices
        (0..num_sequences)."""
        ...

    @abstractmethod
    def frames(self, seq: int, sensor: str) -> np.ndarray:
        """Return the frames indices of particular sequence for a sensor.

        The indices are normally contiguous (ie: :func:`np.arange`).

        :param seq:
            Sequence index.
        :param seq:
            Sequence index.
        :return:
            A list of (sequence, frame) index tuples sorted by sequence and
            frame.
        """
        ...

    @abstractmethod
    def timestamps(self, seq: int, sensor: str) -> np.ndarray:
        """Return the frame timestamps for a given sensor .

        :param seq:
            Sequence index.
        :param sensor:
            Sensor name.
        :return:
            An array of timestamps.

        .. note:: frames are guaranteed to be sorted.

        .. run::

            plt.figure(figsize=(8, 4))

            t0, _, t1 = dataset.timestamps(seq, lidar)[:3]

            sensors = dataset.pcl_sensors + dataset.cam_sensors
            for i, s in enumerate(sensors):
                t = (dataset.timestamps(seq, s) - t0)
                plt.scatter(t, np.full(len(t), i), marker="|", s=100., label=s)

            plt.yticks(np.arange(len(sensors)), labels=sensors)

            plt.xlim(-0.1 * (t1 - t0), t1 - t0 + 0.1 * (t1 - t0))
            plt.xlabel("time")

            plt.savefig(f"docs/source/_static/{name}.timestamps.jpg")

            sphinxrun.show(
                f".. figure:: /_static/{name}.timestamps.jpg"
            )
        """
        ...

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

        .. run::

            plt.figure(figsize=(8, 2.5))

            p0 = dataset.poses(seq, lidar)[0]
            sensors = dataset.pcl_sensors + dataset.cam_sensors
            for i, s in enumerate(sensors):
                if len(dataset.timestamps(0, s)) == 0:
                    continue

                for j, u in enumerate(np.eye(3) * 0.2):
                    p = p0.inv() @ dataset.poses(seq, s)[:4]
                    b = p.apply(u)
                    a = p.apply([0.0, 0.0, 0.0])
                    plt.quiver(
                        a[:, 0],
                        a[:, 1],
                        (b - a)[:, 0],
                        (b - a)[:, 1],
                        width=0.004,
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        color=plt.get_cmap("tab10")(i),
                        label=s if j == 0 else None,
                    )
                    plt.scatter(b[:, 0], b[:, 1], alpha=0.)

            plt.gcf().legend(loc="outside upper center", ncol=5, fontsize="x-small")
            plt.gca().set_aspect("equal")

            plt.savefig(f"docs/source/_static/{name}.poses.jpg")

            sphinxrun.show(
                f".. figure:: /_static/{name}.poses.jpg"
            )
        """
        ...

    @abstractmethod
    def alignment(
        self,
        seq: int,
        frame: int | tuple[int, int],
        coords: str | tuple[str, str],
    ) -> Transformation:
        """Return the transformation from one coordinate system and timestamp
        to another.

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
        ...

    @abstractmethod
    def image(self, seq: int, frame: int, sensor: str) -> Image:
        """Return image from given camera at given frame.

        A default sensor (for instance a front facing camera) should be
        provided for convenience.

        :param seq:
            Sequence index.
        :param frame:
            Frame index.
        :param sensor:
            The image sensor to use.

        .. run::

            from tri3d.misc import nearest_sorted

            try:
                image = dataset.image(seq, cam_frame, camera)
                plt.figure(figsize=(8, 6))
                plt.imshow(image)
                plt.savefig(f"docs/source/_static/{name}.image.jpg")

                sphinxrun.show(
                    f".. figure:: /_static/{name}.image.jpg"
                )
            except NotImplementedError:
                pass
        """
        ...

    @abstractmethod
    def points(
        self, seq: int, frame: int, sensor: str, coords: str | None = None
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

        .. run::

            xyz = dataset.points(seq, frame, lidar)[:, :3]
            xyz = xyz[np.argsort(xyz[:, 2])]

            plt.figure(figsize=(8, 5))
            plt.scatter(xyz[:, 0], xyz[:, 1], s=1., c=xyz[:, 2], clim=(-1, 2.))
            plt.xlim(-25, 25)
            plt.ylim(-10, 15)
            plt.gca().set_aspect("equal")
            plt.savefig(f"docs/source/_static/{name}.points.jpg")

            sphinxrun.show(
                f".. figure:: /_static/{name}.points.jpg"
            )
        """
        ...

    @abstractmethod
    def boxes(self, seq: int, frame: int, coords: str) -> Sequence[type[Box]]:
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

        .. run::

            from tri3d.plot import plot_bbox_cam

            try:
                image = dataset.image(seq, cam_frame, camera)
                boxes = dataset.boxes(seq, cam_frame, coords=imgcoords)

                plt.figure(figsize=(8, 5))
                plt.imshow(image)
                for b in boxes:
                    plot_bbox_cam(b.transform, b.size, image.size)

                plt.savefig(f"docs/source/_static/{name}.boxes.jpg")

                sphinxrun.show(
                    f".. figure:: /_static/{name}.boxes.jpg"
                )
            except NotImplementedError:
                pass
        """
        ...

    @abstractmethod
    def rectangles(self, seq: int, frame: int, sensor: str):
        """Return a list of 2D rectangle annotations.

        .. note:: The default coordinate system should be documented.

        :param seq:
            Sequence index.
        :param frame:
            Frame index or `None` to request annotations for the whole sequence
        :return:
            A list of 2D annotations.
        """
        ...

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

        .. run::

            from tri3d.plot import gen_discrete_cmap

            cmap = gen_discrete_cmap(len(dataset.sem_labels))

            try:
                xyz = dataset.points(seq, frame, lidar)[:, :3]
                semantic = dataset.semantic(seq, frame, lidar)
                order = np.argsort(xyz[:, 2])
                xyz = xyz[order]
                semantic = semantic[order]

                plt.figure(figsize=(8, 6))
                plt.scatter(xyz[:, 0], xyz[:, 1], s=1., c=semantic, cmap=cmap)
                plt.xlim(-25, 25)
                plt.ylim(-10, 15)
                plt.gca().set_aspect("equal")

                plt.savefig(f"docs/source/_static/{name}.semantic.jpg")

                sphinxrun.show(
                    f".. figure:: /_static/{name}.semantic.jpg"
                )
            except NotImplementedError:
                pass
        """
        ...

    @abstractmethod
    def semantic2d(self, seq: int, frame: int, sensor: str):
        """Return pixelwise class annotations.

        :param seq: Sequence index.
        :param frame: Frame index.
        :return: array of pointwise class label
        """
        ...

    @abstractmethod
    def instances(self, seq: int, frame: int, sensor: str):
        """Return pointwise instance ids.

        :param seq:
            Sequence index.
        :param frame:
            Frame index.
        :return:
            array of pointwise instance label

        .. run::

            try:
                xyz = dataset.points(seq, frame, lidar)[:, :3]
                instances = dataset.instances(seq, frame, lidar)
                _, instances = np.unique(instances, return_inverse=True)
                cmap = gen_discrete_cmap(instances.max() + 1)
                order = np.argsort(xyz[:, 2])
                xyz = xyz[order]
                instances = instances[order]

                plt.figure(figsize=(8, 6))
                plt.scatter(xyz[:, 0], xyz[:, 1], s=1., c=instances, cmap=cmap)
                plt.xlim(-25, 25)
                plt.ylim(-10, 15)
                plt.gca().set_aspect("equal")

                plt.savefig(f"docs/source/_static/{name}.instances.jpg")

                sphinxrun.show(
                    f".. figure:: /_static/{name}.instances.jpg"
                )
            except NotImplementedError:
                pass
        """
        ...

    @abstractmethod
    def instances2d(self, seq: int, frame: int, sensor: str):
        """Return pixelwise instance annotations.

        Background label pixels will contain -1. Other instance ids will
        follow dataset-specific rules.

        :param seq: Sequence index.
        :param frame: Frame index.
        :return: array of pointwise instance label
        """
        ...


class Dataset(AbstractDataset):
    # Base dataset with generic implementation of common methods
    # You should probably add cache (misc.memoize_method) over frequently called
    # methods.

    cam_sensors: list[str]
    img_sensors: list[str]
    pcl_sensors: list[str]
    det_labels: list[str]
    sem_labels: list[str]

    _nn_interp_thres = 0.05
    _default_cam_sensor: str
    _default_pcl_sensor: str
    _default_box_coords: str

    @abstractmethod
    def _calibration(
        self, seq: int, src_sensor: str, dst_sensor: str
    ) -> Transformation: ...

    @abstractmethod
    def _poses(self, seq: int, sensor: str) -> RigidTransform: ...

    @abstractmethod
    def _points(self, seq: int, frame: int, sensor: str) -> np.ndarray: ...

    @abstractmethod
    def _boxes(self, seq: int) -> Sequence[Box]: ...

    def frames(self, seq: int, sensor: str) -> np.ndarray:
        return np.arange(len(self.timestamps(seq, sensor)))

    def poses(
        self, seq: int, sensor: str, timeline: str | None = None
    ) -> RigidTransform:
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
                tim_sensor_poses: RigidTransform = self._calibration(
                    seq, sensor, timeline
                )  # type: ignore
                return self._poses(seq, timeline) @ tim_sensor_poses
            except ValueError:
                pass

        # interpolate sensor poses to timeline
        try:  # use sensor poses when available
            poses = self._poses(seq, sensor)
            poses_t = self.timestamps(seq, sensor)
        except ValueError:  # use imu and imu->sensor calib
            sensor2ego: RigidTransform = self._calibration(seq, sensor, "ego")  # type: ignore
            poses = self._poses(seq, "ego") @ sensor2ego
            poses_t = self.timestamps(seq, "ego")

        dst_t = self.timestamps(seq, timeline)

        i1, i2 = misc.lr_bisect(poses_t, dst_t)
        t1 = poses_t[i1]
        t2 = poses_t[i2]
        alpha = (dst_t - t1) / (t2 - t1).clip(min=1e-6)

        return RigidTransform.interpolate(poses[i1], poses[i2], alpha)

    def alignment(
        self,
        seq: int,
        frame: int | tuple[int, int],
        coords: str | tuple[str, str],
    ) -> Transformation:
        # normalize arguments
        if isinstance(frame, int):
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

    def image(self, seq: int, frame: int, sensor: str) -> Image:
        raise NotImplementedError

    def points(
        self, seq: int, frame: int, sensor: str, coords: str | None = None
    ) -> np.ndarray:
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

    def boxes(self, seq: int, frame: int, coords: str) -> Sequence[type[Box]]:
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
        sensor_ts: int = self.timestamps(seq, coords)[frame]
        boxes_timestamps = self.timestamps(seq, "boxes")
        ann_frame = misc.nearest_sorted(boxes_timestamps, sensor_ts)

        # use nearest frame when box and sensor timestamps are close
        if abs(sensor_ts - boxes_timestamps[ann_frame]) < self._nn_interp_thres:
            boxes = [b for b in boxes if b.frame == ann_frame]

            ann2coords = self.alignment(seq, (ann_frame, frame), ("boxes", coords))
            if coords in self.img_sensors:
                sensor = self.cam_sensors[self.img_sensors.index(coords)]
                ann2sensor = self.alignment(seq, (ann_frame, frame), ("boxes", sensor))
            else:
                ann2sensor = ann2coords

            out = []
            for b in boxes:
                obj2coords = ann2coords @ b.transform
                if coords in self.cam_sensors:
                    obj2sensor = ann2sensor @ b.transform
                    heading = -obj2sensor.rotation.as_euler("YZX")[0] - np.pi / 2  # type: ignore
                else:
                    heading = obj2coords.rotation.as_euler("ZYX")[0]  # type: ignore
                out.append(
                    dataclasses.replace(
                        b,
                        center=obj2coords.apply([0, 0, 0]),
                        heading=heading,
                        transform=obj2coords,
                    )
                )

            return out

        # Don't interpolate if only one frame is annotated (ex: ZOD frames)
        if len(boxes_timestamps) < 2:
            return []

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
            self.alignment(seq, (i1, frame), ("boxes", coords)),  # type: ignore
            self.alignment(seq, (i2, frame), ("boxes", coords)),  # type: ignore
            w,
        )

        out = []
        for b1, b2 in tracks:
            obj2coords = ann2coords @ RigidTransform.interpolate(
                b1.transform,  # type: ignore
                b2.transform,  # type: ignore
                w,  # type: ignore
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

    def rectangles(self, seq: int, frame: int, sensor: str):
        raise NotImplementedError

    def semantic(self, seq: int, frame: int, sensor: str):
        raise NotImplementedError

    def semantic2d(self, seq: int, frame: int, sensor: str):
        raise NotImplementedError

    def instances(self, seq: int, frame: int, sensor: str):
        raise NotImplementedError

    def instances2d(self, seq: int, frame: int, sensor: str):
        raise NotImplementedError
