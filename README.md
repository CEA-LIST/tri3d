# Introduction

![Waymo sample](docs/source/waymo.jpg)

Tri3D provides a library to manipulate various 3D driving datasets under a unified API with common conventions.

It aims to address the following objectives:

- Make datasets **interchangeable**.
- Enforce **common coordinate and data format convention**.
- Provide **fast non-sequential access to samples** at any given frame.
- **Simplify geometric transformations** between sensors and timestamps.

As of now, Tri3D supports the following datasets:

- [Argoverse 2](https://cea-list.github.io/tri3d/generated/tri3d.datasets.Argoverse.html)
- [NuScenes](https://cea-list.github.io/tri3d/generated/tri3d.datasets.NuScenes.html)
- [ONCE Dataset](https://cea-list.github.io/tri3d/generated/tri3d.datasets.Once.html)
- [Semantic KITTI](https://cea-list.github.io/tri3d/generated/tri3d.datasets.SemanticKITTI.html)
- [Waymo open dataset](https://cea-list.github.io/tri3d/generated/tri3d.datasets.Waymo.html)
- [Zenseact Open Dataset](https://cea-list.github.io/tri3d/generated/tri3d.datasets.ZODFrames.html)

The documentation is hosted at: https://cea-list.github.io/tri3d/

## Conventions

The following conventions are adopted across all datasets :

- An object **position** designates the center of its bounding box.
- In the **local coordinate system of an object**, the x axis points
  forward, y leftward, and z updward.
- **Length, width and height** are the dimensions of the object along
  x, y and z axes respectively.
- In **camera sensor coordinates**, x points rightward, y points
  downward, z point forward.
- In **image coordinates**, x is the pixel column index starting from
  the left, y is the pixel row index starting from the top, z is the
  depth in meters along the optical axis.
- In **lidar coordinates**, x points in the same direction as the ego
  car, y points leftward, z points upward.

The differences with raw dataset conventions are documented in each
datasets class.

![Coordinates convention](docs/source/coordinates.svg)

Tri3D accounts for timestamps, delays and the movement of the ego car
which carries the sensors. For examples, if the boxes of a given dataset
are annotated at the timestamps of the LiDAR, retrieving the boxes
relative to a camera sensor
(ex: `boxes(seq=0, frame=5, sensor="cam")`) will interpolate the box trajectories at the timestamp of
frame 5 for the camera, and it will also return the object poses
relatively to the position of that camera at this timestamp.

![Sensor timelines and track interpolation](docs/source/timelines.svg)

## Dataset API

All datasets implement a common interface which provides access to data
samples such as:

- Sensor frames.
- Acquisition timestamps.
- Sensor poses.
- Camera images.
- Lidar point clouds.
- 3D box annotations.

Moreover, a powerful `.alignment()` function can compute the geometric transformation between any pair of frame and sensors coordinates.
The [tutorial notebook](docs/source/example.ipynb) goes through most of these functions.

Tri3D datasets are somewhat low-level, ie. they do not enforce the
notion of sweep or keyframe where a sample of each sensor around a
timestamp is assembled in a tuple. Instead, datasets expose all
available samples of each sensor indexed by a per-sensor frame index.
Notably, some sensors may work at a higher frequency and contain more
samples than others for a given recording.

When available, keyframes and timestamps are exposed so that coherent
tuples of samples can be rebuilt easily. Moreover, geometric
transformations and pose interpolation functions are provided to
facilitate the creation of new keyframes.

## Geometric transformations

Tri3D provides a module which facilitates the creation and
manipulation of typical 3D geometric transformations: translation,
rotation, affine, camera projections, pose interpolation.

Transformations have a shared interface which supports:

- **Batching**: list of transformations can be grouped together, and
  broadcasting rules are supported.
- **Application** to 3D points: applying the transformations to
  points, again, boadcasting rules are implemented.
- **Composition**: it is possible to chain transformation together.
- **Inversion**: the inverse of a transformation is readily available.

## Object and sensor poses

In Tri3D, the position and orientation of objects (or sensors) in a
coordinate system are stored as geometric transformation. For instance,
the pose of a camera in a lidar coordinate system is formulated as the
composition of the camera rotation and translation relative to the
lidar. Coincidently, this is also the geometric transformation which
takes points in the camera coordinate system and returns them in the
lidar coordinate system.

For sensors, the sensor poses are accessible via the `.poses()` method.
For object annotations, it is provided by the `Box.transform` attribute.

For example, if we have the pose of a box in lidar coordinates and the
pose of the lidar in camera coordinates, then the position of a point 10
meters in front of that box is given by:

```py
seq = 0
frame = 4

# retrieve poses
box = dataset.boxes(seq, frame, coords="lidar")[0]
box2lidar = box.transform
lidar2cam = dataset.poses(seq, sensor="camera", coords="lidar")[frame]

# compute the position of a point 10m in front of the box, in camera coordinates.
xyz = (lidar2cam @ box2lidar).apply([10., 0, 0])
```
