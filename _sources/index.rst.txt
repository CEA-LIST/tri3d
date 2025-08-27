Introduction
============

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation

   self
   installation
   data_preparation
   example

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Package reference

   datasets
   geometry
   plot

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: External links

   Github repository <https://github.com/CEA-LIST/tri3d>
   PyPi package <https://pypi.org/project/tri3d>

.. image:: waymo.jpg
  :alt: Waymo sample

Tri3D facilitates the utilization of 3D driving datasets by providing:

- A common coordinate and data encoding convention.
- A common API to read data form various datasets.
- Fast non-sequential access to any sensor sample at any frame index.
- A convenient access to geometric transformations between sensors and timestamps.
- Plotting utilities.

As of now, Tri3D supports the following datasets: 

- `NuScenes <https://www.nuscenes.org/nuscenes>`_
- `ONCE Dataset <https://once-for-auto-driving.github.io>`_
- `Semantic KITTI <https://semantic-kitti.org>`_
- `Waymo open dataset <https://waymo.com/open>`_
- `Zenseact Open Dataset <https://zod.zenseact.com/frames>`_


Conventions
-----------

The following conventions are adopted across all datasets :

- An object **position** designates the center of its bounding box.
- In the **local coordinate system of an object**, the x axis points forward, y leftward, and z updward.
- **Length, width and height** are the dimensions of the object along x, y and z axes respectively.
- In **camera sensor coordinates**, x points rightward, y points downward, z point forward.
- In **image coordinates**, x is the pixel column index starting from the left, y is the pixel row index starting from the top, z is the depth in meters along the optical axis.
- In **lidar coordinates**, x points in the same direction as the ego car, y points leftward, z points upward.

The differences with raw dataset conventions are documented in each datasets class.

.. image:: coordinates.svg
  :alt: Coordinates convention

Tri3D accounts for timestamps, delays and the movement of the ego car which carries the sensors.
For examples, if the boxes of a given dataset are annotated at the timestamps of 
the LiDAR, retrieving the boxes relative to a camera sensor 
(ex: :meth:`boxes(seq=0, frame=5, sensor="cam") <tri3d.datasets.NuScenes.boxes>`)
will interpolate the box trajectories at the timestamp of frame 5 for the camera, and 
it will also return the object poses relatively to the position of that camera at this 
timestamp.

.. image:: timelines.svg
  :alt: Sensor timelines and track interpolation

Dataset API
-----------

All datasets implement a common interface which provides access to data samples such as:

- Sensor :meth:`frames <tri3d.datasets.NuScenes.frames>`.
- Acquisition :meth:`timestamps <tri3d.datasets.NuScenes.timestamps>`.
- Sensor :meth:`poses <tri3d.datasets.NuScenes.poses>`.
- Camera :meth:`images <tri3d.datasets.NuScenes.image>`.
- Lidar :meth:`point clouds <tri3d.datasets.NuScenes.points>`.
- 3D :meth:`box annotations <tri3d.datasets.NuScenes.boxes>`.

Moreover, a powerful :meth:`.alignment() <tri3d.datasets.NuScenes.alignment>` 
function can compute the geometric transformation between any pair of frame 
and sensors coordinates.
The `tutorial notebook <./example.ipynb>`_ goes through most of these functions.

.. note::

   Tri3D datasets are somewhat low-level, ie. they do not enforce the notion of sweep or
   keyframe where a sample of each sensor around a timestamp is assembled in a tuple.
   Instead, datasets expose all available samples of each sensor indexed by a per-sensor 
   frame index.
   Notably, some sensors may work at a higher frequency and contain more samples than 
   others for a given recording.
   
   When available, keyframes and timestamps are exposed so that coherent tuples of 
   samples can be rebuilt easily.
   Moreover, geometric transformations and pose interpolation functions are provided to 
   facilitate the creation of new keyframes.

Geometric transformations
-------------------------

Tri3D provides a small library which facilitates the creation and manipulation of
typical 3D geometric transformations: translation, rotation, affine, camera projections.

Transformations have a :class:`shared interface <tri3d.geometry.Transformation>` which supports:

- **Batching**: list of transformations can be grouped together, and broadcasting rules 
  are supported.
- **Application** to 3D points: applying the transformations to points, again, boadcasting
  rules are implemented.
- **Composition**: it is possible to chain transformation together.
- **Inversion**: the inverse of a transformation is readily available.

Objects and sensors poses
-------------------------

In Tri3D, the position and orientation of objects (or sensors) in a coordinate system
are stored as geometric transformation.
For instance, the pose of a camera in a lidar coordinate system is formulated as the 
composition of the camera rotation and translation relative to the lidar.
Coincidently, this is also the geometric transformation which takes points in the 
camera coordinate system and returns them in the lidar coordinate system.

For sensors, the sensor poses are accessible via the 
:meth:`.poses()<tri3d.datasets.NuScenes.poses>` method.
For object annotations, it is provided by the 
:meth:`Box.transform<tri3d.datasets.Box.transform>` attribute.

For example, if we have the pose of a box in lidar coordinates and the pose of the
lidar in camera coordinates, then the position of a point 10 meters in front of that 
box is given by: 

.. code::

   seq = 0
   frame = 4

   # retrieve poses
   box = dataset.boxes(seq, frame, coords="lidar")[0]
   box2lidar = box.transform
   lidar2cam = dataset.poses(seq, sensor="camera", coords="lidar")[frame]

   # compute the position of a point 10m in front of the box, in camera coordinates.
   xyz = (lidar2cam @ box2lidar).apply([10., 0, 0])

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
