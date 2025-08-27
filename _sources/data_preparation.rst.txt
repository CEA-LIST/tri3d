Data preparation
================

Tri3D is capable of loading most datasets without preprocessing, except for the ones
mentioned on this page.

NuScenes
--------

The nuScenes dataset is missing some of the ego pose information, namely the position 
in z and the rotations along x and y.
This can cause issues for temporal aggregation methods where aggregated point clouds and 
annotation boxes are thus not superimposed.

To remedy this, a SLAM based algorithm is provided which will optimize the alignement 
between pairs of point clouds and regress the missing values.

You can download pre-computed poses which we distribute as JSON patch files.
The following commands will download and create new data splits (`v1.1-mini` and 
`v1.1-trainval`) with corrected poses:

.. code-block:: shell

    pip install jsonpatch

.. code-block:: shell

    curl -L -O https://github.com/CEA-LIST/tri3d/releases/download/v0.2.0/v1.1-mini.tar.gz
    cp -r v1.0-mini v1.1-mini
    tar xf v1.1-mini.tar.gz
    jsonpatch v1.0-mini/ego_pose.json v1.1-mini/ego_pose.jsonpatch > v1.1-mini/ego_pose.json
    jsonpatch v1.0-mini/sample_annotation.json v1.1-mini/sample_annotation.jsonpatch > v1.1-mini/sample_annotation.json
    
.. code-block:: shell

    curl -L -O https://github.com/CEA-LIST/tri3d/releases/download/v0.2.0/v1.1-trainval.tar.gz
    cp -r v1.0-trainval v1.1-trainval
    tar xf v1.1-trainval.tar.gz
    jsonpatch v1.0-trainval/ego_pose.json v1.1-trainval/ego_pose.jsonpatch > v1.1-trainval/ego_pose.json
    jsonpatch v1.0-trainval/sample_annotation.json v1.1-trainval/sample_annotation.jsonpatch > v1.1-trainval/sample_annotation.json

In order to recompute the aligned poses, install the `align_nuscenes` extra dependencies
and invoke the script like so:

.. code-block:: shell

    python -m tri3d.datasets.align_nuscenes \
        --root ~/Datasets/NuScenes \
        --subset v1.0-mini \
        --out ~/Datasets/NuScenes/v1.1-mini

Once
----

Each split ("train", "val", "test", "raw") should be a different subfolder inside the
root dataset directory.
The file hierarchy of each split should follow the `original organization 
<https://once-for-auto-driving.github.io/download.html>`_. 

Assuming all archive are stored together, the following commands will decompress the 
whole dataset as required:

.. code-block:: shell

    find . -name 'train_*.tar' -exec tar --transform="s,^,train/," -xf {} \;
    find . -name 'val_*.tar' -exec tar --transform="s,^,val/," -xf {} \;
    find . -name 'test_*.tar' -exec tar --transform="s,^,test/," -xf {} \;
    find . -name 'raw_*.tar' -exec tar --transform="s,^,raw/," -xf {} \;
    find . -name 'raw_*.tar' -exec tar --transform="s,^,raw/," -xf {} \;
    cat raw_lidar_p*.tar.parta* | tar --transform="s,^,raw/," -xf

Waymo
-----

Tri3d supports the Waymo dataset with parquet file format.
However, its files must be re-encoded with better chunking and sorting parameters to allow
faster data loading.

To optimize the sequences in a folder, use the following command:

.. code-block:: shell

    python -m tri3d.datasets.optimize_waymo \
        --input waymo_open_dataset_v_2_0_1 \
        --output optimized_waymo \
        --workers 4

The resulting files in the output directory will contain the same data but sorted, chunked
and compressed with better settings.

.. warning::

    The script uses **a lot** of memory and may cause OOM.
