Data preparation
================

Tri3D is capable to load most datasets without preprocessing, except for the ones
mentioned on this page.

Waymo
-----

Tri3d supports the Waymo dataset with parquet file format.
However, its files must be re-encoded with better chunking and sorting parameters to allow
faster data loading.

To optimize the sequences in a folder, a script is provided that can be used like so:

.. code-block::

    python -m tri3d.datasets.optimize_waymo \
        --input waymo_open_dataset_v_2_0_1/training \
        --output optimized_waymo/training \
        --workers 8

The resulting files in the output directory will contain the same data but sorted, chunked
and compressed with better settings.


Once
----

Each split ("train", "val", "test", "raw") should be a different subfolder inside the
root dataset directory.
The file hierarchy of each split should follow the `original organization 
<https://once-for-auto-driving.github.io/download.html>`_. 

Assuming all archive are stored together, the following commands will decompress the 
whole dataset as required:

.. code-block::

   find . -name 'train_*.tar' -exec tar --transform="s,^,train/," -xf {} \;
   find . -name 'val_*.tar' -exec tar --transform="s,^,val/," -xf {} \;
   find . -name 'test_*.tar' -exec tar --transform="s,^,test/," -xf {} \;
   find . -name 'raw_*.tar' -exec tar --transform="s,^,raw/," -xf {} \;
   find . -name 'raw_*.tar' -exec tar --transform="s,^,raw/," -xf {} \;
   cat raw_lidar_p*.tar.parta* | tar --transform="s,^,raw/," -xf
