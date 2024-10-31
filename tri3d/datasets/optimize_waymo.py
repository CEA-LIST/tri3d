"""Re-encode Waymo parquet files with better settings to optimize random access."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import gc

import pyarrow
import pyarrow.dataset
import pyarrow.parquet as pq
import pyarrow.compute as pc


sorting_columns = {
    "lidar_segmentation": ["key.frame_timestamp_micros", "key.laser_name"],
    "projected_lidar_box": [
        "key.frame_timestamp_micros",
        "key.camera_name",
        "key.laser_object_id",
    ],
    "lidar_camera_synced_box": ["key.frame_timestamp_micros", "key.laser_object_id"],
    "vehicle_pose": ["key.frame_timestamp_micros"],
    "lidar_pose": ["key.frame_timestamp_micros", "key.laser_name"],
    "camera_segmentation": ["key.frame_timestamp_micros", "key.camera_name"],
    "lidar": ["key.frame_timestamp_micros", "key.laser_name"],
    "stats": ["key.frame_timestamp_micros"],
    "lidar_camera_projection": ["key.frame_timestamp_micros", "key.laser_name"],
    "lidar_hkp": ["key.frame_timestamp_micros", "key.laser_object_id"],
    "camera_calibration": ["key.camera_name"],
    "lidar_calibration": ["key.laser_name"],
    "lidar_box": ["key.frame_timestamp_micros", "key.laser_object_id"],
    "camera_box": [
        "key.frame_timestamp_micros",
        "key.camera_name",
        "key.camera_object_id",
    ],
    "camera_to_lidar_box_association": ["key.frame_timestamp_micros"],
    "camera_image": ["key.frame_timestamp_micros", "key.camera_name"],
    "camera_hkp": [
        "key.frame_timestamp_micros",
        "key.camera_name",
        "key.camera_object_id",
    ],
}


# Compress text and range image columns
compressed_columns = [
    "[CameraSegmentationLabelComponent].sequence_id",
    "[LiDARCameraProjectionComponent].range_image_return1.values",
    "[LiDARCameraProjectionComponent].range_image_return2.values",
    "[LiDARComponent].range_image_return1.values",
    "[LiDARComponent].range_image_return2.values",
    "[LiDARPoseComponent].range_image_return1.values",
    "[LiDARSegmentationLabelComponent].range_image_return1.values",
    "[LiDARSegmentationLabelComponent].range_image_return2.values",
    "[StatsComponent].location",
    "[StatsComponent].time_of_day",
    "[StatsComponent].weather",
    "index",
    "key.camera_object_id",
    "key.laser_object_id",
    "key.segment_context_name",
]


def convert_file(task):
    pyarrow.set_cpu_count(4)
    pyarrow.jemalloc_set_decay_ms(0)

    source, destination = task
    print(source)
    destdir = os.path.dirname(destination)
    if not os.path.exists(destdir):
        os.makedirs(destdir, exist_ok=True)

    record_type = os.path.basename(os.path.dirname(source))

    ds = pyarrow.dataset.dataset(source)
    order = pc.sort_indices(
        ds.to_table(sorting_columns[record_type]),
        sort_keys=[(c, "ascending") for c in sorting_columns[record_type]],
    ).to_numpy()

    table = pyarrow.dataset.dataset(source).take(order)

    gc.collect()
    pyarrow.default_memory_pool().release_unused()

    pq.write_table(
        table,
        destination,
        row_group_size=(
            2 if record_type in ["lidar", "camera_image", "lidar_pose"] else 1024
        ),
        compression={
            c + (".list.element" if c.endswith(".values") else ""): "BROTLI"
            if c in compressed_columns
            else "NONE"
            for c in table.schema.names
        },
        sorting_columns=[
            pq.SortingColumn(table.schema.names.index(c))
            for c in sorting_columns[record_type]
        ],
    )


def list_files(input, output):
    for split in os.listdir(input):
        for record in os.listdir(os.path.join(input, split)):
            for filename in os.listdir(os.path.join(input, split, record)):
                if filename.endswith(".parquet"):
                    yield (
                        os.path.join(input, split, record, filename),
                        os.path.join(output, split, record, filename),
                    )


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("input", help="path to the original dataset")
    argparser.add_argument("output", help="path to the converted dataset")
    argparser.add_argument(
        "--workers", "-w", type=int, default=4, help="number of parallel workers"
    )
    args = argparser.parse_args()

    with ProcessPoolExecutor(
        max_workers=args.workers, max_tasks_per_child=1
    ) as executor:
        list(executor.map(convert_file, list_files(args.input, args.output)))


if __name__ == "__main__":
    main()
