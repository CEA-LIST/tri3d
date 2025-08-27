"""Re-encode Waymo parquet files with better settings to optimize random
access.

The script can be called again if it is stopped or crashes, it will skip
already converted files. It will use **a lot** of RAM.
"""

import argparse
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
import tqdm

SORTING_COLUMNS = {
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
COMPRESSED_COLUMNS = [
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


def convert_file(source: pathlib.Path, destination: pathlib.Path):
    destdir = os.path.dirname(destination)
    if not os.path.exists(destdir):
        os.makedirs(destdir, exist_ok=True)

    tmpdest = destination.parent / (destination.name + ".tmp")

    tmpdest.parent.mkdir(parents=True, exist_ok=True)

    record_type = os.path.basename(os.path.dirname(source))

    table = pq.ParquetFile(source, memory_map=True).read()
    table = table.sort_by([(c, "ascending") for c in SORTING_COLUMNS[record_type]])

    pq.write_table(
        table,
        tmpdest,
        row_group_size=(
            1 if record_type in ["lidar", "camera_image", "lidar_pose"] else 1024
        ),
        # compression="ZSTD",
        compression={
            c + (".list.element" if c.endswith(".values") else ""): "ZSTD"
            if c in COMPRESSED_COLUMNS
            else "NONE"
            for c in table.schema.names
        },
        compression_level={
            c + (".list.element" if c.endswith(".values") else ""): 4
            for c in table.schema.names
            if c in COMPRESSED_COLUMNS
        },
        sorting_columns=[
            pq.SortingColumn(table.schema.names.index(c))
            for c in SORTING_COLUMNS[record_type]
        ],
    )

    tmpdest.rename(destination)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--input", type=pathlib.Path, help="path to the original dataset"
    )
    argparser.add_argument(
        "--output", type=pathlib.Path, help="path to the converted dataset"
    )
    argparser.add_argument(
        "--workers", "-w", type=int, default=4, help="number of parallel workers"
    )
    args = argparser.parse_args()

    sources: list[pathlib.Path] = list(args.input.glob("**/*.parquet"))
    sources = sorted(sources, key=lambda s: (s.name, s.parent.name))

    destinations = [args.output / s.relative_to(args.input) for s in sources]

    keep = [i for i, d in enumerate(destinations) if not d.exists()]
    sources = [sources[k] for k in keep]
    destinations = [destinations[k] for k in keep]

    with ProcessPoolExecutor(
        max_workers=args.workers, max_tasks_per_child=1
    ) as executor:
        for _ in tqdm.tqdm(
            executor.map(convert_file, sources, destinations), total=len(sources)
        ):
            pass


if __name__ == "__main__":
    main()
