"""Evaluate performances using kitti testing methodology."""

import logging
import os
import subprocess
import tempfile
from collections import namedtuple
from os import path

import numpy as np

from .geometry import approx_kitti_bbox2d

logging.basicConfig()
logger = logging.getLogger(__name__)

_evaluate_executable = path.abspath(path.join(
    path.dirname(__file__), "bin", "evaluate_object_3d_offline"))


def _write_kitti_annotation_file(filename, annotations, label_map=None):
    with open(filename, 'w') as f:
        for a in annotations:
            if hasattr(a, 'bbox') and a.bbox is not None:
                u1, j1, u2, j2 = a.bbox
            else:
                u1, j1, u2, j2 = approx_kitti_bbox2d(a.position, a.size, a.heading)
            l, w, h = a.size
            x, y, z = a.position
            y += h / 2
            heading = - a.heading
            f.write(
                f"{label_map[a.label] if label_map is not None else a.label} "
                f"{getattr(a, 'truncated', -1):.2f} "
                f"{getattr(a, 'occluded', -1)} "
                f"{getattr(a, 'alpha', -10):d} "
                f"{u1:.2f} {j1:.2f} {u2:.2f} {j2:.2f} "
                f"{h:.2f} {w:.2f} {l:.2f} "
                f"{x:.2f} {y:.2f} {z:.2f} "
                f"{heading:.2f}"
                + (f" {a.score}\n" if hasattr(a, 'score') else "\n"))


KITTIObjectDet = namedtuple(
    'KITTIObjectDet',
    ['position', 'size', 'heading', 'label', 'score', 'bbox'],
    defaults=[None])


def eval_kitti_det(predictions, groundtruths, label_map=None, out_dir=None):
    """Compute PR curves using KITTI evaluation protocol.

    :param predictions:
        A nested list of predictions with the following attributes:
          - `position` object *center* position in kitti camera coordinates
            convention (x right-ward, y down-ward, z forward)
          - `size` l, w, h triplet (not h, w, l as in kitti annotations)
          - `heading` object rotation around y (which points down-ward)
          - `label` predicted label
          - `score` confidence score
        And optionaly:
          - `bbox` bbox in camera image plane
    :param grountruths:
        A nested list of annotations with the following attributes:
          - `position` object *center* position in kitti camera coordinates
            convention (x right-ward, y down-ward, z forward)
          - `size` l, w, h triplet (not h, w, l as in kitti annotations)
          - `heading` object rotation around y (which points down-ward)
          - `label` annotated label
        And optionaly:
          - `bbox` bbox in camera image plane
          - `truncated` truncation level
          - `occluded` occlusion level
          - `height_2d` the height in pixel in camera image
    :param label_map:
        A dictionary mapping from predicted and annotated labels to
        KITTI class names.

    :return:
        A dictionary with the PR curves for `car_3d`, `car_bev`,
        `pedestrian_3d`, `pedestrian_bev`, `cyclist_3d`, `cyclist_bev`
        in `easy`, `moderate`, `hard` difficulties.
    """
    if out_dir is None:
        with tempfile.TemporaryDirectory() as out_dir:
            return eval_kitti_det(predictions, groundtruths, label_map, out_dir)

    # TODO: prediction must include 2d height
    # TODO: recall thresholds?
    os.makedirs(path.join(out_dir, "predictions"), exist_ok=True)
    os.makedirs(path.join(out_dir, "predictions", "data"), exist_ok=True)
    os.makedirs(path.join(out_dir, "groundtruth"), exist_ok=True)

    result_dir = path.join(out_dir, "predictions")
    gt_dir = path.join(out_dir, "groundtruth")

    for i, annotations in enumerate(groundtruths):
        _write_kitti_annotation_file(
            filename=path.join(gt_dir, f"{i:06d}.txt"),
            annotations=annotations,
            label_map=label_map)

    for i, annotations in enumerate(predictions):
        _write_kitti_annotation_file(
            filename=path.join(result_dir, "data", f"{i:06d}.txt"),
            annotations=annotations,
            label_map=label_map)

    # Override plotting programs with fake ones to skip plotting
    if not path.exists(path.join(out_dir, 'bin')):
        os.makedirs(path.join(out_dir, 'bin'))
        os.symlink('/bin/true', path.join(out_dir, 'bin', 'gnuplot'))
        os.symlink('/bin/true', path.join(out_dir, 'bin', 'ps2pdf'))
        os.symlink('/bin/true', path.join(out_dir, 'bin', 'pdfcrop'))

    # Run evaluation program
    os.environ['PATH'] = path.join(out_dir, 'bin') + ":" + os.environ['PATH']
    proc = subprocess.run([_evaluate_executable, gt_dir, result_dir],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    os.environ['PATH'] = ":".join(os.environ['PATH'].split(":")[1:])
    if proc.returncode != 0:
        raise RuntimeError(f"evaluation script error ({proc.returncode}):\n"
                           f"{proc.stdout}")

    results = {}
    for c in ['car', 'pedestrian', 'cyclist']:
        filename = path.join(result_dir, f"stats_{c.lower()}_detection_3d.txt")
        try:
            easy, moderate, hard = np.loadtxt(filename)
        except IOError:
            zero = np.zeros(41)
            results[c + "_3d"] = {'easy': zero, 'moderate': zero, 'hard': zero}
        else:
            results[c + "_3d"] = {'easy': easy, 'moderate': moderate, 'hard': hard}

        filename = path.join(result_dir, f"stats_{c.lower()}_detection_ground.txt")
        try:
            easy, moderate, hard = np.loadtxt(filename)
        except IOError:
            zero = np.zeros(41)
            results[c + "_ground"] = {'easy': zero, 'moderate': zero, 'hard': zero}
        else:
            results[c + "_ground"] = {'easy': easy, 'moderate': moderate, 'hard': hard}

    return results
