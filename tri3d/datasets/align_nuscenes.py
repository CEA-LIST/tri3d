"""Fix absence of z position and xy rotation data in the ego pose."""

import argparse
import functools
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch_cluster
from torch import Tensor
from torch.nn import functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from ..geometry import RigidTransform, as_matrix
from ..misc import lr_bisect
from .nuscenes import NuScenes


def subsample(xyz: Tensor, bin_size=0.075):
    idx = xyz.add(xyz.min()).div(bin_size).long()
    idx = idx[:, 0] * 2**40 + idx[:, 1] * 2**20 + idx[:, 2]
    u, index = idx.unique(return_inverse=True)
    out = xyz.new_empty([u.shape[0], 3]).scatter_reduce_(
        dim=0,
        index=index.unsqueeze(1).expand(-1, 3),
        src=xyz,
        reduce="mean",
        include_self=False,
    )
    return out


def correction_matrix(dr, dz):
    """Convert variables to a 4x4 transformation matrix."""
    cx = torch.cos(dr[0])
    sx = torch.sin(dr[0])
    cy = torch.cos(dr[1])
    sy = torch.sin(dr[1])
    z = dr.new_zeros([])
    o = dr.new_ones([])

    return torch.stack(
        [cy, sx * sy, -cx * sy, z, z, cx, sx, z, sy, -sx * cy, cx * cy, dz, z, z, z, o]
    ).view(4, 4)


def icp(
    xyz_source: Tensor,
    xyz_target: Tensor,
    init_transform: Tensor,
    total_steps: int = 50,
    lr: float = 0.02,
    eps: float = 0.1,
):
    device = xyz_source.device

    # Variables to optimize: rotation around x and y and tranlation along z
    dr = torch.zeros([2], device=device, requires_grad=True)
    dz = torch.zeros([], device=device, requires_grad=True)

    optimizer = AdamW([dr, dz], lr=0.0001, fused=True)
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=lr, total_steps=total_steps)

    xyz_source_ = xyz_source @ init_transform[:3, :3].T + init_transform[:3, 3]
    nn = torch_cluster.nearest(xyz_source_, xyz_target)

    for it in range(total_steps):
        fix_transform = correction_matrix(dr, dz)
        t = init_transform @ fix_transform

        xyz_source_ = xyz_source @ t[:3, :3].T + t[:3, 3]

        if it > 0 and it % 10 == 0:
            nn = torch_cluster.nearest(xyz_source_, xyz_target)

        dist = xyz_source_ - xyz_target[nn]
        dist[:, 2] *= 4
        dist = dist.norm(dim=1)

        thres = dist.detach().quantile(0.95).clamp_max(eps)
        inliners = dist < thres
        loss = (dist * inliners).square()

        loss = loss * torch.where(xyz_source_[:, 2] < -1.0, 1.0, 0.5)

        loss = loss.mean()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    with torch.no_grad():
        fix_transform = correction_matrix(dr, dz)

    return init_transform @ fix_transform


def aggregate_points(
    dataset: NuScenes, seq, frame, window, min_dist=3.0, max_dist=40.0, device="cuda"
):
    n_pcds = len(dataset.timestamps(seq, "LIDAR_TOP"))

    xyz = []
    f = min(max(-window[0], frame), n_pcds - window[-1] - 1)
    for i in window:
        xyz_f = dataset.points(seq, f + i)[:, :3]

        t = dataset.alignment(seq, (f + i, frame), "LIDAR_TOP")
        xyz_f = t.apply(xyz_f)

        xyz_f = torch.from_numpy(xyz_f).float()

        xyz_f = xyz_f[xyz_f.norm(dim=1) > min_dist]

        xyz.append(xyz_f)

    xyz = torch.cat(xyz)
    xyz = xyz[xyz.norm(dim=1) < max_dist].to(device, non_blocking=True)

    return subsample(xyz)


def probable_poses(transform_graph, start_pose, start_node=0):
    out = {start_node: start_pose}

    for next_node in sorted(set(j for _, j in transform_graph.keys())):
        predecessors = [i for (i, j) in transform_graph.keys() if j == next_node]
        poses = [
            out[p] @ torch.linalg.inv(transform_graph[(p, next_node)]).double()
            for p in predecessors
        ]
        poses = torch.stack([p.ravel() for p in poses])
        d = torch.cdist(poses, poses).mean(dim=1)
        i = d.argmin()
        out[next_node] = poses[i].view(4, 4)

    return out


def build_pose_graph(dataset: NuScenes, seq, stride=5, window=7, device="cuda"):
    """Graph of transformations between lidar sweeps."""

    n_pcds = len(dataset.timestamps(seq, "LIDAR_TOP"))
    graph = {}

    for source_id in range(0, n_pcds, stride):
        for target_id in range(
            source_id + stride, source_id + (window + 1) * stride, stride
        ):
            if target_id >= n_pcds:
                continue

            xyz_source = aggregate_points(
                dataset, seq, source_id, [-2, 0, 2], device=device
            )
            xyz_target = aggregate_points(
                dataset, seq, target_id, [-4, -2, 0, 2, 4], device=device
            )
            init_transform = (
                torch.from_numpy(
                    as_matrix(
                        dataset.alignment(seq, (source_id, target_id), "LIDAR_TOP")
                    )
                )
                .float()
                .to(device, non_blocking=True)
            )

            transform = icp(xyz_source, xyz_target, init_transform)
            transform = transform.cpu()

            graph[(source_id, target_id)] = transform

    return graph


def bundle_adjustment(dataset: NuScenes, seq, graph: dict[tuple[int, int], Tensor]):
    poses_init = probable_poses(
        graph, torch.from_numpy(as_matrix(dataset.poses(seq, "LIDAR_TOP")[0]))
    )

    poses = {i: p.clone().requires_grad_() for i, p in poses_init.items()}

    grad_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=torch.double,
    )

    total_steps = 1000
    optimizer = SGD(poses.values(), lr=1e-4)
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=1e-3, total_steps=total_steps)

    for i in range(total_steps):
        optimizer.zero_grad()

        icp_losses = {b: [] for _, b in graph.keys()}
        for (a, b), icp_ab in graph.items():
            icp_losses[b].append(
                F.mse_loss(torch.linalg.inv(poses[b]) @ poses[a], icp_ab.double())
            )

        icp_losses = [torch.stack(l) for l in icp_losses.values()]
        icp_losses = [
            torch.sum(l * l.le(l.detach().min().clamp_min(1e-6))) for l in icp_losses
        ]
        icp_loss = sum(icp_losses)

        norm_losses = []
        for p in poses.values():
            rot_loss = F.mse_loss(
                p[:3, :3] @ p[:3, :3].T, torch.eye(3, dtype=torch.double)
            )
            trs_loss = F.mse_loss(p[3], torch.eye(4, dtype=torch.double)[3])
            norm_losses.append(rot_loss + trs_loss)

        norm_losses = torch.stack(norm_losses)
        norm_loss = norm_losses.sum()

        total_loss = icp_loss + norm_loss
        total_loss.backward()

        for p in poses.values():
            p.grad.data *= grad_mask  # type: ignore

        optimizer.step()
        lr_scheduler.step()

        # if i % 100 == 0:
        #     print((icp_loss.item(), trs_loss.item()))

    return {i: p.detach() for i, p in poses.items()}


def fix_poses(dataset: NuScenes, seq, ego_pose, sample_annotation, device="cuda"):
    # if os.path.exists(f"/tmp/{seq}.pt"):
    #     graph, lidar_poses = torch.load(f"/tmp/{seq}.pt", weights_only=False)

    # else:
    # print(f"building pose graph {seq}")
    graph = build_pose_graph(dataset, seq, device=device)

    # print(f"bundle adjustment {seq}")
    lidar_poses = bundle_adjustment(dataset, seq, graph)

    # torch.save((graph, lidar_poses), f"/tmp/{seq}.pt")

    # print(f"interpolating poses {seq}")

    ego2lidar = dataset._calibration(seq, "ego", "LIDAR_TOP")

    lidar_timeline = dataset.timestamps(seq, "LIDAR_TOP")[sorted(lidar_poses.keys())]
    lidar_poses = [
        RigidTransform.from_matrix(p) for _, p in sorted(lidar_poses.items())
    ]
    lidar_interp = {t: ep for t, ep in zip(lidar_timeline, lidar_poses)}

    old_ego_poses = [
        RigidTransform(ep["rotation"], ep["translation"]) for ep in ego_pose
    ]
    old_timeline = np.array([ep["timestamp"] for ep in ego_pose])
    old_interp = {t: ep for t, ep in zip(old_timeline, old_ego_poses)}

    ego_pose_new = []
    for ep in ego_pose:
        t = ep["timestamp"]
        if t not in lidar_interp:
            a, b = lr_bisect(lidar_timeline, t)
            w = (t - lidar_timeline[a]) / (lidar_timeline[b] - lidar_timeline[a])
            lidar_interp[t] = RigidTransform.interpolate(
                lidar_poses[a], lidar_poses[b], w
            )

        p: RigidTransform = lidar_interp[t] @ ego2lidar  # type: ignore

        ego_pose_new.append(
            ep
            | {
                "rotation": p.rotation.quat.tolist(),
                "translation": p.translation.vec.tolist(),
            }
        )

    sample_annotation_new = []
    for sa in sample_annotation:
        t = sa.pop("sample_timestamp")

        obj2world = RigidTransform(sa["rotation"], sa["translation"])
        world2ego_old = old_interp[t].inv()
        lidar2world_new = lidar_interp[t]

        obj2world_new: RigidTransform = (
            lidar2world_new @ ego2lidar @ world2ego_old @ obj2world
        )  # type: ignore

        sample_annotation_new.append(
            sa
            | {
                "rotation": obj2world_new.rotation.quat.tolist(),
                "translation": obj2world_new.translation.vec.tolist(),
            }
        )

    return ego_pose_new, sample_annotation_new


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--root", type=pathlib.Path, required=True, help="dataset root dir"
    )
    argparser.add_argument(
        "--subset", required=True, help="split name (ex: 'v1.0-mini')"
    )
    argparser.add_argument("--out", type=pathlib.Path, required=True, help="output dir")
    argparser.add_argument("--workers", type=int, default=1)
    args = argparser.parse_args()

    dataset = NuScenes(args.root, args.subset)

    sample_seq = {
        sample_t: seq
        for seq in dataset.sequences()
        for sample_t in dataset.sample_tokens(seq)
    }

    with open(dataset.root / dataset.subset / "sample.json", "rb") as f:
        sample = json.load(f)

    sample_dict = {s["token"]: s for s in sample}

    with open(dataset.root / dataset.subset / "ego_pose.json", "rb") as f:
        ego_pose = json.load(f)

    ego_pose_dict = {ep["token"]: ep for ep in ego_pose}

    with open(dataset.root / dataset.subset / "sample_annotation.json", "rb") as f:
        sample_annotation = json.load(f)

    ego_pose_per_scene = [[] for _ in dataset.sequences()]
    with open(dataset.root / dataset.subset / "sample_data.json", "rb") as f:
        for sd in json.load(f):
            seq = sample_seq[sd["sample_token"]]
            ep = ego_pose_dict[sd["ego_pose_token"]]
            ego_pose_per_scene[seq].append(ep)

    # sort sample data by timestamps
    ego_pose_per_scene = [
        sorted(epps, key=lambda p: p["timestamp"]) for epps in ego_pose_per_scene
    ]

    sample_annotation_per_scene = [[] for _ in dataset.sequences()]
    for sa in sample_annotation:
        seq = sample_seq[sa["sample_token"]]
        st = sample_dict[sa["sample_token"]]["timestamp"]
        sample_annotation_per_scene[seq].append(sa | {"sample_timestamp": st})

    ego_pose_new = []
    sample_annotation_new = []
    sequences = dataset.sequences()
    with ThreadPoolExecutor(args.workers) as p:
        for epn, san in tqdm(
            p.map(
                functools.partial(fix_poses, dataset),
                sequences,
                ego_pose_per_scene,
                sample_annotation_per_scene,
            ),
            total=len(sequences),
        ):
            ego_pose_new.extend(epn)
            sample_annotation_new.extend(san)

    # Reorder like the original data to facilitate diffs
    ego_pose_new = {ep["token"]: ep for ep in ego_pose_new}
    ego_pose_new = [ego_pose_new[ep["token"]] for ep in ego_pose]

    sample_annotation_new = {sa["token"]: sa for sa in sample_annotation_new}
    sample_annotation_new = [
        sample_annotation_new[ep["token"]] for ep in sample_annotation
    ]

    # Generate RFC 6902 JSON patch
    ego_pose_patch = [
        {"op": "replace", "path": f"/{i}/{k}", "value": b[k]}
        for i, (a, b) in enumerate(zip(ego_pose_dict, ego_pose_new))
        for k in a.keys()
        if a[k] != b[k]
    ]

    sample_annotation_patch = [
        {"op": "replace", "path": f"/{i}/{k}", "value": b[k]}
        for i, (a, b) in enumerate(zip(sample_annotation, sample_annotation_new))
        for k in a.keys()
        if a[k] != b[k]
    ]

    # Save
    os.makedirs(args.out, exist_ok=True)

    with open(args.out / "ego_pose.json", "w") as f:
        json.dump(ego_pose_new, f)
    with open(args.out / "ego_pose.patch.json", "w") as f:
        json.dump(ego_pose_patch, f)

    with open(args.out / "sample_annotation.json", "w") as f:
        json.dump(sample_annotation_new, f)
    with open(args.out / "sample_annotation.patch.json", "w") as f:
        json.dump(sample_annotation_patch, f)


if __name__ == "__main__":
    main()
