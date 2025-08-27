import numbers
from typing import List, Tuple

import numpy as np

from . import geometry

bbox_vertices = np.array(
    [
        (-0.5, -0.5, -0.5),
        (+0.5, -0.5, -0.5),
        (+0.5, +0.5, -0.5),
        (-0.5, +0.5, -0.5),
        (-0.5, -0.5, +0.5),
        (+0.5, -0.5, +0.5),
        (+0.5, +0.5, +0.5),
        (-0.5, +0.5, +0.5),
        (+0.5, +0.0, -0.5),
        (+0.0, +0.0, -0.5),
    ]
)
bbox_edges = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [8, 9],
    ]
)


try:
    import k3d
except ImportError:
    pass
else:

    def plot_bboxes_3d(
        plot: k3d.Plot,
        transforms: List[geometry.Transformation],
        sizes: np.ndarray,
        c: int | Tuple | np.ndarray | None = None,
        *kargs,
        **kwargs,
    ):
        """Add 3D boxes to a k3d plot.

        :param plot: k3d plot
        :param transform: N object local to scene coordinate
            transformations.
        :param sizes: N by 3 array of object sizes
        :param c: edge colors, either as a single or per box value.
        """
        if isinstance(c, int):
            c = np.full([len(sizes)], c, dtype=np.uint32)
        elif isinstance(c, tuple):
            c = np.tile(to_k3d_colors(c), (len(sizes), 1))
        elif c is None:
            c = np.full([len(sizes)], 0x0000FF00, dtype=np.uint32)
        else:
            c = np.asarray(c)
            if c.ndim == 2:
                c = to_k3d_colors(c)

        vertices = [np.empty([0, 3])]
        edges = [np.empty([0, 2], dtype=int)]

        for t, s in zip(transforms, sizes):
            vertices.append(t.apply(bbox_vertices * s))
            edges.append(bbox_edges + len(edges) * len(bbox_vertices))

        vertices = np.concatenate(vertices)
        edges = np.concatenate(edges)
        colors = c.repeat(len(bbox_vertices))

        plot += k3d.lines(
            vertices.astype(np.float32),
            edges,
            indices_type="segment",
            colors=colors,
            *kargs,
            **kwargs,
        )


try:
    from matplotlib import colors as mcolors
    from matplotlib import patches as mpatches
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
except ImportError:
    pass
else:

    def gen_discrete_cmap(n):
        """Return a matplotlib discrete colormap for n classes."""
        continous_cmap = plt.get_cmap("nipy_spectral")
        colors = continous_cmap(np.linspace(0, 1, n + 2)[1:-1])
        return mcolors.ListedColormap(colors, name="from_list", N=None)

    def to_k3d_colors(colors):
        """Convert RGB color triplets into int32 values."""
        colors = np.asarray(colors)
        if not issubclass(colors.dtype.type, numbers.Integral):
            colors = np.floor(colors * 255)

        colors = colors.astype(np.uint32)

        return (
            (colors[..., 0] << np.uint32(16))
            | (colors[..., 1] << np.uint32(8))
            | colors[..., 2]
        )

    def plot_bbox_cam(
        transform: geometry.Transformation,
        size: np.ndarray,
        img_size: Tuple[int, int],
        ax=None,
        **kwargs,
    ):
        """Add a 3D box to a matplotlib plot.

        :param transform: transformation from box local to image coordinates
        :param size: object size (length, width height)
        :param img_size: plot size (width, height)
        :param kwargs: arguments forwarded to :func:`LineCollection`
        """
        if ax is None:
            ax = plt.gca()

        pts_2d = transform.apply(bbox_vertices * size)
        invisible = all(
            (pts_2d[:, 2] < 0)
            | (pts_2d[:, 0] < 0)
            | (pts_2d[:, 0] > img_size[0])
            | (pts_2d[:, 1] < 0)
            | (pts_2d[:, 1] > img_size[1])
        )
        if not invisible:
            lc = LineCollection(pts_2d[bbox_edges, :2], **kwargs)
            ax.add_collection(lc)

    def plot_rectangles(centers, sizes, ax=None, **kwargs):
        ax = ax or plt.gca()
        n = len(centers)

        kwargs = {
            k: (v if isinstance(v, list) and len(v) == n else [v] * n)
            for k, v in kwargs.items()
        }

        for i in range(n):
            ax.add_patch(
                mpatches.Rectangle(
                    centers[i], *sizes[i], **{k: v[i] for k, v in kwargs.items()}
                )
            )

    def plot_annotations_bev(center, size, heading, c=None, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if c is None or isinstance(c, tuple) or isinstance(c, str):
            c = [c] * len(center)

        for p_, (l, w, _), h, c_ in zip(center, size, heading, c):
            x = p_[0] - l / 2 * np.cos(h) + w / 2 * np.sin(h)
            y = p_[1] - l / 2 * np.sin(h) - w / 2 * np.cos(h)
            r = mpatches.Rectangle(
                xy=(x, y),
                width=l,
                height=w,
                angle=h / np.pi * 180,
                fill=False,
                color=c_,
                **kwargs,
            )
            ax.add_patch(r)

            a = mpatches.FancyArrow(
                p_[0],
                p_[1],
                l / 2 * np.cos(h),
                l / 2 * np.sin(h),
                width=0,
                head_width=0.3,
                fill=True,
                color=c_,
                length_includes_head=True,
                **kwargs,
            )
            ax.add_patch(a)


def auto_range(x, percentile=0.1, nticks=7, start=None, stop=None):
    """Propose histogram bins (start, stop, step) which best describe data.

    :param x: a list of scalar values
    :param percentile: *percentage* of edge values that can be ignored,
        or two values to distinguish left and right margins
    :param nticks: number of ticks to aim for
    :param start: overrides the start value from automatic range
    :param stop: overrides the stop value from automatic range
    """
    if len(x) < 2:
        return 0, 1, 0.1

    if not hasattr(percentile, "__len__"):
        percentile = percentile, 100 - percentile

    xmin, xmax = np.percentile(x, percentile)
    if np.isnan(xmin):
        xmin = np.min(x)
    if np.isnan(xmax):
        xmax = np.max(x)
    if xmax - xmin < 1e-7:  # single mode collapse -> revert to min/max range
        xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin < 1e-7:  # single mode collapse -> revert to min/max range
        return 0, 1, 0.1

    magnitude = 10 ** (np.floor(np.log(xmax - xmin) / np.log(10) - 1))
    steps = np.array([1, 2, 5, 10]) * magnitude
    xmin = np.floor(xmin / steps) * steps if start is None else np.full(4, start)
    xmax = np.ceil(xmax / steps) * steps if stop is None else np.full(4, stop)
    candidate_nticks = (xmax - xmin) / steps + 1
    best = np.argmin(np.abs(nticks - candidate_nticks))
    return xmin[best], xmax[best], steps[best]
