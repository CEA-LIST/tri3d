import logging
import weakref
import functools
import inspect
from collections import OrderedDict
from typing import Callable, ParamSpec, TypeVar

import numpy as np


logger = logging.getLogger(".".join(__name__.split(".")[:-1]))


def invert_index(a):
    """Invert an index table.

    :param a:
        An array of positive integer indexes.
    :return:
        An array where the i-th element contains the index of value `i` in `a`,
        if such a value exists.
    """
    a = np.asarray(a)

    assert min(a) >= 0

    size = max(a) + 1
    index = np.full((size,), size, dtype=a.dtype)
    index[a] = np.arange(len(a))

    return index


def dict2lut(d: dict):
    a = np.full([max(d.keys()) + 1 - min(min(d.keys()), 0)], -1, dtype=np.int64)
    for k, v in d.items():
        a[k] = v

    return a


def all_equal(values):
    it = iter(values)
    v0 = next(it)
    return all(v == v0 for v in it)


class interp_nearest:
    def __init__(self, x, y):
        self.x = np.concatenate([np.asarray(x), [np.inf]])
        self.y = np.asarray(y)

    def __call__(self, x):
        x = np.asarray(x)
        i = np.searchsorted(self.x, x)
        left = self.x[np.fmax(0, i - 1)]
        right = self.x[i]
        i -= x - left < right - x
        return self.y[np.fmax(0, i)]


def arg_groupby(indices):
    keys = np.unique(indices)
    return {k: np.argwhere(indices == k)[:, 0] for k in keys}


_memoize_cache = weakref.WeakKeyDictionary()
P = ParamSpec("P")
T = TypeVar("T")


def memoize_method(maxsize=1):
    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        signature = inspect.signature(f)

        @functools.wraps(f)
        def wrapped(self, *args, **kwargs):
            # normalize args into hashable object
            bound_args = signature.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            bound_args = tuple(sorted(bound_args.arguments.items()))

            # create cache entry on first call
            if self not in _memoize_cache:
                _memoize_cache[self] = OrderedDict()

            cache = _memoize_cache[self]

            # return cached result when possible
            if bound_args in cache:
                # move entry to last?
                return cache[bound_args]

            # compute
            value = f(self, *args, **kwargs)

            # cache
            cache[bound_args] = value

            # drop old values
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)

            return value

        return wrapped

    return decorator


def lr_bisect(a, x):
    """return the indices of the surrounding values of x in a.

    TODO: document left of left and right of right cases
    """
    a = np.asarray(a)

    i1 = np.searchsorted(a, x, side="right") - 1
    i1 = i1.clip(min=0, max=max(0, len(a) - 2))
    i2 = (i1 + 1).clip(max=len(a) - 1)

    return i1, i2


def nearest_sorted(a, x):
    i1, i2 = lr_bisect(a, x)
    a1 = a[i1]
    a2 = a[i2]

    return np.where(np.abs(x - a1) < np.abs(a2 - x), i1, i2)
