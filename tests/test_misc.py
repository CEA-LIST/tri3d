import numpy as np

from tri3d import misc


def test_memoize_method():
    class A:
        def __init__(self):
            self.calls = 0

        @misc.memoize_method()
        def do(self, a, b, c=0):
            self.calls += 1
            return a, b, c

    a = A()
    b = A()

    assert a.do(1, b=2, c=0) == (1, 2, 0) and a.calls == 1 and b.calls == 0
    assert a.do(1, 2) == (1, 2, 0) and a.calls == 1 and b.calls == 0
    assert a.do(1, b=2) == (1, 2, 0) and a.calls == 1 and b.calls == 0
    assert a.do(1, 3) == (1, 3, 0) and a.calls == 2 and b.calls == 0
    assert a.do(1, b=2) == (1, 2, 0) and a.calls == 3 and b.calls == 0


def test_lr_bisect():
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # in between
    np.testing.assert_allclose(misc.lr_bisect(a, 0.0), [0, 1])
    np.testing.assert_allclose(misc.lr_bisect(a, -1.0), [0, 1])
    np.testing.assert_allclose(misc.lr_bisect(a, 0.5), [0, 1])
    np.testing.assert_allclose(misc.lr_bisect(a, 4.5), [3, 4])
    np.testing.assert_allclose(misc.lr_bisect(a, [0.0, 4.5]), [[0, 3], [1, 4]])
