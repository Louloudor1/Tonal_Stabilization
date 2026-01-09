import numpy as np

from tonal_stabilization.my_module import typed_function


def test_typed_function():
    assert not typed_function(np.zeros(10), "")
