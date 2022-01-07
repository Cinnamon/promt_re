import numpy as np


def f(a, b):
    """
    row wise product
    :param a: a 2D list or 2D np.array
    :param b: a 1D list or 1D np.array
    :return: a 2D np.array
    """
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    assert a.shape[0] == b.shape[0]
    b = np.expand_dims(b, -1)
    return a * b

# Test
a = [[1, 2],
     [3, 4]]
b = [3, 10]
print(f(a, b))
"""
[[ 3  6]
 [30 40]]
"""
