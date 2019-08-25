import numpy as np

colours = [
    np.array([31,119,180]),
    np.array([255,127,14]),
    np.array([44,160,44]),
    np.array([214,39,40]),
    np.array([148,103,189]),
    np.array([140,86,75]),
    np.array([227,119,194]),
    np.array([127, 127, 127]),
    np.array([188, 189, 34]),
    np.array([23, 190, 207])
]

permutations = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0]
]

def get_colour(i):
    return colours[i % 10][permutations[i // 10]]

def is_close(a, b, atol=1e-12):
    return np.abs(a - b) < atol