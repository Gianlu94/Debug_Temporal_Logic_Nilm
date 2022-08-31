import numpy as np


def power_values(base, max_power, seq_len):
    return np.minimum(float(base) ** np.arange(1, seq_len +1 ), max_power + 1).astype(int).tolist()


def seq_reduce(sequence, size_reduce):
    remainder = (len(sequence) % size_reduce)
    padval = size_reduce - remainder if remainder > 0 else 0
    arr_pad = np.pad(sequence,(0, padval), mode="edge")
    return arr_pad.reshape((-1, size_reduce)).mean(axis=1).astype(arr_pad.dtype)


def range_reduce(ranges, size_reduce):
    return [ [b // size_reduce, e // size_reduce] for b, e in ranges]


def seq_expand(sequence, size_reduce, original_size):
    return np.asarray(sequence).repeat(size_reduce)[:original_size].tolist()


def compute_change_points(sequence):
    return [np.where(sequence == e)[0][0]  for e in np.unique(sequence)]


