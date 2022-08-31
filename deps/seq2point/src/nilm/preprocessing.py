import numpy as np
from numpy.random import RandomState

import tensorflow as tf



def normalize_labels(dataset, mean, std):
    return dataset.map(lambda X, y: (X, (y - mean) / std) )

def train_test_split_dataset(dataset, elements):

    ds_test = dataset.take(elements)
    ds_train = dataset.skip(elements)

    return ds_train, ds_test


def concatenate_datasets(list_datasets):

    ds_cat = list_datasets[0]

    for ds in list_datasets[1:]:
        ds_cat = ds_cat.concatenate(ds)

    return ds_cat


def make_window_dataset(arr_data, size_seq, seed, shuffle=True):
    assert size_seq % 2 == 1

    size_skip = (size_seq - 1)  // 2

    X = arr_data[:,0]
    y = arr_data[:,1][size_skip:]


    ds_windows = tf.keras.utils.timeseries_dataset_from_array(X, y, size_seq,
            batch_size=None, shuffle=shuffle, seed=seed)

    return ds_windows


def load_ukdale_dataset(path, key):
    loaded = np.load(path)

    meta = loaded["arr_0"]

    to_load_ind = np.where((meta[:, 0] == key))[0]
    to_load_str = np.char.add("arr_", (to_load_ind + 1).astype(str))


    return meta[to_load_ind, :], np.array([loaded[i] for i in to_load_str])


def split_ukdale_train_test(meta, datasets, train_houses=None):
    test_indices = (meta[:, 1] ==  "2").nonzero()[0]
    if train_houses is None:
        train_indices = (meta[:, 1] !=  "2").nonzero()[0]
    else:
        train_indices = np.isin(meta[:, 1], train_houses).nonzero()[0]


    train_d = np.concatenate([datasets[i] for i in train_indices])
    test_d = np.concatenate([datasets[i] for i in test_indices])
    return train_d, test_d


def train_val_split(arr_data, fraction_val):
    assert fraction_val >= 0 and  fraction_val <= 1
    cutoff_index = 0
    if fraction_val > 0:
        cutoff_index = int(len(arr_data) * fraction_val)

    return arr_data[cutoff_index:], arr_data[:cutoff_index]


def transform_ukdale_to_chain2(dataset):
    mains = dataset[:, 0]

    binned = mains.astype(np.uint16) // 300

    change = np.ones(mains.shape, dtype=np.bool)
    change[1:] = ~ (binned[:-1] == binned[1:])

    idx = np.where(change,np.arange(change.shape[0]),0)
    np.maximum.accumulate(idx,axis=0, out=idx)
    output = mains[idx]

    return np.array([output, dataset[:, 1]]).T


def compute_nomralization_factors(arr_data):
    return arr_data.mean(), arr_data.std()
