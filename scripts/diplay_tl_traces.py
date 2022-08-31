import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tlnilm.fitting



def load_ukdale_dataset(path, key, house="1"):
    loaded = np.load(path)

    meta = loaded["arr_0"]

    to_load_ind = np.where((meta[:, 0] == key) & (meta[:, 1] == house))[0]
    to_load_str = np.char.add("arr_", (to_load_ind + 1).astype(str))


    return meta[to_load_ind, :], np.array([loaded[i] for i in to_load_str])


def plot_predictions(gt, active):
    fig, ax = plt.subplots()
    points = np.arange(len(gt))

    ax.plot(points, gt, "C0", label="gt")
    ax.plot(points, active, "C1", label="data", alpha=0.7)
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("power")
    plt.show()



def display(params):

    meta, data = load_ukdale_dataset(params.path, params.appliance, params.house)
    data = data.squeeze()

    if params.limit is not None:
        data = data[:params.limit]

    if params.bare:
        plot_predictions(data[:, 1], data[:, 0])

    else:

        starts, stops = tlnilm.fitting.activation_windows(data, **tlnilm.fitting._PREPROC[params.appliance])

        lims = np.zeros(data[:, 1].shape)
        lims[starts] = 1
        lims[stops] = -1

        plot_predictions(data[:, 1], lims* 100)





def main():
    parser = argparse.ArgumentParser(description='Training utility for seq2point')
    parser.add_argument("path", type=str,
                          help='The directory containing the formatted dataset')
    parser.add_argument("appliance", type=str,
                          help='The appliance to target')
    parser.add_argument("-v", "--verbose", action="store_true",
                          help='tensorboard log directory')
    parser.add_argument("-l", "--limit",default=None, type=int,
                          help='data limit')
    parser.add_argument("-b", "--bare", action="store_true",
                          help='display bare data')
    parser.add_argument("--house",default="1", type=str,
                          help='data limit')
    params = parser.parse_args()


    display(params)

if __name__ == '__main__':
    main()














