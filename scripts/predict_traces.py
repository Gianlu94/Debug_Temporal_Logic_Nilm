import argparse
import datetime
from pathlib import Path
import json

import minizinc
import matplotlib.pyplot as plt
import numpy as np

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
    ax.plot(points, active, "C1", label="active")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("power")
    plt.show()


def compute_limits(states, dev_frac):
    states = np.asarray(states)
    unique_states = np.unique(states)
    unique_states.sort()

    list_limits = []
    
    for s in unique_states:
        pos_count = int((states == s).sum())


        dev_int = int(pos_count * dev_frac)

        list_limits.append([max(0, pos_count - dev_int), min(len(states), pos_count + dev_int)])

    return list_limits




def display(params):

    meta, data = load_ukdale_dataset(params.path, params.appliance, "1")
    data = data.squeeze()

    data = data[:500000]

    starts, stops = tlnilm.fitting.activation_windows(data, **tlnilm.fitting._PREPROC[params.appliance])

    lims = np.zeros(data[:, 1].shape)
    lims[starts] = 1
    lims[stops] = -1

    arr_to_fit = data[starts[0]:stops[0], 1]
    arr_to_predict = data[starts[2]:stops[2], 1]

    fit_dict = {"app_powers": list(arr_to_fit.astype(int)), "MAX_STATES": 4}

    solved = tlnilm.fitting.run_optimization(str(Path(__file__).parent.parent /"src" / "minizinc" / "fit_appliance.mzn"),
                                dict_const=fit_dict,
                                solver="or-tools",
                                processes=4)



    dict_sol = vars(solved.solution)

    predict_dict = {"idle_mean": 0, 
                    "pred_powers": arr_to_predict.astype(int).tolist(),
                    "states_limits": compute_limits(solved.solution.states_map, 0.25),
                    **{ k: dict_sol[k] for k  in ["pow_base","pow_multiplier"]}
            }


    if params.dzn is not None:
        with open(params.dzn, "w") as f:
            json.dump(predict_dict,f, indent=4, sort_keys=True) 
        return 

    fwd = tlnilm.fitting.run_optimization(str(Path(__file__).parent.parent /"src" / "minizinc" / "predict_appliance.mzn"),
                                dict_const=predict_dict,
                                solver="or-tools",
                                processes=4)


    fwd = fwd.solution
    pred_curve = tlnilm.fitting.prepare_curve(
        fwd.change_point,
        fwd.states_map,
        dict_sol["pow_multiplier"],
        dict_sol["pow_base"])

    plot_predictions(arr_to_predict,pred_curve)






def main():
    parser = argparse.ArgumentParser(description='Training utility for seq2point')
    parser.add_argument("path", type=str,
                          help='The directory containing the formatted dataset')
    parser.add_argument("appliance", type=str,
                          help='The appliance to target')
    parser.add_argument("-v", "--verbose", action="store_true",
                          help='tensorboard log directory')
    parser.add_argument("-d", "--dzn", default=None, type=str,
                          help='save dzn data file')
    params = parser.parse_args()


    display(params)

if __name__ == '__main__':
    main()














