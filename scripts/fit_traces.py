import argparse
import datetime
from pathlib import Path

import minizinc
import matplotlib.pyplot as plt
import numpy as np

import json
import tlnilm.fitting
import tlnilm.reduction as red



def load_ukdale_dataset(path, key, house="1"):
    loaded = np.load(path)

    meta = loaded["arr_0"]

    to_load_ind = np.where((meta[:, 0] == key) & (meta[:, 1] == house))[0]
    to_load_str = np.char.add("arr_", (to_load_ind + 1).astype(str))


    return meta[to_load_ind, :], np.array([loaded[i] for i in to_load_str])


def plot_predictions(ax, gt, active):

    points = np.arange(len(gt))
    ax.plot(points, gt, "C0", label="app")
    ax.plot(points, active, "C1", label="model")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("power")





def display(params):

    meta, data = load_ukdale_dataset(params.path, params.appliance, "1")
    data = data.squeeze()

    data = data[:500000]

    reduction = params.reduction

    arr_train = tlnilm.fitting.prepare_sequences(data, params.sequences, tlnilm.fitting._PREPROC[params.appliance])

    arr_to_fit = [red.seq_reduce(s, reduction).tolist() for s in  arr_train]

    param_dict = {"app_powers": arr_to_fit,
                  "SEQUENCES": len(arr_to_fit),
                  "SEQUENCE_LENGHT": len(arr_to_fit[0]),
                  "MAX_STATES": params.fit_states,
                "pow_values": red.seq_reduce(red.power_values(2, np.max(arr_to_fit), len(arr_train[0])), reduction).tolist()
                 }

    if params.debug:
        dump_file = "debug.json"
        with open(dump_file, "w") as f:
            json.dump(param_dict,f, indent=4, sort_keys=True)


    solved = tlnilm.fitting.run_optimization(str(Path(__file__).parent.parent /"src" / "minizinc" / "fit_appliance.mzn"),
                                dict_const=param_dict,
                                solver="or-tools",
                                processes=8)

    sol = solved.solution
#    print(sol)


    fig, axs = plt.subplots(params.sequences)

    for e, ax in enumerate(axs):
        pred_curve = compute_predicted_curve(sol.states_map[e], reduction,
                len(arr_train[0]), sol.pow_multiplier[e],sol.pow_base[e])

        plot_predictions(ax, arr_train[e],pred_curve)
    plt.show()



def compute_predicted_curve(states_map, reduction, full_len, pow_multiplier, pow_base ):
    expanded_state_map = red.seq_expand(states_map, reduction, full_len)

    curves = tlnilm.fitting.get_pow_curves(pow_multiplier,pow_base,full_len, 0)

    pred_curve = tlnilm.fitting.prepare_curve(
    curves,
    red.compute_change_points(expanded_state_map),
    expanded_state_map)

    return pred_curve
    

def main():
    parser = argparse.ArgumentParser(description='Training utility for seq2point')
    parser.add_argument("path", type=str,
                          help='The directory containing the formatted dataset')
    parser.add_argument("appliance", type=str,
                          help='The appliance to target')
    parser.add_argument("-s", "--sequences", type=int, default=1,
                          help='number of sequences to use')
    parser.add_argument("-r", "--reduction", type=int, default=1,
                          help='reduction factor to use')
    parser.add_argument("--fit-states", type=int, default=3,
                          help='number of states for fitting')
    parser.add_argument("-v", "--verbose", action="store_true",
                          help='tensorboard log directory')
    parser.add_argument("-d", "--debug", action="store_true",
                          help='tensorboard log directory')
    params = parser.parse_args()


    display(params)

if __name__ == '__main__':
    main()














