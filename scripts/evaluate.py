import argparse
import sys
import os
from pathlib import Path
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import optuna
from optuna.samplers import RandomSampler
import minizinc
import matplotlib.pyplot as plt
import numpy as np

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

import tlnilm.fitting
import tlnilm.reduction as red



UNSAT_MZN_COST = sys.maxsize

def load_ukdale_dataset(path, key, house="1"):
    loaded = np.load(path)

    meta = loaded["arr_0"]

    to_load_ind = np.where((meta[:, 0] == key) & (meta[:, 1] == house))[0]
    to_load_str = np.char.add("arr_", (to_load_ind + 1).astype(str))


    return meta[to_load_ind, :], np.array([loaded[i] for i in to_load_str])


def compute_limits(states, dev_frac):
    states = np.asarray(states)
    unique_states = np.unique(states)
    unique_states.sort()

    list_limits = []

    for s in unique_states:
        pos_count = int((states == s).sum())


        dev_int = int(pos_count * dev_frac)

        list_limits.append([max(0, pos_count - dev_int), pos_count + dev_int])

    return list_limits




def train(path, appliance, states, sequences=1, cost_weights=(1,1,1,1,1),
        reduce_to_len=None, solver="gecode", processes=1, figure=None, timeout=None):
    meta, data = load_ukdale_dataset(path, appliance, "1")
    data = data.squeeze() # train set targets for the appliance

    arr_train = tlnilm.fitting.prepare_sequences(data, sequences, tlnilm.fitting._PREPROC[appliance], seed=0)

    reduction = 1
    if reduce_to_len is not None:
        reduction = max(int(np.ceil(len(arr_train[0]) / reduce_to_len)), 1)

    arr_to_fit = [red.seq_reduce(s, reduction).tolist() for s in  arr_train]

    param_dict = {"app_powers": arr_to_fit,
                  "SEQUENCES": len(arr_to_fit),
                  "SEQUENCE_LENGHT": len(arr_to_fit[0]),
                  "MAX_STATES": states,
                  "cost_weights": cost_weights,
                "pow_values": red.seq_reduce(red.power_values(2, np.max(arr_to_fit), len(arr_train[0])), reduction).tolist()

            }

    fail_reason = None
    result = None
    status = None
    try:
        status, result = tlnilm.fitting.run_optimization(str(Path(__file__).parent.parent /"src" / "minizinc" / "fit_appliance.mzn"),
                                    dict_const=param_dict,
                                    solver=solver,
                                    processes=processes,
                                    timeout=timeout)
    except minizinc.error.MiniZincError as e:
        fail_reason = str(e)


    if (status == 'UNSATISFIABLE'):
        fail_reason = "Unsatisfable fit problem"

    if fail_reason is not None:
        dump_file = "debug.json"
        with open(dump_file, "w") as f:
            json.dump(param_dict,f, indent=4, sort_keys=True)
            return status
            #raise ValueError(fail_reason + f" dzn dumped in {dump_file}")

    dict_sol = result

    dict_sol_exp = {**dict_sol, "counter_limits": (np.array(dict_sol["counter_limits"]) * reduction).tolist()}

    if figure is not None:
        curves = get_curves(
                    [(l + h)//2 for l,h in   dict_sol_exp["pow_multiplier_limits"]],
                    [(l + h)//2 for l,h in   dict_sol_exp["pow_base_limits"]],
                    len(arr_train[0]),
                    0)

        pred_curve = np.concatenate([[0]] + [ curves[e][:(l +h) //2] for e, (l, h) in enumerate(dict_sol_exp["pow_multiplier_limits"], start=1)] + [[0]])
        plot_fittings(figure, pred_curve)

    return dict_sol_exp


def predict_event(mains_powers, pred_powers, states_limits, pow_base_limits,
                  pow_multiplier_limits, solver, processes,
                  reduce_to_len, idle_val, debug=False, deviation=0, seq_w=(1,1), no_event_w=1, timeout=None):

    pow_multiplier = np.array(pow_multiplier_limits).mean(axis=1).astype(int).tolist()


    reduction = 1
    if reduce_to_len is not None:
        reduction = max(int(np.ceil(len(mains_powers)/ reduce_to_len)), 1)


    predict_dict = {"idle_mean": idle_val,
                    "pred_powers": red.seq_reduce(pred_powers, reduction).tolist(),
                    "mains_powers": red.seq_reduce(mains_powers, reduction).tolist(),
                    "seq_weights" : seq_w,
                    "no_ev_cost" : no_event_w,
                    "states_limits": deviate_limits(red.range_reduce(states_limits, reduction), deviation),
                    "pow_base_limits" : deviate_limits(pow_base_limits, deviation),
                    "pow_multiplier" : pow_multiplier,
                    "baseline": int(np.mean(mains_powers[:5] + mains_powers[-5:])),
                    "pow_values": red.seq_reduce(red.power_values(2, max(pred_powers), len(pred_powers)),reduction).tolist(),

}

    fail_reason = None
    result = None
    status = None
    try:
        status, result = tlnilm.fitting.run_optimization(str(Path(__file__).parent.parent /"src" / "minizinc" / "predict_appliance.mzn"),
                                    dict_const=predict_dict,
                                    solver=solver,
                                    processes=processes,
                                    timeout=timeout
                                    )
    except minizinc.error.MiniZincError as e:
        fail_reason = str(e)

    if  (status== 'UNSATISFIABLE'):
        fail_reason = "Unsatisfable fit problem"


    debug  = debug or (fail_reason is not None)

    if debug:
        dump_file = "debug.json"
        with open(dump_file, "w") as f:
            json.dump(predict_dict,f, indent=4, sort_keys=True)

        if fail_reason is not None:
            return status
            #raise ValueError(f"Unsatisfable prediction problem, dzn dumped in {dump_file}")

    if result["no_event"]:
        pred_curve = np.zeros(len(pred_powers), dtype=int)

    else:
        expanded_state_map = red.seq_expand(result["states_map"], reduction, len(pred_powers))

        curves = get_curves(pow_multiplier,result["pow_base"],len(pred_powers), idle_val)

        pred_curve = tlnilm.fitting.prepare_curve(
        curves,
        red.compute_change_points(expanded_state_map),
        expanded_state_map)

    return pred_curve


def get_curves(pow_multiplier, pow_base, length, idle_val):

    idle_curve = np.tile(idle_val,length).tolist()
    pow_curves = [list(tlnilm.fitting.pow_fn(2, pow_multiplier[i], pow_base[i],length))
                 for i in range(len(pow_multiplier))]

    return [idle_curve] + pow_curves + [idle_curve]


def gen_valid_intervals(starts, stops, skipsize):
    for b,e in zip(starts, stops):
        if e - b < skipsize:
            yield b,e


def predict(
        dict_sol, data, appliance, skipsize, idle_val, reduce_to_len=None, solver="gecode", processes=1,
        solver_processes=8, figure=None, debug=False, deviation=0, seq_w=(1,1), no_event_w=1):

    starts, stops = tlnilm.fitting.activation_windows(data, **tlnilm.fitting._PREPROC[appliance])


    arr_optimized = data[:, 1].copy()

    pool = Parallel(n_jobs=processes)

    list_ret = pool(delayed(predict_event)(data[b:e, 0].astype(int).tolist(),
                                           data[b:e, 1].astype(int).tolist(),
                                       dict_sol["counter_limits"], dict_sol["pow_base_limits"],
                                        dict_sol["pow_multiplier_limits"], solver, solver_processes,
                                        reduce_to_len, idle_val, debug=debug, deviation=deviation, seq_w=seq_w, no_event_w=no_event_w)
                    for b,e in tqdm(gen_valid_intervals(starts, stops, skipsize),
                                    total=starts.size, smoothing=0)
                    )

    if list_ret == 'UNSATISFIABLE': return list_ret
    
    for (b,e), a in zip(gen_valid_intervals(starts, stops, skipsize),list_ret):
        arr_optimized[b:e] = a

    if figure is not None:
        plot_predictions(figure, data[:, 2], data[:, 1], arr_optimized, data[:, 0],starts, stops)

    return arr_optimized

def _set_hyperparameter(params, name, type_trial, range_start, range_end,  step=1):
    params[name] = type_trial(name, range_start, range_end, step=step)
    

def evaluate_objective(trial, params):
    fig, (ax1, ax2) = plt.subplots(2)
    
    # hyperparameters
    _set_hyperparameter(
        params, "sequences", trial.suggest_int, params["sequences_search"][0], params["sequences_search"][1]
    )

    _set_hyperparameter(
        params, "deviation", trial.suggest_float, params["deviation_search"][0], params["deviation_search"][1]
    )

    _set_hyperparameter(
        params, "skip_size", trial.suggest_int, params["skip_size_search"][0], params["skip_size_search"][1],
        params["skip_size_search"][2]
    )

    _set_hyperparameter(
        params, "train_reduction", trial.suggest_int, params["train_reduction_search"][0], params["train_reduction_search"][1],
        params["train_reduction_search"][2]
    )

    _set_hyperparameter(
        params, "predict_reduction", trial.suggest_int, params["predict_reduction_search"][0], params["predict_reduction_search"][1],
        params["predict_reduction_search"][2]
    )
    
    hps_cost_weights = list(params["cost_weights"])
    hps_seq_weights = list(params["seq_weights"])
    
    for i in range(len(params["cost_weights"])):
        weight_str = "cw{}".format(i)
        range_start, range_end = params["cost_weights_search"][weight_str][0], params["cost_weights_search"][weight_str][1]
        hps_cost_weights[i] = trial.suggest_int(weight_str, range_start, range_end)
    
    for i in range(len(params["seq_weights"])):
        weight_str = "sw{}".format(i)
        range_start, range_end = params["seq_weights_search"][weight_str][0], params["seq_weights_search"][weight_str][1]
        hps_seq_weights[i] = trial.suggest_int(weight_str, range_start, range_end)
        
    params["cost_weights"] = tuple(hps_cost_weights)
    params["hps_weights"] = tuple(hps_seq_weights)
    _set_hyperparameter(
        params, "no_event_weight", trial.suggest_int, params["no_event_weight_search"][0], params["no_event_weight_search"][1],
    )
    
    # training and validation
    dict_sol = train(params["path"], params["appliance"], params["fit_states"],
                     reduce_to_len=params["train_reduction"], sequences=params["sequences"], cost_weights=params["cost_weights"],
                     solver=params["fit_solver"], processes=params["fit_processes"], figure=ax1)
    
    if dict_sol == 'UNSATISFIABLE': return UNSAT_MZN_COST
    
    test_data = np.load(params["predictions"])["arr_0"].squeeze().T[
                params["start"]:]  # array [(nn output,nn input, targets)].T
    if params["limit"] is not None:
        test_data = test_data[:params["limit"]]
    arr_pred = predict(dict_sol, test_data, params["appliance"],
                                   params["skip_size"], params["idle_val"],
                                   solver=params["predict_solver"], processes=params["processes"], figure=ax2,
                                   debug=params["debug"], reduce_to_len=params["predict_reduction"],
                                   deviation=params["deviation"], seq_w=params["seq_weights"], no_event_w=params["no_event_weight"])

    if arr_pred == 'UNSATISFIABLE': return UNSAT_MZN_COST

    mae_nn = np.abs(test_data[:, 1] - test_data[:, 2]).mean()
    mae_minizinc = np.abs(arr_pred - test_data[:, 2]).mean()
    
    # normalised signal aggregate error (SAE)
    
    # skip last window if window_size < t
    # t = 86400 #1v = 6s -> 6*10*60*24 = 86400 (1 day)
    # n_days = test_data.shape[0] // t
    #
    # r_gt_days = [test_data[(d * t):((d+1) * t), 2] for d in range(0, n_days)]
    # r_nn_days = [test_data[(d * t):((d+1) * t), 1] for d in range(0, n_days)]
    # r_mnz_days = [arr_pred[(d * t):((d+1) * t)] for d in range(0, n_days)]
    #
    # sae_nn = np.mean([(np.abs(np.sum(r_nn) - np.sum(r_gt)) / np.sum(r_gt)) for r_nn, r_gt in zip(r_nn_days, r_gt_days)])
    # sae_minizinc = np.mean([(np.abs(np.sum(r_mnz) - np.sum(r_gt)) / np.sum(r_gt)) for r_mnz, r_gt in zip(r_mnz_days, r_gt_days)])
    
    # include last window even if window_size < t
    t = 86400
    tot_t = test_data.shape[0]
    
    r_gt_days = [test_data[s:(s + t), 2] for s in range(0, tot_t, t)]
    r_nn_days = [test_data[s:(s + t), 1] for s in range(0, tot_t, t)]
    r_mnz_days = [arr_pred[s:(s + t)] for s in range(0, tot_t, t)]
    
    # take average of SAE over days
    sae_nn = np.mean([np.abs(np.sum(r_nn) - np.sum(r_gt)) for r_nn, r_gt in zip(r_nn_days, r_gt_days)])
    sae_minizinc = np.mean([np.abs(np.sum(r_mnz) - np.sum(r_gt)) for r_mnz, r_gt in zip(r_mnz_days, r_gt_days)])
    
    if not params["csv"]:
        print("nn MAE:\t", mae_nn)
        print("op MAE:\t", mae_minizinc)
        print("nn SAE:\t", sae_nn)
        print("op SAE:\t", sae_minizinc)
    else:
        print(params["appliance"], ",", mae_nn, ",", mae_minizinc, ",", sae_nn, ",", sae_minizinc, sep="")
    
    if params["graphical"]:
        plt.show()
        
    plt.close()
    
    return mae_minizinc

def evaluate(params):

    fig, (ax1, ax2) = plt.subplots(2)

    dict_sol = train(params["path"], params["appliance"], params["fit_states"],
                    reduce_to_len=params["train_reduction"], sequences=params["sequences"],
                     solver=params["fit_solver"], processes=params["fit_processes"], figure=ax1)

    if dict_sol == 'UNSATISFIABLE': return UNSAT_MZN_COST
    
    test_data = np.load(params["predictions"])["arr_0"].squeeze().T[params["start"]:] # array [(nn output,nn input, targets)].T
    if params["limit"] is not None:
        test_data = test_data[:params["limit"]]
    arr_pred = predict(dict_sol, test_data, params["appliance"],
                       params["skip_size"], params["idle_val"],
                       solver=params["predict_solver"], processes=params["processes"], figure=ax2,
                       debug=params["debug"], reduce_to_len=params["predict_reduction"],
                       deviation=params["deviation"],seq_w=params["seq_weights"])

    if arr_pred == 'UNSATISFIABLE': return UNSAT_MZN_COST
    
    mae_nn = np.abs(test_data[:, 1] - test_data[:, 2]).mean()
    mae_minizinc = np.abs(arr_pred - test_data[:, 2]).mean()

    # normalised signal aggregate error (SAE)
    
    # skip last window if window_size < t
    # t = 86400 #1v = 6s -> 6*10*60*24 = 86400 (1 day)
    # n_days = test_data.shape[0] // t
    #
    # r_gt_days = [test_data[(d * t):((d+1) * t), 2] for d in range(0, n_days)]
    # r_nn_days = [test_data[(d * t):((d+1) * t), 1] for d in range(0, n_days)]
    # r_mnz_days = [arr_pred[(d * t):((d+1) * t)] for d in range(0, n_days)]
    #
    # sae_nn = np.mean([(np.abs(np.sum(r_nn) - np.sum(r_gt)) / np.sum(r_gt)) for r_nn, r_gt in zip(r_nn_days, r_gt_days)])
    # sae_minizinc = np.mean([(np.abs(np.sum(r_mnz) - np.sum(r_gt)) / np.sum(r_gt)) for r_mnz, r_gt in zip(r_mnz_days, r_gt_days)])

    # include last window even if window_size < t
    t = 86400
    tot_t = test_data.shape[0]

    r_gt_days = [test_data[s:(s + t), 2] for s in range(0, tot_t, t)]
    r_nn_days = [test_data[s:(s + t), 1] for s in range(0, tot_t, t)]
    r_mnz_days = [arr_pred[s:(s + t)] for s in range(0, tot_t, t)]

    # take average of SAE over days
    sae_nn = np.mean([np.abs(np.sum(r_nn) - np.sum(r_gt)) for r_nn, r_gt in zip(r_nn_days, r_gt_days)])
    sae_minizinc = np.mean([np.abs(np.sum(r_mnz) - np.sum(r_gt)) for r_mnz, r_gt in zip(r_mnz_days, r_gt_days)])

    if not params["csv"]:
        print("nn MAE:\t", mae_nn)
        print("op MAE:\t", mae_minizinc)
        print("nn SAE:\t", sae_nn)
        print("op SAE:\t", sae_minizinc)
    else:
        print(params["appliance"], ",", mae_nn, ",", mae_minizinc, ",", sae_nn, ",", sae_minizinc, sep="")

    if params["graphical"]:
        plt.show()
    
    plt.close()


def deviate_limits(values, deviation):
    ret = []
    for vl, vu in values:
        val_dev = int((vu - vl) * deviation)
        ret.append([max(0, vl - val_dev),vu + val_dev])

    return ret


def plot_fittings(ax, curve):
    points = np.arange(len(curve))

    ax.plot(points, curve, "C0", label="gt")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("power")


def plot_predictions(ax, gt, nn, mzn, mains, starts, stops):
    points = np.arange(len(gt))

    ax.plot(points, gt, "C0", label="gt")
    ax.plot(points, nn, "C1", label="nn output")
    ax.plot(points, mzn, "C2", label="mzn output")
    ax.plot(points, mains, "C3", label="mains")
    ax.vlines(starts, 0, max(gt) ,"C9", label="start")
    ax.vlines(stops, 0, max(gt) ,"C8", label="stops")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("power")


def set_best_hps(params, best_hps):
    cost_weights = [cw for cw in list(params["cost_weights"])]
    seq_weights = [sw for sw in list(params["seq_weights"])]
    
    # set entries to params to the best hps found
    print("Best Hyperparameters: \n")
    
    for hp, value in best_hps.items():
        if "cw" in hp:
            pos = int(hp[-1])
            cost_weights[pos] = value
        elif "sw" in hp:
            pos = int(hp[-1])
            seq_weights[pos] = value
        else:
            params[hp] = value
            print("     {} = {}".format(hp, str(value)))
            
    params["cost_weights"] = tuple(cost_weights)
    params["seq_weights"] = tuple(seq_weights)
    print("     cost_weights = {}".format(str(cost_weights)))
    print("     seq_weights = {}".format(str(seq_weights)))
    

def main():
    parser = argparse.ArgumentParser(description='Training utility for seq2point')
    parser.add_argument("path", type=str,
                          help='The directory containing the formatted dataset')
    parser.add_argument("appliance", type=str,
                          help='The appliance to target')
    parser.add_argument("predictions", type=str,
                          help='file containing the predictions')
    parser.add_argument("-v", "--verbose", action="store_true",
                          help='tensorboard log directory')
    parser.add_argument("--csv", action="store_true",
                          help='print data in csv format')
    parser.add_argument("--start", type=int, default=0,
                          help='stop')
    parser.add_argument("--limit", type=int, default=None,
                          help='stop')
    parser.add_argument("-f", "--fail", action="store_true",
                          help='trigger prediction error')
    parser.add_argument("--skip-size", type=int, default=np.inf,
                          help='skip traces larger than')
    parser.add_argument("-p", "--processes", type=int, default=1,
                          help='number of processes to use')
    parser.add_argument("--fit-processes", type=int, default=1,
                          help='number of fit processes to use')
    parser.add_argument("--fit-solver", type=str, default="gecode",
                          help='solver to use for fitting')
    parser.add_argument("--predict-solver", type=str, default="gecode",
                          help='solver to use for prediction')
    parser.add_argument("--train-reduction", type=int, default=None,
                          help='reduce sequences ou to this len for training')
    parser.add_argument("--predict-reduction", type=int, default=None,
                          help='reduce sequences ou to this len for prediction')
    parser.add_argument("--train-timeout", type=int, default=None,
                          help='time limit for training problem')
    parser.add_argument("--predict-timeout", type=int, default=None,
                          help='time limit for prediction problem')
    parser.add_argument("--graphical", action="store_true",
                          help='display plots')
    parser.add_argument("--fit-states", type=int, default=3,
                          help='number of states for fitting')
    parser.add_argument("--idle-val", type=int, default=15,
                          help='idle power consumption')
    parser.add_argument("-s", "--sequences", type=int, default=1,
                          help='number of sequences to use')
    parser.add_argument("--debug", action="store_true",
                          help='tensorboard log directory')
    parser.add_argument("--path-hps-file", type=str,
                          help='The path to the hyperparameters selection file')
    parser.add_argument("--deviation", type=float, default=0,
                          help='deviation for "pow_base" and "states_limits" predict parametes')
    parser.add_argument("--cost-weights", type=int, default=(1, 1, 1, 1, 1), nargs=5,
                        help='weights to balance each component of fit_appliance cost function')
    parser.add_argument("--seq-weights", type=int, default=(1,1),nargs=2,
                          help='weights to balance nn outputs/mains weight in prediction')
    parser.add_argument("--no-event-weight", type=int, default=2, nargs=1,
                        help='weight for no event in prediction')
    parser.add_argument("--study-to-eval", type=str,
                        help='The optuna study from which get the best hps <name-number>')
    params = vars(parser.parse_args())

    if params["path_hps_file"]:
        dict_cfgfile = json.loads(Path(params["path_hps_file"]).read_text())

        params = {**params, **dict_cfgfile}
        
        study_save_path = "hps/{}/".format(params["study_name"])
        os.makedirs(study_save_path, exist_ok=True)
        study_app = params["study_name"] + "-{}".format(params["appliance"])
        study = optuna.create_study(study_name=study_app, sampler=RandomSampler(seed=params["seed"]),
                                    storage="sqlite:///{}{}.db".format(study_save_path, study_app), load_if_exists=True)
        study.optimize(lambda trial: evaluate_objective(trial, params), n_trials=params["trials"])
        
        #joblib.dump(study, "{}{}.pkl".format(study_save_path, study_name))
    else:
        if params["study_to_eval"]:
            study_name = params["study_to_eval"]+"-{}".format(params["appliance"])
            path_to_study = "sqlite:///hps/{}/{}.db".format(params["study_to_eval"], study_name)
            print("Loading hyperparameters from: {}".format(path_to_study))
            study = optuna.create_study(study_name=study_name, storage=path_to_study, load_if_exists=True)
            set_best_hps(params, study.best_trial.params)
            
        evaluate(params)

if __name__ == '__main__':
    main()














