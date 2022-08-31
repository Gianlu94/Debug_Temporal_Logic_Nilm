import datetime

import minizinc
import numpy as np
import pandas as pd
from joblib import Memory

_PREPROC = {
    "kettle" : {
                "interval": (500, 3000),
                "standby_power": 5,
                "standby_min_time": 5
    },
    "fridge" : {
                "interval": (30, 500),
                "standby_power": 20,
                "standby_min_time": 5
    },
    "dishwasher" : {
                "interval": (10, 3000),
                "standby_power": 50,
                "standby_min_time": 50
    },
    "washingmachine" : {
                "interval": (50, 3000),
                "standby_power": 5,
                "standby_min_time": 50
    },
    "microwave" : {
                "interval": (500, 2000),
                "standby_power": 10,
                "standby_min_time": 5
    },
}


_CACHE = Memory(location="joblib-cache", verbose=0)

def activation_windows(data, interval, standby_power, standby_min_time):
    data = data.copy()
    data[:, 1] =  np.clip(data[:, 1], None, interval[1])
    data[:, 1] = np.where(data[:, 1] < interval[0], 0, data[:, 1])

    activations = np.where(data[:, 1] > standby_power, 1.0, np.nan)
    filled = pd.DataFrame(activations, columns=["data"]).fillna(method="ffill", limit=standby_min_time).fillna(method="bfill", limit=standby_min_time).fillna(0).to_numpy().squeeze()

    startstop = np.append([0],np.diff(filled))

    starts = np.where(startstop > 0)[0]
    stops = np.where(startstop < 0)[0]

    assert (starts.size - stops.size) in (-1,0,1)

    if starts[0] > stops[0]:
        stops = stops[1:]

    if starts[-1] > stops[-1]:
        starts = starts[:-1]

    assert starts.size == stops.size

    return starts, stops

def gen_activations(data, starts, stops):
    for b, e in zip(starts, stops):
        yield data[b:e]

@_CACHE.cache
def run_optimization(mzn_file, dict_const={}, solver="gecode", timeout=None, processes=1):
    problem = minizinc.Model(mzn_file)
    solver = minizinc.Solver.lookup(solver)

    instance = minizinc.Instance(solver, problem)

    for key, value in dict_const.items():
        instance[key] = value

    if timeout is not None:
        timeout = datetime.timedelta(seconds=timeout)

    result = instance.solve(timeout=timeout, processes=processes)

    return result.status.name, vars(result.solution)


def pow_fn(power, mul, base, size):
    return mul * power**-(np.arange(1,size+1).astype(float)) + base


def prepare_curve(curve, change_point, states_map):
    
    points = []
    for i in range(len(change_point)):
        cp = change_point
        l = cp[i+1] - cp[i] if i < len(cp) -1 else len(states_map) - cp[i]
        list_exp_points= list(curve[i][:l])
        points.extend(list_exp_points)

        assert len(list_exp_points) == (np.array(states_map) == i).sum()

    assert len(points) == len(states_map)
    return points


def prepare_sequences(data, seq_num, app_params, seed):
    starts, stops = activation_windows(data, **app_params)

    rng = np.random.RandomState(seed)
    list_seq = [data[starts[s]:stops[s], 1].astype(int)
                for s in rng.choice(len(starts), size=seq_num, replace=False)]
    max_len = max([len(s) for s in list_seq]) 

    list_padded = [pad_to_len(s, max_len).tolist() for s in list_seq]

    return list_padded


def pad_to_len(arr, lenght):
    assert len(arr) <= lenght

    rem = lenght - len(arr)

    return np.pad(arr, (rem // 2, rem //2 + rem % 2), mode="edge")

def get_pow_curves(pow_multiplier, pow_base, length, idle_val):

        idle_curve = np.tile(idle_val,length).tolist()
        pow_curves = [list(pow_fn(2, pow_multiplier[i], pow_base[i],length))
                     for i in range(len(pow_multiplier))]

        return [idle_curve] + pow_curves + [idle_curve]
