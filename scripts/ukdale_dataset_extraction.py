import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


DS_INFO = {
    'kettle': {
        'houses': [1, 2, 3, 4, 5],
        'channels': [10, 8, 2, 3, 18],
    },
    'microwave': {
        'houses': [1, 2, 5],
        'channels': [13, 15, 23],
    },
    'fridge': {
        'houses': [1, 2, 5],
        'channels': [12, 14, 19],
    },
    'dishwasher': {
        'houses': [1, 2, 5],
        'channels': [6, 13, 22],
    },
    'washingmachine': {
        'houses': [1, 2, 5],
        'channels': [5, 12, 24],
    }
}


def load_house_appliance(mains_path, appliance_path, sampling_rate):

    mains_df = pd.read_csv(mains_path,
                         delim_whitespace=True,
                         names=['time',"power"]

                        )
    app_df = pd.read_csv(appliance_path,
                         delim_whitespace=True,
                         names=['time',"data"]
                        )

    mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s')
    app_df['time'] = pd.to_datetime(app_df['time'], unit='s')

    mains_df.set_index('time', inplace=True)
    app_df.set_index('time', inplace=True)

    df_align = mains_df.join(app_df, how='outer'). \
        resample(str(sampling_rate)+'s')

    df_align = df_align.mean().fillna(method="ffill").dropna()

    df_align.reset_index(inplace=True)

    return df_align[['power', 'data']].to_numpy().astype(np.uint16)


def load_datasets(paths, workers, sampling_rate):

    pool = Parallel(n_jobs=workers)

    datasets_list = pool(delayed(load_house_appliance)(m, a, sampling_rate)
                         for m,a in paths)
    
    return datasets_list


def gen_house_iterator():
    for key in DS_INFO:
        for house, channel in zip(DS_INFO[key]['houses'],DS_INFO[key]['channels']):
            yield (key, house, channel)



def gen_paths(data_dir, info):
    for app, house, channel in info:
        assert channel != 1
        base_path =  Path(data_dir) / ("house_" + str(house))

        mains_path =  base_path / 'channel_1.dat'
        appliance_path = base_path / ('channel_' + str(channel) + '.dat')

        yield str(mains_path), str(appliance_path)


def process_ukdale(params):
    
    meta = list(gen_house_iterator())
    paths = gen_paths(params.data_dir, meta)

    dataset_list = load_datasets(paths, workers=params.workers, sampling_rate=params.sampling_rate)

    np.savez(params.save_path,
             np.asarray(meta, dtype=str), *dataset_list
            )





def main():
    parser = argparse.ArgumentParser(description='Extraction and formatting \
                                     utility for the UKDALE dataset')
    parser.add_argument("data_dir", type=str,
                          help='The directory containing the UKDALE data')
    parser.add_argument("save_path", type=str,
                          help='The file to store the formatted datset')
    parser.add_argument("-w","--workers", type=int, default=1,
                          help='The number of processes to use')
    parser.add_argument("-s","--sampling-rate", type=int, default=6,
                          help='Sampling rate for the data')
    params = parser.parse_args()
    
    process_ukdale(params)

if __name__ == '__main__':
    main()














