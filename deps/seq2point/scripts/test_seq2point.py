import argparse
import sys

import tensorflow as tf 
import numpy as np

import nilm.training
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test(params):
    split = params.split
    target_data = None
    
    if split == "val":
        _, target_data, _ = nilm.training.load_data(params.path, params.appliance, 0.15, chaindue=params.chaindue)
    elif split == "test":
        _, _, target_data = nilm.training.load_data(params.path, params.appliance, 0.15, chaindue=params.chaindue)
    else:
        sys.stderr.write("ERROR: Not recognized '{}' for split parameter\n".format(split))
        sys.exit(-1)

    ds_target_data = nilm.training.create_datasets(target_data, params.batch_size, params.length)

    if params.take is not None:
        ds_target_data = ds_target_data.take(params.take)

    model = tf.keras.models.load_model(params.model) 


    if params.plot or params.save is not None:

        test_history = model.predict(ds_target_data, verbose=params.verbose)[1].squeeze()

        gt = np.concatenate([y for X, y in ds_target_data])
        cumulative = np.concatenate([X[:, params.length//2] for X, y in ds_target_data])

        if params.save is not None:
            np.savez(params.save, np.array([cumulative, test_history, gt]))

        if params.plot:
            title = params.appliance
            if params.chaindue:
                title = title  + " on simulated chaindue"
            plot_predictions(gt, test_history, cumulative,  title)

    else:
        test_history = model.evaluate(ds_target_data, verbose=params.verbose)


def plot_predictions(gt, pred, cum, app):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    points = np.arange(len(gt))

    ax.plot(points, gt, "C0", label="gt")
    ax.plot(points, pred, "C1", label="pred", alpha=0.8)
#    ax.plot(points, cum, "C2", label="mains", alpha=0.5)
    ax.legend()
    ax.set_title(app)
    ax.set_xlabel("time")
    ax.set_ylabel("power")
    plt.show()




def main():
    parser = argparse.ArgumentParser(description='Training utility for seq2point')
    parser.add_argument("path", type=str,
                          help='The directory containing the formatted dataset')
    parser.add_argument("appliance", type=str,
                          help='The appliance to target')
    parser.add_argument("model", type=str,
                          help='The pretrained model to test')
    parser.add_argument("-s", "--split", type=str,
                        help='The split for which generating the embed files (val or test)')
    parser.add_argument("-c","--chaindue", action="store_true",
                          help='Transform data in chain2 format')
    parser.add_argument("-l","--length", type=int, default=599,
                          help='Lenght of each sequence')
    parser.add_argument("-b","--batch-size", type=int, default=1000,
                          help='batch size')
    parser.add_argument("-v", "--verbose", action="store_true",
                          help='tensorboard log directory')
    parser.add_argument("--plot", action="store_true",
                          help='plot pred vs gt')
    parser.add_argument("--on-val", action="store_true",
                          help='do plots on validation')
    parser.add_argument("--save", default=None, type=str,
                          help='savefile for  predictions')
    parser.add_argument("--take", default=None, type=int,
                          help='number of batches to use, default all of them')
    params = parser.parse_args()
    

    test(params)

if __name__ == '__main__':
    main()














