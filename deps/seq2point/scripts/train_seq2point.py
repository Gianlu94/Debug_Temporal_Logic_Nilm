import argparse
from pathlib import Path
import json
from types import SimpleNamespace
import tensorflow as tf

import nilm.training


DEFAULT_MODEL_PARAMS = """[[[30, 10, 1], [30, 8, 1], [40, 6, 1], [50, 5, 1], [50, 5, 1]],
                          [[1024]]]"""

def train(params, savepath):
    noise = None
    if params.noise_std is not None:
        noise = tf.keras.layers.GaussianNoise(params.noise_std, name="noise")

    nilm.training.train(params.seed,params.path, params.appliance, params.validation_fraction,
                       params.batch_size, params.length, params.model,
                        json.loads(params.model_params),params.learning_rate, savepath, params.tensorboard_dir,
                       params.name, params.epochs, chaindue=params.chaindue, verbose=params.verbose, noise=noise)






def main():
    parser = argparse.ArgumentParser(description='Training utility for seq2point')
    parser.add_argument("path", type=str,
                          help='The directory containing the formatted dataset')
    parser.add_argument("appliance", type=str,
                          help='The appliance to target')
    parser.add_argument("name", type=str,
                          help='name to use for saving and logging')
    parser.add_argument("-c","--chaindue", action="store_true",
                          help='Transform data in chain2 format')
    parser.add_argument("-l","--length", type=int, default=599,
                          help='Lenght of each sequence')
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                          help='Learning rate')
    parser.add_argument("-e","--epochs", type=int, default=50,
                          help='Epochs')
    parser.add_argument("-b","--batch-size", type=int, default=1000,
                          help='Epochs')
    parser.add_argument("-s","--savedir", default="/tmp/",
                          help='savefile for trained model')
    parser.add_argument("-r","--seed", default=1, type=int,
                          help='seed for the experiments')
    parser.add_argument("-m","--model", default="cnn_fc", type=str,
                          help='model anme')
    parser.add_argument("-p","--model-params", default=DEFAULT_MODEL_PARAMS, type=str,
                          help='model params')
    parser.add_argument("--tensorboard-dir", default=None, type=str,
                          help='tensorboard log directory')
    parser.add_argument("--validation-fraction", default=0.15, type=float,
                          help='fraction of trainset to use for vaidation')
    parser.add_argument("-v", "--verbose", action="store_true",
                          help='tensorboard log directory')
    parser.add_argument("-n", "--noise-std", default=None, type=float,
                          help='std of gaussian noise to use')
    params = parser.parse_args()

    

    savepath = Path(params.savedir) / nilm.training.get_name(params.name, params.appliance, params.model, json.loads(params.model_params))
    savepath.mkdir(parents=True, exist_ok=True)
    train(params, str(savepath))

if __name__ == '__main__':
    main()














