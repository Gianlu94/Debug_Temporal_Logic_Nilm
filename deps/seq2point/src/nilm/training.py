import numpy as np
import tensorflow as tf
import os
import random
import pathlib

from . import preprocessing
from . import model

#tf.debugging.set_log_device_placement(True)



def a2t(arr):
    return tf.convert_to_tensor(arr)



def set_seed(seed):
    if seed is not None:
        os.environ['PYTHONHASHSEED']=str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)



def load_data(path_data, str_appliance, fraction_val, chaindue=False):
    metadata, datasets = preprocessing.load_ukdale_dataset(path_data, str_appliance)

    if chaindue:
        datasets = [preprocessing.transform_ukdale_to_chain2(d) for d in datasets]

    datasets = [d.astype(np.float32) for d in datasets]

    all_train_data, test_data = preprocessing.split_ukdale_train_test(metadata, datasets)

    train_data, val_data = preprocessing.train_val_split(all_train_data, fraction_val)

    return train_data, val_data, test_data


def create_datasets(data, size_batch, size_seq, shuffle=False, seed=None, mean=None, std=None):
    ds = preprocessing.make_window_dataset(a2t(data), size_seq, seed, shuffle)

    ds = ds.batch(size_batch)

    if mean is not None and std is not None:
        ds = preprocessing.normalize_labels(ds, mean , std)

    return ds.prefetch(tf.data.AUTOTUNE)


def create_layers( model_name, model_params, length, mains_mean, mains_std, app_mean,app_std, noise=None):
    input_m = model.input_module(length)
    preproc_m = model.normalize_module(input_m,
                                       mains_mean,
                                       mains_std)
    if noise is not None:
        preproc_m = noise(preproc_m)

    nn_m = getattr(model, model_name)(preproc_m, length, *model_params)
    denorm_m = model.denormalize_module(nn_m, app_mean,
                                        app_std)

    return input_m, (nn_m, denorm_m)

def try_load_model(model_dir,restart=False):

    path_savedmodel = None
    list_savefiles = sorted(pathlib.Path(model_dir).glob("*.hdf5"))
    if not restart:
        if len(list_savefiles) > 0:
            path_savedmodel = list_savefiles[-1]

    else:
        for p in list_savefiles:
            p.unlink()

    if path_savedmodel is not None:
        epoch = int(path_savedmodel.name.rstrip(".hdf5"))
        return tf.keras.models.load_model(str(path_savedmodel)), epoch

    else:
        return None


def create_model(model_name, model_params, length, mains_mean, mains_std, app_mean,app_std, learning_rate, noise=None):
    inputs, outputs = create_layers(model_name, model_params, length, mains_mean, mains_std, app_mean,app_std, noise=noise)

    network = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

    metrics = [["msle"], ["mae"]]

    network.compile(optimizer=optimizer,
                    loss="mse",metrics=metrics, loss_weights=[1.0, 0], run_eagerly=False)
    network.summary()

    return network


def get_model(model_name, model_params, length, mains_mean, mains_std,
              app_mean,app_std, learning_rate, model_dir,restart=False, noise=None):

    network = try_load_model(model_dir, restart=restart)

    if network is None:
        network = (create_model(model_name, model_params, length,
                               mains_mean, mains_std, app_mean,app_std, learning_rate, noise=noise),
                   0)

    return network


def get_name(basename, appliance, model, model_params):
    params_str = "".join(np.concatenate([np.array(p).flatten()
                                         for p in model_params]).astype(str))
    return basename  + "_" + appliance + "_" + model+ "_" +  params_str


def get_callbacks(model_dir, tensorboard_dir, epoch, cardinality):

    callbacks = []

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_denormalized_mae', patience=5)

    callbacks.append(earlystopping)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_dir + "/{epoch:02d}.hdf5",
                                                    save_best_only=False)
    callbacks.append(checkpoint)


    if tensorboard_dir is not None:
        tensorboard_callback = TBLogger(log_dir=tensorboard_dir,
                                        offset=cardinality * epoch, update_freq=100)
        callbacks.append(tensorboard_callback)

    return callbacks


def train(seed, path_data,str_appliance, fraction_val,
            size_batch, size_seq, model_name,
          model_params, learning_rate, checkpoints_dir, tensorboard_dir, exp_name,epochs,
          chaindue=False, noise=None, verbose=True):
    set_seed(seed)

    train_data, val_data, test_data = load_data(path_data, str_appliance, fraction_val, chaindue=chaindue)

    mean_mains, std_mains = preprocessing.compute_nomralization_factors(train_data[:, 0])
    mean_app, std_app = preprocessing.compute_nomralization_factors(train_data[:, 1])

    ds_train = create_datasets(train_data, size_batch, size_seq, shuffle=True,
                               seed=seed, mean=mean_app, std=std_app)
    ds_val = create_datasets(val_data, size_batch, size_seq)


    network, start_epoch = get_model(model_name, model_params, size_seq, mean_mains, std_mains,
                        mean_app, std_app, learning_rate, checkpoints_dir,
                        restart=False, noise=noise)

    path_tb_logdir = tensorboard_dir + "/" + get_name(exp_name, model_name, str_appliance,  model_params) if tensorboard_dir is not None else None

    callbacks = get_callbacks(checkpoints_dir, path_tb_logdir, ds_train.cardinality().numpy(), start_epoch)

    training_history = network.fit(ds_train,
                                 epochs=epochs,
                                 initial_epoch=start_epoch,
                                 verbose=verbose,
                                 validation_data = ds_val,
                                 callbacks=callbacks,
                                 #steps_per_epoch=10,
                                 #validation_steps=10,
                                )


class TBLogger(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, offset=0, update_freq=1):
        self.log_dir = log_dir
        self.offset = offset
        self.update_freq = update_freq

        self.train_writer =  tf.summary.create_file_writer(self.log_dir + "/train")

        self.batch_counter = int(self.offset)

        self.accumulated = None
        self.test_counter = 0

        self.metrics_dict = {"loss" : "loss", "denormalized_mae": "mae"}


    def reset_accumulated(self):
        self.accumulated = {k:0. for k in self.metrics_dict}
        self.test_counter = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batch_counter += 1

        if (self.batch_counter % self.update_freq) != 0:
            return

        with self.train_writer.as_default():
            for metric, name in self.metrics_dict.items():
                tf.summary.scalar(name, logs[metric], step=self.batch_counter)

    def on_test_begin(self, epoch, logs=None):
        self.reset_accumulated()

    def on_test_end(self, epoch, logs=None):

        with self.train_writer.as_default():
            for metric, name in self.metrics_dict.items():
                tf.summary.scalar("validation " + name, self.accumulated[metric] / self.test_counter, step=self.batch_counter)

    def on_test_batch_end(self, batch, logs=None):
        self.test_counter += 1
        for metric in self.metrics_dict:
            self.accumulated[metric] += logs[metric]
