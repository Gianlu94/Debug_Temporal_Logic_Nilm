import tensorflow as tf 


def input_module(input_len):
    return tf.keras.layers.Input(shape=(input_len,))


def cnn_fc(input_layer, window_size, conv_params, fc_params):
    reshaped = tf.keras.layers.Reshape(target_shape=(-1,1))(input_layer)
    for conv in conv_params:
        reshaped = tf.keras.layers.Conv1D(filters=conv[0], kernel_size=conv[1],
                                          strides=conv[2], padding="same", activation="relu")(reshaped)
        if len(conv) == 4:
            reshaped = tf.keras.layers.ZeroPadding1D(padding=conv[3])(reshaped)
            reshaped = tf.keras.layers.MaxPool1D(pool_size=conv[3])(reshaped)


    reshaped = tf.keras.layers.Flatten()(reshaped)
    for fc in fc_params:
        reshaped = tf.keras.layers.Dense(fc[0],activation="relu")(reshaped)

    output_layer = tf.keras.layers.Dense(1,
                                          activation="linear", name="nn_output")(reshaped)

    return output_layer


def normalize_module(input_layer, mean, std):

    norm = tf.keras.layers.Normalization(mean=mean, variance=std**2, axis=None)(input_layer)

    return norm

def denormalize_module(input_layer, mean,std):
    denorm = tf.keras.layers.Dense(1,
                                   kernel_initializer=tf.constant_initializer(std),
                                   bias_initializer=tf.constant_initializer(mean),
                                   trainable=False, name="denormalized"
                                  )(input_layer)

    return denorm

def load(path):
    return tf.keras.models.load_model(path)


def save(path, model):
    model.save(path)
