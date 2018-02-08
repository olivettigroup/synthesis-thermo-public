from keras.layers import (Input, Dense, Lambda,
                          RepeatVector, Reshape)
from keras.layers.merge import concatenate
from keras import backend as K
from keras.models import Model
from keras.optimizers import RMSprop
from keras.constraints import non_neg
from keras.utils import plot_model
import numpy as np


class ThermoNet(object):
    def __init__(self, input_dim=2, X_train=None):
        self.thermo_net = None
        self.set_training_X(X_train)
        self.build_thermo_net(input_dim=input_dim)

    def set_training_X(self, X_train):
        self._X_train = X_train

    def build_thermo_net(self, input_dim=2, inner_dim=None):
        if inner_dim is None:
            inner_dim = len(self._X_train)

        def gaussian(args):
            x, sig_vals, scale_vals = args
            mu_vals = K.constant(self._X_train)

            # Some datasets may need sigma clipping to prevent learning
            # a "single bowl"-type solution by learning a single wide Gaussian
            # sig_vals = K.clip(sig_vals, 0.0, MAX_SIGMA)

            sig_reshape = K.reshape(sig_vals, (-1, inner_dim, 1))
            sig = K.concatenate([sig_reshape]*input_dim, axis=2)

            gauss_mix = K.sum(-scale_vals*K.exp(-K.sum(K.pow((x-mu_vals)/sig, 2), axis=2)), axis=1)
            batch_gauss_mix = K.reshape(gauss_mix, (-1, 1))
            return batch_gauss_mix

        latent_input = Input(shape=(input_dim,), name='latent_input')
        x_repeat = RepeatVector(inner_dim)(latent_input)
        scale_layer = Dense(inner_dim, activation="softplus", kernel_constraint=non_neg(), name="scale_layer")(latent_input)
        sig_layer = Dense(inner_dim, activation="softplus", kernel_constraint=non_neg(), name="sigma_layer")(latent_input)
        lambda_out = Lambda(gaussian, name="gaussian")([x_repeat, sig_layer, scale_layer])

        thermo_net = Model(inputs=[latent_input], outputs=[lambda_out])
        thermo_net.compile(
            optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=K.epsilon(), decay=0.0),
            loss="mean_squared_error"
        )
        self.thermo_net = thermo_net

    def train(self, x_data, y_data, epochs=100, batch_size=32):
        self._X_train = x_data
        self.thermo_net.fit(
            x=x_data,
            y=y_data,
            epochs=epochs,
            batch_size=batch_size
        )

    def save_models(self, save_path=""):
        self.thermo_net.save_weights(save_path + "thermo_net.h5")

    def load_models(self, load_path=""):
        self.thermo_net.load_weights(load_path + "thermo_net.h5")


if __name__ == "__main__":
    thermo_net = ThermoNet(X_train=np.zeros(shape=(100,2)))
    thermo_net.build_thermo_net()
    thermo_net.thermo_net.summary()
    plot_model(thermo_net.thermo_net, to_file='thermo_net.png')
