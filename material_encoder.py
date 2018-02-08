from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import objectives
import json
import numpy as np


class MaterialEncoder(object):
    def __init__(self):
        self.vae = None
        self.encoder = None
        self.decoder = None
        
    def build_vae(self, original_dim=20, latent_dim=2, intermediate_dim=16):
        x = Input(shape=(original_dim,), name="mat_inp")
        h = Dense(intermediate_dim, activation="relu", name="h_enc")(x)
        z_mean = Dense(latent_dim, name="means_enc")(h)
        z_log_var = Dense(latent_dim, name="vars_enc")(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(latent_dim,), mean=0.0, stddev=1.0)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling)([z_mean, z_log_var])

        decoder_h = Dense(intermediate_dim, activation="relu", name="h_dec")
        decoder_mean = Dense(original_dim, activation='sigmoid', name="means_dec")
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)     

        def vae_loss(x, x_decoded_mean):
            rec_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

            return K.mean(rec_loss + kl_loss)

        encoder = Model(inputs=[x], outputs=[z_mean])

        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        decoder = Model(inputs=[decoder_input], outputs=[_x_decoded_mean])

        vae = Model(inputs=[x], outputs=[x_decoded_mean])
        vae.compile(
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),
            loss=vae_loss
         )

        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def train(self, x_data, num_epochs, val_split, batch_size, history_path=""):       
        history = self.vae.fit(
            x=x_data,
            y=x_data,
            shuffle=True,
            validation_split=val_split,
            epochs=num_epochs,
            batch_size=batch_size
        )

        with open(history_path + "training_history.json", "w") as f:
            f.write(json.dumps(history.history, indent=2))

    def save_models(self, save_path=""):
        self.vae.save_weights(save_path + "material_vae.h5")
        self.encoder.save_weights(save_path + "material_encoder.h5")
        self.decoder.save_weights(save_path + "material_decoder.h5")

    def load_models(self, load_path=""):
        self.build_vae()
        self.vae.load_weights(load_path + "material_vae.h5")
        self.encoder.load_weights(load_path + "material_encoder.h5")
        self.decoder.load_weights(load_path + "material_decoder.h5")

if __name__ == "__main__":
    mat_enc = MaterialEncoder()
    mat_enc.build_vae()
    mat_enc.vae.summary()
    plot_model(mat_enc.vae, to_file='material_vae.png')