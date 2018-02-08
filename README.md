# synthesis-thermo-public
Implementations of material-embedding and thermodynamic-function-learning models

## thermo_net.py

This defines a class which implements a **feed-forward neural network**, which is constrained to learn the parameters of a particular type of Gaussian kernel (i.e., inverted Gaussians centered at training points).
It's intended to be used when you know, by assumption, that all training data should induce a type of negative Gaussian potential well, and that all training data represent observations of local minima in the function which you're trying to learn.

After training, you'll have access to a single model:

1. `thermo_net`, the learned function which outputs a single scalar value for each input

Running this file as-is should generate a summary of an example default model architecture:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
latent_input (InputLayer)        (None, 2)             0
____________________________________________________________________________________________________
repeat_vector_2 (RepeatVector)   (None, 100, 2)        0           latent_input[0][0]
____________________________________________________________________________________________________
sigma_layer (Dense)              (None, 100)           300         latent_input[0][0]
____________________________________________________________________________________________________
scale_layer (Dense)              (None, 100)           300         latent_input[0][0]
____________________________________________________________________________________________________
gaussian (Lambda)                (None, 1)             0           repeat_vector_2[0][0]
                                                                   sigma_layer[0][0]
                                                                   scale_layer[0][0]
====================================================================================================
Total params: 600
Trainable params: 600
Non-trainable params: 0
____________________________________________________________________________________________________
```

You'll also generate the `thermo_net.png` file, which shows this model architecture graphically.

## material_encoder.py

This defines a class which implements a **variational autoencoder**. It's intended to encode materials (i.e., atomic coordinates of some sort), but, in principle, could be extended for a variety of other purposes.
The class comes with some utility functions for saving and loading trained models, as the default `keras` methods don't play well with models containing `Lambda()` layers.

After training, you'll have access to three models:

1. `vae`, the full encoder-decoder
1. `encoder`, the encoder into the latent vectors
1. `decoder`, the decoder from the latent vectors

Running this file as-is should generate a summary of an example default model architecture:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
mat_inp (InputLayer)             (None, 20)            0
____________________________________________________________________________________________________
h_enc (Dense)                    (None, 16)            336         mat_inp[0][0]
____________________________________________________________________________________________________
means_enc (Dense)                (None, 2)             34          h_enc[0][0]
____________________________________________________________________________________________________
vars_enc (Dense)                 (None, 2)             34          h_enc[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 2)             0           means_enc[0][0]
                                                                   vars_enc[0][0]
____________________________________________________________________________________________________
h_dec (Dense)                    (None, 16)            48          lambda_1[0][0]
____________________________________________________________________________________________________
means_dec (Dense)                (None, 20)            340         h_dec[0][0]
====================================================================================================
Total params: 792
Trainable params: 792
Non-trainable params: 0
____________________________________________________________________________________________________
```

You'll also generate the `material_vae.png` file, which shows this model architecture graphically.

## synthesis_dataset.json

This file catalogs the different TiO2 polymorphs listed in the Materials Project (at the time of writing), along with whether or not each polymorph has been experimentally verified or not (based on cross referencing against the Inorganic Crystal Structure Database).