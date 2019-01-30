## Lightweight Probabilistic Deep Network
This repository contains an unofficial implementation of [Lightweight Probabilistic Deep Networks](https://arxiv.org/abs/1805.11327) using Keras (and assuming Tensorflow backend). The library is still under development and not all Keras layers are currently supported. Moreover, only ReLU and Linear are supported as layer activations.

### How to use
A Keras probabilistic model can be built from scratch or converted from an existing model.

Let assume we want to build the equivalent of the following model:
```py
from keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes))
```

We can either use the *conversion utility*
```py
from lpdn import convert_to_lpdn

lp_model = convert_to_lpdn(model)
```

or build the model *from scratch* by replacing the original layers with the Lightweight Propabilistic (LP-) equivalent.

```py
from lpdn import LPDense, LPFlatten, LPActivation, LPConv2D, LPMaxPooling2D

lp_model = Sequential()
lp_model.add(LPConv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
lp_model.add(LPConv2D(64, (3, 3), activation='relu'))
lp_model.add(LPMaxPooling2D(pool_size=(2, 2)))
lp_model.add(LPFlatten())
lp_model.add(LPDense(num_classes))
```
If `model` takes an input of shape `[batch, n_features]`, `lp_model` requires an input of shape `[batch, n_features, 2]` where mean and variance of the input features are stacked along the last dimension. Similarly, the output will also have one additional dimension to account for mean and variance.

