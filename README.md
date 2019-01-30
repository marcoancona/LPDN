## Lightweight Probabilistic Deep Network
This repository contains an unofficial implementation of [Lightweight Probabilistic Deep Networks](https://arxiv.org/abs/1805.11327) using Keras (and assuming Tensorflow backend). The library is still under development and not all Keras layers are currently supported.

### How to use
#### Build a model from scratch
A Keras model can be build from scratch just replacing the original layers with the Lightweight Propabilistic (LP-) equivalent.

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
model.add(Activation('softmax'))
```

We can either use the conversion utility 
```py
from lpdn import convert_to_lpdn

lp_model = convert_to_lpdn(Model(inputs=model.inputs, outputs=model.layers[-2].output))
```

or build the model from scratch

```py
from lpdn import LPDense, LPFlatten, LPActivation, LPConv2D, LPMaxPooling2D

model = Sequential()
model.add(LPConv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(LPConv2D(64, (3, 3), activation='relu'))
model.add(LPMaxPooling2D(pool_size=(2, 2)))
model.add(LPFlatten())
model.add(LPDense(num_classes))
```

**Notice that softmax is not supported yet, therefore we removed it in both cases**

