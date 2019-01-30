import logging, warnings
from keras.models import Model
from keras.layers import Dense, Flatten, Activation, Input, InputLayer, Conv2D, MaxPooling2D, Conv1D, AveragePooling1D, Dropout
from ..layers.pooling import LPMaxPooling2D, LPAveragePooling1D
from ..layers.dense import LPDense
from ..layers.convolution import LPConv1D, LPConv2D
from ..layers.activation import LPActivation
from ..layers.reshape import LPFlatten

IGNORE_LIST = [Dropout, InputLayer]


def convert_to_lpdn(keras_model, input_shape=None):
    # Create an equivalent probabilistic model.
    if input_shape is None:
        input_shape = keras_model.layers[0].input_shape[1:] + (2,)
        logging.info("Inferred input shape: " + str(input_shape))

    lp_input = Input(shape=input_shape)
    y = lp_input
    for li, l in enumerate(keras_model.layers):
        if isinstance(l, Conv2D):
            y = LPConv2D(l.filters, l.kernel_size, padding=l.padding, activation=l.activation, name=l.name)(y)
        elif isinstance(l, Conv1D):
            y = LPConv1D(l.filters, l.kernel_size, padding=l.padding, activation=l.activation, name=l.name)(y)
        elif isinstance(l, Dense):
            y = LPDense(l.units, activation=l.activation, name=l.name)(y)
        elif isinstance(l, MaxPooling2D):
            y = LPMaxPooling2D(l.pool_size, strides=l.strides, name=l.name)(y)
        elif isinstance(l, AveragePooling1D):
            y = LPAveragePooling1D(l.pool_size, strides=l.strides, name=l.name)(y)
        elif isinstance(l, Flatten):
            y = LPFlatten(name=l.name)(y)
        elif isinstance(l, Activation):
            y = LPActivation(l.activation, name=l.name)(y)
        elif any([isinstance(l, layerclass) for layerclass in IGNORE_LIST]):
            logging.info("Ignoring layer " + l.name)
        else:
            raise RuntimeError("Layer %s not supported" % str(l))

    model = Model(inputs=lp_input, outputs=y)
    return model
