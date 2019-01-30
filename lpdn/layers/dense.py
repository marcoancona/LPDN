import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from .activation import filter_activation


class LPDense(Dense):
    """
    Propagate distributions over a probabilistic Dense layer
    """
    def __init__(self, units, **kwargs):
        super(LPDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        return super(LPDense, self).build(input_shape[:-1])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(LPDense, self).compute_output_shape(input_shape[:-1])
        return original_output_shape + (2,)

    def assert_input_compatibility(self, inputs):
        return super(LPDense, self).assert_input_compatibility(inputs[..., 0])

    def call(self, inputs):
        m = inputs[..., 0]
        v = inputs[..., 1]

        m = K.dot(m, self.kernel)
        v = K.dot(v, self.kernel ** 2)

        if self.use_bias:
            m += self.bias

        if self.activation is not None:
            m, v = filter_activation(self.activation.__name__, m, v)

        return tf.stack([m, v], -1)