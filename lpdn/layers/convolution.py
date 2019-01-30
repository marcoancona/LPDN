import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Conv1D
from .activation import filter_activation


class LPConv2D(Conv2D):
    """
    Propagate distributions over a probabilistic Conv2D layer
    """
    def __init__(self, filters, kernel_size, **kwargs):
        super(LPConv2D, self).__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        return super(LPConv2D, self).build(input_shape[:-1])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(LPConv2D, self).compute_output_shape(input_shape[:-1])
        return original_output_shape + (2,)

    def assert_input_compatibility(self, inputs):
        print (inputs)
        return super(LPConv2D, self).assert_input_compatibility(inputs[..., 0])

    def _conv2d(self, input, kernel):
        return K.conv2d(input, kernel, self.strides, self.padding, self.data_format, self.dilation_rate)

    def call(self, inputs):
        m = inputs[..., 0]
        v = inputs[..., 1]

        m = self._conv2d(m, self.kernel)
        v = self._conv2d(v, self.kernel**2)

        if self.use_bias:
            m = K.bias_add(
                m,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            m, v = filter_activation(self.activation.__name__, m, v)

        return tf.stack([m, v], -1)


class LPConv1D(Conv1D):
    """
    Propagate distributions over a probabilistic Conv1D layer
    """
    def __init__(self, filters, kernel_size, **kwargs):
        super(LPConv1D, self).__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        return super(LPConv1D, self).build(input_shape[:-1])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(LPConv1D, self).compute_output_shape(input_shape[:-1])
        return original_output_shape + (2,)

    def assert_input_compatibility(self, inputs):
        return super(LPConv1D, self).assert_input_compatibility(inputs[..., 0])

    def _conv1d(self, input, kernel):
        return K.conv1d(input, kernel, self.strides, self.padding, self.data_format)

    def call(self, inputs):
        m = inputs[..., 0]
        v = inputs[..., 1]

        m = self._conv1d(m, self.kernel)
        v = self._conv1d(v, self.kernel**2)

        if self.use_bias:
            m = K.bias_add(
                m,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            m, v = filter_activation(self.activation.__name__, m, v)

        return tf.stack([m, v], -1)
