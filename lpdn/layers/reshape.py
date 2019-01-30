import tensorflow as tf
from keras import backend as K
from keras.layers import Flatten
import numpy as np


class LPFlatten(Flatten):
    """
    Propagate distributions over a Dense layer
    """
    def __init__(self, **kwargs):
        super(LPFlatten, self).__init__(**kwargs)
        self.n_batch = None
        self.n_feat = None

    def compute_output_shape(self, input_shape):
        self.n_batch = input_shape[0]
        self.n_feat = np.prod(input_shape[1:-1])
        return self.n_batch, self.n_feat, 2

    def assert_input_compatibility(self, inputs):
        return super(LPFlatten, self).assert_input_compatibility(inputs[..., 0])

    def call(self, inputs):
        n_batch = tf.shape(inputs)[0]
        return K.reshape(inputs, (n_batch, -1, 2))
