import tensorflow as tf
from keras import backend as K
from keras.layers import MaxPooling2D, AveragePooling1D
import numpy as np

from tensorflow.contrib import distributions as dist
exp = np.exp
normal = dist.Normal(loc=0., scale=1.)


def _ab_max_pooling(a, b):
    mu_a = a[..., 0]
    va = a[..., 1]
    mu_b = b[..., 0]
    vb = b[..., 1]
    vavb = tf.maximum(va + vb, 0.00001) ** 0.5

    muamub = mu_a - mu_b
    muamub_p = mu_a + mu_b
    alpha = muamub / vavb

    mu_c = vavb * normal.prob(alpha) + muamub * normal.cdf(alpha) + mu_b
    vc = muamub_p * vavb * normal.prob(alpha)
    vc += (mu_a ** 2 + va) * normal.cdf(alpha) + (mu_b ** 2 + vb) * (1. - normal.cdf(alpha)) - mu_c ** 2
    return tf.stack([mu_c, vc], -1)


class LPMaxPooling2D(MaxPooling2D):

    def assert_input_compatibility(self, inputs):
        return super(LPMaxPooling2D, self).assert_input_compatibility(inputs[..., 0])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(LPMaxPooling2D, self).compute_output_shape(input_shape[:-1])
        return original_output_shape + (2,)

    def extract_patches(self, x):
        return tf.extract_image_patches(
            x,
            ksizes=(1,) + self.pool_size + (1,),
            strides=(1,) + self.strides + (1,),
            padding='VALID',
            rates=[1, 1, 1, 1]
        )

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):

        m = inputs[..., 0]
        v = inputs[..., 1]

        n_channels = m.get_shape().as_list()[-1]
        n_pool = np.prod(self.pool_size)

        # TODO: can all this long pipeline of reshaping, transpositions be simplified or made more efficient?
        m = self.extract_patches(m)
        v = self.extract_patches(v)

        patches_shape = m.get_shape().as_list()
        if patches_shape[0] is None:
            patches_shape[0] = -1

        m = tf.reshape(m, (patches_shape[0:3] + [n_pool, n_channels]))
        v = tf.reshape(v,   (patches_shape[0:3] + [n_pool, n_channels]))

        m = tf.transpose(m, (0, 1, 2, 4, 3))
        v = tf.transpose(v,   (0, 1, 2, 4, 3))

        # Everything in batch dimension except dimension with pooling elements
        m = tf.reshape(m, (-1, n_pool))
        v = tf.reshape(v,   (-1, n_pool))

        # Transpose because scan is over dimension 0
        m = tf.transpose(m)
        v = tf.transpose(v)

        # Apply max pooling in sequence
        tmp = tf.stack([m, v], -1)
        tmp = tf.scan(_ab_max_pooling, tmp, reverse=True)
        m = tmp[0, :, 0]
        v = tmp[0, :, 1]

        # Start inverting all reshaping to bet (batch, 1)
        m = tf.transpose(m)
        v = tf.transpose(v)

        m = tf.reshape(m, (patches_shape[0:3] + [n_channels,]))
        v = tf.reshape(v,   (patches_shape[0:3] + [n_channels,]))

        return tf.stack([m, v], -1)


class LPAveragePooling1D(AveragePooling1D):

    def assert_input_compatibility(self, inputs):
        return super(LPAveragePooling1D, self).assert_input_compatibility(inputs[..., 0])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(LPAveragePooling1D, self).compute_output_shape(input_shape[:-1])
        return original_output_shape + (2,)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        m = inputs[..., 0]
        v = inputs[..., 1]

        m = K.pool2d(m, pool_size, strides,padding, data_format, pool_mode='avg')
        v = K.pool2d(v, pool_size, strides,padding, data_format, pool_mode='avg') / self.pool_size

        return tf.stack([m, v], -1)


