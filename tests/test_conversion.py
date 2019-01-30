from unittest import TestCase
import pkg_resources
import logging, warnings
import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Input, Conv2D, MaxPooling2D, Conv1D, AveragePooling1D, Dropout

from lpdn import convert_to_lpdn


def dense_sequential_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(10,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    return model


def dense_functional_model():
    input = Input((10,))
    y = Dense(32, activation='relu')(input)
    y = Dense(32, activation='relu')(y)
    return Model(inputs=input, outputs=y)


class TestModelConversion(TestCase):

    def setUp(self):
        pass
        # self.session = tf.Session()

    def tearDown(self):
        pass
        #self.session.close()
        #tf.reset_default_graph()

    def test_tf_available(self):
        try:
            pkg_resources.require('tensorflow>=1.0')
        except Exception:
            self.fail("Tensorflow requirement not met")

    def test_dense_sequential_model(self):
        model = dense_sequential_model()
        lp_model = convert_to_lpdn(model)
        model.summary()
        lp_model.summary()

    def test_dense_functional_model(self):
        model = dense_functional_model()
        lp_model = convert_to_lpdn(model)
        model.summary()
        lp_model.summary()