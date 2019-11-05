import tensorflow.keras as keras
from tensorflow.keras.layers import concatenate, Conv2D, Dense, Flatten, Layer, MaxPooling2D

from week_7.backstage.utils import *


class Inception(Layer):

    """
    This is an example of how can you define your own layer from existing layers. It is very similar to how we define
    models.

    Change the __init__ and call methods so that the Inception layer looks like the layer depicted in the figure
    in the notebook. The Inception layer consists of four branches, that are concatenated at the end.

    You should be able to build this layer from the layers you have already seen - Conv2D, MaxPooling2D and concatenate.

    Several notes:
        1) The 3x3 max-pooling layer in the 4th branch has 1x1 stride
        2) You should use concatenate with lowercase `c` at the start. Concatenate with uppercase `C` is slightly
        different and does not work properly in Layer definitions.
    """

    def __init__(self, filters, activation):
        """
        :param filters: How many filters are in the convolutional layers within this layer.
        :param activation: What is the activation function used
        """
        super(Inception, self).__init__()

        # FIXME: Initialize all the layers you need for the Inception layer
        self.b1conv1 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.b2conv1 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.b3conv1 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.b4conv1 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.conv3 = Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation=activation)
        self.conv5 = Conv2D(
            filters=filters,
            kernel_size=5,
            padding='same',
            activation=activation)
        self.pool = MaxPooling2D(pool_size=(3, 3),strides=1, padding='same')

    def call(self, x):
        # FIXME: Build the Inception layer
        b1 = self.b1conv1(x)
        b2 = self.conv3(self.b2conv1(x))
        b3 = self.conv5(self.b3conv1(x))
        b4 = self.b4conv1(self.pool(x))
        #x = self.conv1(x)
        #x = self.conv2(x)
        #return x
        return concatenate([b1, b2, b3, b4])


class InceptionNet(keras.Model):
    """
    Inception version of the simple CNN we used previously. You do not need to change anything here.
    """

    def __init__(self, filters, dim_output):
        super(InceptionNet, self).__init__()
        self.model_layers = [
            Inception(
                filters=filters,
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Inception(
                filters=filters,
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Inception(
                filters=filters,
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(
                units=512,
                activation='relu'),
            Dense(
                units=dim_output,
                activation='softmax')
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
