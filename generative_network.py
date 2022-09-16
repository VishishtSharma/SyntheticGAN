# Written by Vishisht Sharma 18315
# This code is for the Generative Network for the GAN to use


# Importing all the necessary libraries
from typing import Optional, Tuple
from tensorflow.keras.layers import Layer
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from tensorflow.keras.optimizers import Optimizer, Adam


class GenerativeNetwork():                                                          # Defining the Generative network

    def __init__(self, optimizer: Optimizer = Adam(0.0002, 0.5)):                    # Specifing the Optimizer to use in the Network
        self.optimizer = optimizer

    def conv2d(self, initial_layer: Layer, filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int],        # Defining the Different Convolution layers to use in Generative network 
               momentum: Optional[float] = 0.99, lrelu_alpha=0.2):                                                      # by using the keras Conv2D layers and improvising them with parameters
        down_conv = Conv2D(filters, kernel_size, strides=strides, padding='same',
                           kernel_initializer=RandomNormal(stddev=0.02))(initial_layer)
        down_conv = BatchNormalization(momentum=momentum)(down_conv) if momentum else down_conv
        down_conv = LeakyReLU(alpha=lrelu_alpha)(down_conv)
        return down_conv

    def intermediate(self, initial_layer: Layer, filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int]):
        conv = Conv2D(filters, kernel_size, strides=strides, padding='same')(initial_layer)
        conv = Activation('relu')(conv)
        return conv

    def deconv2d(self, initial_layer: Layer, skipped_layer: Layer, filters: int, kernel_size: Tuple[int, int],
                 strides: Tuple[int, int], dropout_rate: Optional[float], activation='relu',
                 momentum: Optional[float] = 0.99):
        up_conv = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same',
                                  kernel_initializer=RandomNormal(stddev=0.02))(initial_layer)
        up_conv = BatchNormalization(momentum=momentum)(up_conv) if momentum else up_conv
        up_conv = Dropout(dropout_rate)(up_conv) if dropout_rate else up_conv
        up_conv = Concatenate()([up_conv, skipped_layer])
        up_conv = Activation(activation)(up_conv)
        return up_conv

    def build(self, init_filters: int,                                                                      # Building the Generative model for use in the GAN
              input_shape: Tuple[int, int, int],                                                            # Defining the variables for use in the model
              output_channels: int,
              kernel_size: Tuple[int, int] = (4, 4),
              strides: Tuple[int, int] = (2, 2),
              dropout_rate: Optional[float] = None,
              output_activation: str = 'tanh',
              compile: bool = True) -> Model:
        input = Input(shape=input_shape, name='condition_mask')

        d1 = self.conv2d(input, init_filters, kernel_size, strides, momentum=None)                          # Defining the Generatove model by specifing the layers for the model
        d2 = self.conv2d(d1, init_filters * 2, kernel_size, strides)                                        # Defining the layers for the network
        d3 = self.conv2d(d2, init_filters * 4, kernel_size, strides)
        d4 = self.conv2d(d3, init_filters * 8, kernel_size, strides)
        d5 = self.conv2d(d4, init_filters * 8, kernel_size, strides)
        d6 = self.conv2d(d5, init_filters * 8, kernel_size, strides)
        d7 = self.conv2d(d6, init_filters * 8, kernel_size, strides)

        intermediate = self.intermediate(d7, init_filters * 8, kernel_size, strides)

        u1 = self.deconv2d(intermediate, d7, init_filters * 8, kernel_size, strides, dropout_rate=dropout_rate)
        u2 = self.deconv2d(u1, d6, init_filters * 8, kernel_size, strides, dropout_rate=dropout_rate)
        u3 = self.deconv2d(u2, d5, init_filters * 8, kernel_size, strides, dropout_rate=dropout_rate)
        u4 = self.deconv2d(u3, d4, init_filters * 8, kernel_size, strides, dropout_rate=None)
        u5 = self.deconv2d(u4, d3, init_filters * 4, kernel_size, strides, dropout_rate=None)
        u6 = self.deconv2d(u5, d2, init_filters * 2, kernel_size, strides, dropout_rate=None)
        u7 = self.deconv2d(u6, d1, init_filters, kernel_size, strides, dropout_rate=None)

        output = Conv2DTranspose(output_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                 activation=output_activation)(u7)

        model = Model(input, output, name='generator')
        if compile:
            model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])                       # Compiling the model using the earlier defined optimizers and loss functions

        return model
