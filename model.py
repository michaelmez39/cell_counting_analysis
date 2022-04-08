import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class blocc(layers.Layer):
    def __init__(self, filters, k_size, stride = 1, pad = 0, activation = layers.LeakyReLU()):
        super().__init__()
        self.pad = pad
        self.conv1 = layers.Conv2D(filters, kernel_size = k_size, strides = stride, padding = 'valid')
        self.activation = activation
        self.batchNorm = layers.BatchNormalization()

    def call(self, input_tensor, training = False):
        input = input_tensor
        if self.pad != 0:
            input = tf.pad(input, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]], 'constant')

        x = self.conv1(input)
        x = self.batchNorm(x, training = training)
        return self.activation(x)

class simplebloc(layers.Layer):
    def __init__(self, out1, out3, activation = layers.LeakyReLU()):
        super().__init__()

        self.conv1 = blocc(out1, 1, pad = 0, activation = activation)
        self.conv2 = blocc(out3, 3, pad = 1, activation = activation)

    def call(self, input):
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(input)

        output = tf.concat(values = [conv1_out, conv2_out], axis = 3)
        return output

class countception(keras.Model):
    def __init__(self, in_channels = 3, out_channels = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = layers.LeakyReLU(0.01)
        self.final_activation = layers.LeakyReLU(0.01)
        self.patch_size = 32

        self.conv1 = blocc(64, 3, pad = self.patch_size, activation = self.activation)
        self.simple1 = simplebloc(16, 16, activation = self.activation)
        self.simple2 = simplebloc(16, 32, activation = self.activation)
        self.conv2 = blocc(16, 14, activation = self.activation)
        self.simple3 = simplebloc(112, 48, activation=self.activation)
        self.simple4 = simplebloc(64, 32, activation=self.activation)
        self.simple5 = simplebloc(40, 40, activation=self.activation)
        self.simple6 = simplebloc(32, 96, activation=self.activation)
        self.conv3 = blocc(32, 18, activation=self.activation)
        self.conv4 = blocc(64, 1, activation=self.activation)
        self.conv5 = blocc(64, 1, activation=self.activation)
        self.conv6 = blocc(1, 1, activation = self.final_activation)

    def call(self, input):
        x = self.conv1(input)
        x = self.simple1(x)
        x = self.simple2(x)
        x = self.conv2(x)
        x = self.simple3(x)
        x = self.simple4(x)
        x = self.simple5(x)
        x = self.simple6(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x