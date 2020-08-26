"""
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ConvBNReLU(layers.Layer):
    def __init__(self, filters, kernel_size, stride, padding, **kwargs):
        super(ConvBNReLU, self).__init__()
        kernel_initializer = 'glorot_normal'
        bias_initializer = tf.keras.initializers.Constant(0.02)
        self.conv = layers.Conv2D(filters, kernel_size, stride, padding,
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer,
                                  use_bias=True, **kwargs)
        self.bn = layers.BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x


class Conv2Layer(layers.Layer):
    def __init__(self, filters_1, filters, stride, **kwargs):
        super(Conv2Layer, self).__init__()
        self.conv1 = ConvBNReLU(filters_1, 3, stride, padding='same', **kwargs)
        self.conv2 = ConvBNReLU(filters, 1, 1, padding='valid', **kwargs)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Conv3Layer(layers.Layer):
    def __init__(self, filters_1, filters_2, filters, stride, **kwargs):
        super(Conv3Layer, self).__init__()
        self.conv1 = ConvBNReLU(filters_1, 3, stride, padding='same', **kwargs)
        self.conv2 = ConvBNReLU(filters_2, 1, 1, padding='valid', **kwargs)
        self.conv3 = ConvBNReLU(filters, 3, 1, padding='same', **kwargs)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def multibox(num_classes, boxes_per_location):
    """
    """
    conf_layers = []
    loc_layers = []
    kernel_initializer = 'glorot_normal'
    bias_initializer = tf.keras.initializers.Constant(0.02)
    for num_boxes in boxes_per_location:
        conf_layers.append(
            layers.Conv2D(num_boxes * num_classes, 3, 1,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          padding='same', use_bias=True)
        )
        loc_layers.append(
            layers.Conv2D(num_boxes * 4, 3, 1,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          padding='same', use_bias=True)
        )
    return conf_layers, loc_layers


def FaceDetection(input_shape, num_classes, boxes_per_location, training=True):
    """
    """
    features = []
    x = inputs = layers.Input(input_shape)

    x = Conv2Layer(32, 16, 2)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = Conv2Layer(32, 32, 1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = Conv3Layer(64, 32, 64, 1)(x)
    features.append(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = Conv3Layer(128, 64, 128, 1)(x)
    features.append(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = Conv3Layer(256, 128, 256, 1)(x)
    features.append(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = Conv3Layer(256, 256, 256, 1)(x)
    features.append(x)

    conf_layers, loc_layers = multibox(num_classes, boxes_per_location)
    confs = []
    locs = []
    for x, c, l in zip(features, conf_layers, loc_layers):
        confs.append(c(x))
        locs.append(l(x))

    confs = tf.concat(
        [tf.reshape(o, (tf.shape(o)[0], -1)) for o in confs],
        axis=1)
    locs = tf.concat(
        [tf.reshape(o, (tf.shape(o)[0], -1)) for o in locs],
        axis=1)

    confs = tf.reshape(confs, (tf.shape(confs)[0], -1, num_classes))
    locs = tf.reshape(locs, (tf.shape(locs)[0], -1, 4))
    if training:
        outputs = (confs, locs)
    else:
        outputs = (tf.nn.softmax(confs, axis=-1), locs)
    return tf.keras.Model(inputs, outputs)
