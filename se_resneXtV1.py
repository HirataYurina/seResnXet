# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:se_resneXt.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf


# TODO:循环太多，运行速度太慢，需要改进
# TODO:将多个循环的分组卷积封装成layer
# TODO:采用论文中的第三种结构，执行效率更高
class Conv2DGroup(keras.layers.Layer):
    
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv2DGroup, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups

        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(tf.keras.layers.Conv2D(filters=self.group_out_num,
                                                         kernel_size=kernel_size,
                                                         strides=strides,
                                                         padding=padding,
                                                         data_format=data_format,
                                                         dilation_rate=dilation_rate,
                                                         activation=activation,
                                                         use_bias=use_bias,
                                                         kernel_initializer=kernel_initializer,
                                                         bias_initializer=bias_initializer,
                                                         kernel_regularizer=kernel_regularizer,
                                                         bias_regularizer=bias_regularizer,
                                                         activity_regularizer=activity_regularizer,
                                                         kernel_constraint=kernel_constraint,
                                                         bias_constraint=bias_constraint,
                                                         **kwargs))

    def call(self, inputs, **kwargs):
        con_list = self.conv_list
        features = []
        for i in range(self.groups):
            feature = con_list[i](inputs[..., i * self.group_in_num:(i + 1) * self.group_in_num])
            features.append(feature)
        features = layers.Concatenate()(features)

        return features


def conv_block(inputs, kernel_size, filters, strides=2, cardinality=32):

    filter1, filter2, filter3 = filters

    x = layers.Conv2D(filter1, 1, strides=strides)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = Conv2DGroup(filter2, filter2, kernel_size, padding='same', groups=cardinality)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filter3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    y = layers.Conv2D(filter3, 1, strides=strides)(inputs)
    y = layers.BatchNormalization()(y)

    outputs = layers.Add()([x, y])
    outputs = layers.ReLU()(outputs)

    return outputs


def identity_block(inputs, kernel_size, filters, cardinality=32):

    filter1, filter2, filter3 = filters

    x = layers.Conv2D(filter1, 1)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = Conv2DGroup(filter2, filter2, kernel_size, padding='same', groups=cardinality)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filter3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Add()([x, inputs])
    outputs = layers.ReLU()(outputs)

    return outputs


def squeeze_and_excitation(inputs, reduction_ratio=4):
    """
        paper：Squeeze-and-Excitation Networks
        在特征信息中，通道之间存在依赖关系，通过senet可以显示地去表示
        这些通道之间的依赖关系。采用神经网络学习每个通道的weights，
        重要的通道给予大的weights，不重要的通道给予小的weights。
    Args:
        inputs: features
        reduction_ratio: the reduction ratio

    Returns: y: scale (1, 1, C)

    """
    channels = inputs.shape[-1]

    c_reduce = (channels // reduction_ratio)

    # (batch, c)
    y = layers.GlobalMaxPool2D()(inputs)
    # (batch, c/r)
    y = layers.Dense(c_reduce)(y)
    y = layers.ReLU()(y)
    # (batch, c)
    y = layers.Dense(channels)(y)
    y = keras.activations.sigmoid(y)

    y = layers.Reshape(target_shape=(1, 1, -1))(y)

    return y


# TODO:将block层进行封装，使得代码更加优雅
def se_resnext(inputs):
    x = layers.ZeroPadding2D((3, 3))(inputs)
    x = layers.Conv2D(64, 7, 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPool2D(3, strides=2, padding='same')(x)

    # block*3
    x1 = conv_block(x, 3, [128, 128, 256], strides=1)
    scale1 = squeeze_and_excitation(x1)
    x1 = tf.multiply(x1, scale1)

    x2 = identity_block(x1, 3, [128, 128, 256])
    scale2 = squeeze_and_excitation(x2)
    x2 = tf.multiply(x2, scale2)

    x3 = identity_block(x2, 3, [128, 128, 256])
    scale3 = squeeze_and_excitation(x3)
    x3 = tf.multiply(x3, scale3)

    # block*4
    x4 = conv_block(x3, 3, [256, 256, 512])
    scale4 = squeeze_and_excitation(x4)
    x4 = tf.multiply(x4, scale4)

    x5 = identity_block(x4, 3, [256, 256, 512])
    scale5 = squeeze_and_excitation(x5)
    x5 = tf.multiply(x5, scale5)

    x6 = identity_block(x5, 3, [256, 256, 512])
    scale6 = squeeze_and_excitation(x6)
    x6 = tf.multiply(x6, scale6)

    x7 = identity_block(x6, 3, [256, 256, 512])
    scale7 = squeeze_and_excitation(x7)
    x7 = tf.multiply(x7, scale7)

    # block*6
    x8 = conv_block(x7, 3, [512, 512, 1024])
    scale8 = squeeze_and_excitation(x8)
    x8 = tf.multiply(x8, scale8)

    x9 = identity_block(x8, 3, [512, 512, 1024])
    scale9 = squeeze_and_excitation(x9)
    x9 = tf.multiply(x9, scale9)

    x10 = identity_block(x9, 3, [512, 512, 1024])
    scale10 = squeeze_and_excitation(x10)
    x10 = tf.multiply(x10, scale10)

    x11 = identity_block(x10, 3, [512, 512, 1024])
    scale11 = squeeze_and_excitation(x11)
    x11 = tf.multiply(x11, scale11)

    x12 = identity_block(x11, 3, [512, 512, 1024])
    scale12 = squeeze_and_excitation(x12)
    x12 = tf.multiply(x12, scale12)

    x13 = identity_block(x12, 3, [512, 512, 1024])
    scale13 = squeeze_and_excitation(x13)
    x13 = tf.multiply(x13, scale13)

    # block*3
    x14 = conv_block(x13, 3, [1024, 1024, 2048])
    scale14 = squeeze_and_excitation(x14)
    x14 = tf.multiply(x14, scale14)

    x15 = identity_block(x14, 3, [1024, 1024, 2048])
    scale15 = squeeze_and_excitation(x15)
    x15 = tf.multiply(x15, scale15)

    x16 = identity_block(x15, 3, [1024, 1024, 2048])
    scale16 = squeeze_and_excitation(x16)
    x16 = tf.multiply(x16, scale16)

    return x16


if __name__ == '__main__':

    img_inputs = keras.Input(shape=(224, 224, 3))
    outs = se_resnext(img_inputs)

    se_resnext_model = keras.Model(img_inputs, outs)

    # se_resnext_model.summary()

    num_layers = len(se_resnext_model.layers)
    print('the number of total layers is {}'.format(num_layers))
