import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import activations


class GroupConv2D(tf.keras.layers.Layer):
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
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

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
                                                         activation=activations.get(activation),
                                                         use_bias=use_bias,
                                                         kernel_initializer=initializers.get(kernel_initializer),
                                                         bias_initializer=initializers.get(bias_initializer),
                                                         kernel_regularizer=regularizers.get(kernel_regularizer),
                                                         bias_regularizer=regularizers.get(bias_regularizer),
                                                         activity_regularizer=regularizers.get(activity_regularizer),
                                                         kernel_constraint=constraints.get(kernel_constraint),
                                                         bias_constraint=constraints.get(bias_constraint),
                                                         **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i*self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out


class ResNeXt_BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(ResNeXt_BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.group_conv = GroupConv2D(input_channels=filters,
                                      output_channels=filters,
                                      kernel_size=(3, 3),
                                      strides=strides,
                                      padding="same",
                                      groups=groups)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=2 * filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.shortcut_conv = tf.keras.layers.Conv2D(filters=2 * filters,
                                                    kernel_size=(1, 1),
                                                    strides=strides,
                                                    padding="same")
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output


def build_ResNeXt_block(filters, strides, groups, repeat_num):
    block = tf.keras.Sequential()
    block.add(ResNeXt_BottleNeck(filters=filters,
                                 strides=strides,
                                 groups=groups))
    for _ in range(1, repeat_num):
        block.add(ResNeXt_BottleNeck(filters=filters,
                                     strides=1,
                                     groups=groups))

    return block

NUM_CLASSES = 10


class ResNeXt(tf.keras.Model):
    def __init__(self, repeat_num_list, cardinality):
        if len(repeat_num_list) != 4:
            raise ValueError("The length of repeat_num_list must be four.")
        super(ResNeXt, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")
        self.block1 = build_ResNeXt_block(filters=128,
                                          strides=1,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[0])
        self.block2 = build_ResNeXt_block(filters=256,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[1])
        self.block3 = build_ResNeXt_block(filters=512,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[2])
        self.block4 = build_ResNeXt_block(filters=1024,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[3])
        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        x = self.pool2(x)
        x = self.fc(x)

        return x


def ResNeXt50():
    return ResNeXt(repeat_num_list=[3, 4, 6, 3],
                   cardinality=32)


def ResNeXt101():
    return ResNeXt(repeat_num_list=[3, 4, 23, 3],
                   cardinality=32)

if __name__ == '__main__':
    model = ResNeXt50()
    import tensorflow.keras as keras

    inputs = keras.Input(shape=(224, 224, 3))
    outputs = model(inputs)

    model1 = keras.Model(inputs, outputs)