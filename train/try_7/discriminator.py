from keras.layers import Layer
from include.my_circular_layer import Conv2D_circular
from keras import layers, Sequential
from train.try_7.Dense_block import Bottleneck
import tensorflow as tf
from keras.models import Model

class CustomLayer_expand_dims(Layer):
    def __init__(self, **kwargs):
        super(CustomLayer_expand_dims, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        # 增加一个新的维度
        expanded = tf.expand_dims(x, axis=1)
        # 可以添加更多的处理
        return expanded


class Discriminator(Layer):
    def conv2(self,out_chanenl):
        return Conv2D_circular(filters=out_chanenl,
                               strides=1,
                               kernel_size=3,
                               padding='same',
                               activation='elu')

    def __init__(self):
        super(Discriminator, self).__init__()
        self.channels = 64
        self.message_length = 1
        self.input_img = layers.Input(shape=(32, 32, 1), batch_size=64)

        self.pre_layer = layers.Conv2D(64, (1, 1), dilation_rate=1, activation='elu', padding='same')

        self.first_layer = Sequential([self.conv2(self.channels),
                                         layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
                                         layers.ReLU()])

        self.second_layer = Sequential([self.conv2(self.channels),
                                         layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
                                         layers.ReLU()])

        self.third_layer = Sequential([self.conv2(self.channels),
                                         layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
                                         layers.ReLU()])

        self.fourth_layer = Sequential([self.conv2(self.channels),
                                         layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
                                         layers.ReLU()])

        self.Dense_block1 = Bottleneck(self.channels, name='dis_Dense_block1')
        self.Dense_block2 = Bottleneck(self.channels, name='dis_Dense_block2')
        self.Dense_block3 = Bottleneck(self.channels, name='dis_Dense_block3')

        self.fivth_layer = Sequential([self.conv2(self.message_length),
                                       layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
                                       layers.ReLU()])

        self.average = Sequential([layers.GlobalAveragePooling2D(),
                                   CustomLayer_expand_dims(),
                                   CustomLayer_expand_dims()])

        self.linear = tf.keras.layers.Dense(1)

        input_img = self.pre_layer(self.input_img)
        feature0 = self.first_layer(input_img)
        feature1 = self.second_layer(feature0)
        feature2 = self.third_layer(tf.concat([feature0, feature1], axis=-1))
        feature3 = self.fourth_layer(tf.concat([feature0, feature1, feature2], axis=-1))
        x = self.fivth_layer(feature3)
        X = self.average(x)
        X = tf.squeeze(X, axis=[1, 2])
        validity = self.linear(X)
        self.discriminator = Model(self.input_img, validity)





    # def call(self, x, **kwargs):
    #     feature0 = self.first_layer(x)
    #     feature1 = self.second_layer(feature0)
    #     feature2 = self.third_layer(tf.concat([feature0, feature1], axis=-1))
    #     feature3 = self.fourth_layer(tf.concat([feature0, feature1, feature2], axis=-1))
    #     x = self.fivth_layer(feature3)
    #     X = self.average(x)
    #     X = tf.squeeze(X, axis=[1, 2])
    #     X = self.linear(X)
    #     return X
