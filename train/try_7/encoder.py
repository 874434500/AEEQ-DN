from keras.layers import Layer
from keras import layers, Sequential
from train.try_7.Dense_block import Bottleneck
import tensorflow as tf
from include.my_circular_layer import Conv2D_circular
from train.try_7.Convnet import ConvBNRelu
from train.try_7.SENet import SENet

class Encoder(Layer):
    def conv2(self, out_chanenl):
        return Conv2D_circular(filters=out_chanenl,
                             strides=1,
                             kernel_size=3,
                             padding='same',
                             activation='elu')


    def __init__(self):
        super(Encoder, self).__init__()
        self.H = 4
        self.W = 4
        self.conv_channels = 64
        self.message_length = 1

        self.pre_layer1 = ConvBNRelu(filters=self.conv_channels)
        self.pre_layer2 = SENet(filters=1, blocks=4)

        self.first_layer = Sequential(
            self.conv2(self.conv_channels)
        )

        self.Dense_block1 = Bottleneck(self.conv_channels, name='enc_Dense_block1')
        self.Dense_block2 = Bottleneck(self.conv_channels, name='enc_Dense_block2')
        self.Dense_block3 = Bottleneck(self.conv_channels, name='enc_Dense_block3')
        self.Dense_block_a1 = Bottleneck(self.conv_channels, name='enc_Dense_block_a1')
        self.Dense_block_a2 = Bottleneck(self.conv_channels, name='enc_Dense_block_a2')
        self.Dense_block_a3 = Bottleneck(self.conv_channels, name='enc_Dense_block_a3')

        self.fivth_layer = Sequential([
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
            self.conv2(self.conv_channels),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
            self.conv2(self.message_length)]
        )

        self.sixth_layer = Sequential([
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
            self.conv2(self.conv_channels),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
            self.conv2(self.message_length),
            layers.Softmax(axis=-1)
        ])

        self.final_layer = Conv2D_circular(filters=64, kernel_size=3, padding='same', activation='elu')

    def call(self, x, **kwargs):
        image, expanded_message = x

        image = self.pre_layer1(image)
        expanded_message = self.pre_layer2(expanded_message)
        feature0 = self.first_layer(image)
        feature1 = self.Dense_block1(tf.concat((feature0, expanded_message), axis=-1), last=True)
        feature2 = self.Dense_block2(tf.concat((feature0, expanded_message, feature1), axis=-1), last=True)
        feature3 = self.Dense_block3(tf.concat((feature0, expanded_message, feature1, feature2), axis=-1), last=True)
        feature3 = self.fivth_layer(tf.concat((feature3, expanded_message), axis=-1))

        feature_attention1 = self.Dense_block_a1(feature0)
        feature_attention2 = self.Dense_block_a2(tf.concat((feature0, feature_attention1), axis=-1))
        feature_attention3 = self.Dense_block_a3(tf.concat((feature0, feature_attention1, feature_attention2), axis=-1), last=True)
        feature_mask = (self.sixth_layer(feature_attention3)) * 30
        feature = feature3 * feature_mask
        im_w = self.final_layer(feature)
        return im_w




