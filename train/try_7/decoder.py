from keras.layers import Layer
from keras import layers, Sequential
from train.try_7.Dense_block import Bottleneck
from include.my_circular_layer import Conv2D_circular
from train.try_7.Convnet import ConvBNRelu

class Decoder(Layer):
        def conv2(self, out_chanenl):
            return Conv2D_circular(filters=out_chanenl,
                                 strides=1,
                                 kernel_size=3,
                                 padding='same',
                                 activation='elu')


        def __init__(self):
            super(Decoder, self).__init__()
            self.channels = 64
            self.message_length = 1

            self.pre_layer = ConvBNRelu(filters=self.channels)

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

            self.Dense_block1 = Bottleneck(self.channels, name='dec_Dense_block1')
            self.Dense_block2 = Bottleneck(self.channels, name='dec_Dense_block2')
            self.Dense_block3 = Bottleneck(self.channels, name='dec_Dense_block3')

            self.fivth_layer = Sequential([self.conv2(self.message_length),
                                          layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
                                          layers.ReLU()])


        def call(self, image_with_wm, **kwargs):

            image_with_wm = self.pre_layer(image_with_wm)
            feature0 = self.first_layer(image_with_wm)
            feature1 = self.second_layer(feature0)
            feature2 = self.third_layer(layers.concatenate([feature0, feature1], axis=-1))
            feature3 = self.fourth_layer(layers.concatenate([feature0, feature1, feature2], axis=-1))
            x = self.fivth_layer(feature3)
            return x



