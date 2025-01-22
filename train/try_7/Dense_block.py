from keras.layers import Layer, ReLU, BatchNormalization, Conv2D, Concatenate
from include.my_circular_layer import Conv2D_circular


class Bottleneck(Layer):
    def __init__(self, growthRate, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        interChannels = 4 * growthRate
        self.bn1 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)
        self.conv1 = Conv2D_circular(filters=interChannels, kernel_size=1, use_bias=False, activation='elu')
        self.bn2 = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)
        self.conv2 = Conv2D_circular(filters=growthRate, kernel_size=3, use_bias=False, activation='elu', padding='same')


    def call(self, x,last=False, **kwargs):
        out = self.conv1(self.bn1(x))
        out = self.conv2(self.bn2(out))
        if last:
            return out
        else:
            return Concatenate(axis=-1)([x, out])


