from keras import layers
from keras.layers import Layer
from keras import Sequential


class ConvBNRelu(Layer):
    """
    	A sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, filters, stride=1, **kwargs):
        super(ConvBNRelu, self).__init__(**kwargs)
        self.layers = Sequential([
            layers.Conv2D(filters=filters, strides=stride, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
            layers.ReLU(),
        ])

    def call(self, x, **kwargs):
        return self.layers(x)

class ConvNet(Layer):
    '''
    Network that composed by layers of ConvBNRelu
    '''

    def __init__(self, filters, blocks):
        super(ConvNet, self).__init__()

        layer = [ConvBNRelu(filters)] if blocks != 0 else []
        for _ in range(blocks - 1):
            LayerS = ConvBNRelu(filters)
            layer.append(LayerS)
        self.layers = Sequential(layer)

    def call(self, x, **kwargs):
        return self.layers(x)
