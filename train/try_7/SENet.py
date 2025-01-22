from keras import layers
from keras.layers import Layer
from keras.activations import relu
from keras import Sequential
from mbrs.command_function import CustomLayer_expand_dims
from mbrs.command_function import CustomLayer_sigmoid

class BasicBlock(Layer):
    def __init__(self, filters, r, drop_rate):
        super(BasicBlock, self).__init__()
        self.downsample = None
        # if (in_channels != out_channels):
        self.downsample = Sequential([
            layers.Conv2D(filters=filters, kernel_size=1, padding='same', tride=drop_rate, use_bias=False),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
        ])

        self.left = Sequential([
            layers.Conv2D(filters=filters, kernel_size=3, padding='same', tride=drop_rate, use_bias=False),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
            layers.ReLU(),
            layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
        ])

        self.se = Sequential([
            layers.GlobalAveragePooling2D(),
            CustomLayer_expand_dims(),
            CustomLayer_expand_dims(),
            layers.Conv2D(filters=filters // r, kernel_size=1, use_bias=False),
            layers.ReLU(),
            layers.Conv2D(filters=filters // r, kernel_size=1, use_bias=False),
            CustomLayer_sigmoid(),
        ])

    def call(self, x, **kwargs):
        identity = x
        x = self.left(x)
        scale = self.se(x)
        x = x * scale

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = relu(x)
        return x




class BottleneckBlock(Layer):
    def __init__(self, filters, drop_rate):
        super(BottleneckBlock, self).__init__()
        self.downsample = None
        # if (in_channels != out_channels):
        self.downsample = Sequential([
            layers.Conv2D(filters=filters, kernel_size=1, padding='same', strides=drop_rate, use_bias=False),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
        ])

        self.left = Sequential([
            layers.Conv2D(filters=filters, kernel_size=1, strides=drop_rate, padding='same', use_bias=False),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
            layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
            layers.ReLU(),
            layers.Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False),
            layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5),
        ])

        self.se = Sequential([
            layers.GlobalAveragePooling2D(),
            CustomLayer_expand_dims(),
            CustomLayer_expand_dims(),
            layers.Conv2D(filters=filters, kernel_size=1, use_bias=False),
            layers.ReLU(),
            layers.Conv2D(filters=filters, kernel_size=1, use_bias=False),
            CustomLayer_sigmoid(),
        ])

    def call(self, x, **kwargs):
        identity = x
        x = self.left(x)
        scale = self.se(x)
        x = x * scale

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = relu(x)
        return x

class SENet(Layer):
    '''
    	SENet, with BasicBlock and BottleneckBlock
    '''

    def __init__(self, filters, blocks, block_type="BottleneckBlock", drop_rate=1, **kwargs):
        super(SENet, self).__init__(**kwargs)

        Layers = [eval(block_type)(filters, drop_rate)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer_layer = eval(block_type)(filters, drop_rate)
            Layers.append(layer_layer)

        self.layers = Sequential(Layers)

    def call(self, x, **kwargs):
        return self.layers(x)



class SENet_decoder(Layer):
    '''
    ResNet, with BasicBlock and BottleneckBlock
    '''

    def __init__(self, filters, blocks, block_type="BottleneckBlock", r=8, drop_rate=2):
        super(SENet_decoder, self).__init__()

        layers = [eval(block_type)(filters, r, 1)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer1 = eval(block_type)(filters, r, 1)
            layers.append(layer1)
            layer2 = eval(block_type)(filters* drop_rate, r, drop_rate)
            filters *= drop_rate
            layers.append(layer2)

        self.layers = Sequential(*layers)

    def call(self, x, **kwargs):
        return self.layers(x)
