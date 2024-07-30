import tensorflow as tf
from tensorflow.keras import layers, Model

from subspectralnorm import SubSpectralNorm

class ConvBNReLU(tf.keras.layers.Layer):
    def __init__(self, in_plane, out_plane, idx, kernel_size=3, stride=1, groups=1, use_dilation=False, activation=True, swish=False, BN=True, ssn=False):
        super(ConvBNReLU, self).__init__()
        self.idx = idx

        def get_padding(kernel_size, use_dilation):
            rate = 1
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2 ** self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        padding, rate = get_padding(kernel_size, use_dilation)

        self.layers_list = [
            layers.Conv2D(out_plane, kernel_size, stride, padding='same', dilation_rate=rate, groups=groups, use_bias=False)
        ]

        if ssn:
            self.layers_list.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            self.layers_list.append(layers.BatchNormalization())

        if swish:
            self.layers_list.append(layers.Activation('swish'))
        elif activation:
            self.layers_list.append(layers.ReLU())

        self.block = tf.keras.Sequential(self.layers_list)

    def call(self, x):
        return self.block(x)


class BCResBlock(Model):
    def __init__(self, in_plane, out_plane, idx, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane

        kernel_size = (3, 3)

        layers_list = []
        if self.transition_block:
            layers_list.append(ConvBNReLU(in_plane, out_plane, idx, kernel_size=1, stride=1))
            in_plane = out_plane

        layers_list.append(ConvBNReLU(in_plane, out_plane, idx, kernel_size=(kernel_size[0], 1), stride=(stride[0], 1), groups=in_plane, ssn=True, activation=False))
        self.f2 = tf.keras.Sequential(layers_list)
        self.avg_gpool = layers.GlobalAveragePooling2D()

        self.f1 = tf.keras.Sequential([
            ConvBNReLU(out_plane, out_plane, idx, kernel_size=(1, kernel_size[1]), stride=(1, stride[1]), groups=out_plane, swish=True, use_dilation=True),
            layers.Conv2D(out_plane, 1, use_bias=False),
            layers.Dropout(0.1),
        ])

    def call(self, x):
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = tf.nn.relu(x)
        return x

def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride):
    stages = []
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stages.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stages

class BCResNets(Model):
    def __init__(self, base_c, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.n = [2, 2, 4, 4]
        self.c = [base_c * 2, base_c, int(base_c * 1.5), base_c * 2, int(base_c * 2.5), base_c * 4]
        self.s = [1, 2]
        self.build_network()

    def build_network(self):
        self.cnn_head = tf.keras.Sequential([
            layers.Conv2D(self.c[0], 5, strides=(2, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        self.BCBlocks = []
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        self.classifier = tf.keras.Sequential([
            layers.Conv2D(self.c[-2], (5, 5), groups=self.c[-2], padding='same', use_bias=False),
            layers.Conv2D(self.c[-1], 1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.num_classes),
        ])

    def call(self, x):
        x = self.cnn_head(x)
        for block_stage in self.BCBlocks:
            for block in block_stage:
                x = block(x)
        x = self.classifier(x)
        return x
