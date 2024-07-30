import tensorflow as tf
from tensorflow.keras import layers

class SubSpectralNorm(tf.keras.layers.Layer):
    def __init__(self, num_features, spec_groups=16, affine="Sub", batch=True, dim=2):
        super(SubSpectralNorm, self).__init__()
        self.spec_groups = spec_groups
        self.affine_all = False
        self.sub_dim = dim
        self.num_features = num_features

        if affine == "Sub":
            self.affine_norm = True
        elif affine == "All":
            self.affine_all = True
            self.weight = self.add_weight(
                shape=(1, num_features, 1, 1),
                initializer='ones',
                trainable=True,
                name='weight'
            )
            self.bias = self.add_weight(
                shape=(1, num_features, 1, 1),
                initializer='zeros',
                trainable=True,
                name='bias'
            )

        if batch:
            self.ssnorm = layers.BatchNormalization(center=self.affine_norm, scale=self.affine_norm)
        else:
            self.ssnorm = layers.LayerNormalization(center=self.affine_norm, scale=self.affine_norm)

    def call(self, x, training=False):
        if self.sub_dim in (3, -1):
            x = tf.transpose(x, perm=[0, 1, 3, 2])

        b, c, h, w = tf.shape(x)
        assert h % self.spec_groups == 0

        x = tf.reshape(x, [b, c * self.spec_groups, h // self.spec_groups, w])
        x = self.ssnorm(x, training=training)
        x = tf.reshape(x, [b, c, h, w])

        if self.affine_all:
            x = x * self.weight + self.bias

        if self.sub_dim in (3, -1):
            x = tf.transpose(x, perm=[0, 1, 3, 2])

        return x
