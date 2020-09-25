# pylint: disable=C,R,E1101
import math
import tensorflow as tf

from .s2_fft import S2_fft_real
from .so3_fft import SO3_ifft_real
from s2cnn import s2_mm
from s2cnn import s2_rft


class S2Convolution(tf.keras.layers.Layer):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid, **kwarg):
        '''
        :param nfeature_in: number of input fearures
        :param nfeature_out: number of output features
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        :param grid: points of the sphere defining the kernel, tuple of (alpha, beta)'s
        '''
        super().__init__(**kwarg)
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in = b_in
        self.b_out = b_out
        self.grid = grid
        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 4.) / (self.b_in ** 2.))

    def build(self, input_shape):
        kernel_init = tf.random_uniform_initializer(minval=-1, maxval=1)
        self.kernel = self.add_weight(
            shape=(self.nfeature_in, self.nfeature_out, len(self.grid)),
            initializer=kernel_init,
            trainable=True)

        self.bias = self.add_weight(
            shape=(1, self.nfeature_out, 1, 1, 1),
            initializer='zeros',
            trainable=True)
        super().build(input_shape)

    def call(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        assert x.shape[1] == self.nfeature_in
        assert x.shape[2] == 2 * self.b_in
        assert x.shape[3] == 2 * self.b_in
        x = S2_fft_real.forward(x, self.b_out)  # [l * m, batch, feature_in]
        y = s2_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m, feature_in, feature_out]
        x = s2_mm(x, y)  # [l * m * n, batch, feature_out]
        x = SO3_ifft_real.forward(x)  # [batch, feature_out, beta, alpha, gamma]

        x = x + self.bias

        return x
