# pylint: disable=R,C,E1101
import math
from functools import lru_cache
import tensorflow as tf


@tf.function
def so3_mm(x, y):
    '''
    :param x: [l * m * n,   batch,    feature_in]
    :param y: [l * m * n, feature_in, feature_out]
    :return:  [l * m * n,   batch,    feature_out]
    '''
    nfeature_in = x.shape[2]
    nfeature_out = y.shape[2]
    assert y.shape[1] == nfeature_in
    nspec = x.shape[0]
    assert y.shape[0] == nspec
    nl = round((3 / 4 * nspec) ** (1 / 3))
    assert nspec == nl * (4 * nl ** 2 - 1) // 3

    Fz_list = []
    begin = 0
    for l in range(nl):
        L = 2 * l + 1
        size = L ** 2

        Fx = x[begin:begin + size]  # [m * n,   batch,    feature_in]
        Fy = y[begin:begin + size]  # [m * n, feature_in, feature_out]

        Fx = tf.reshape(Fx, (L, L, -1, nfeature_in))  # [m, n, batch, feature_in]
        Fx = tf.transpose(Fx, (2, 0, 3, 1))  # [batch, m, feature_in, n]
        Fx = tf.reshape(Fx, (-1, nfeature_in * L))  # [batch * m, feature_in * n]

        Fy = tf.reshape(Fy, (L, L, nfeature_in, nfeature_out))  # [m, n, feature_in, feature_out]
        Fy = tf.transpose(Fy, (2, 1, 0, 3))  # [feature_in, n, m, feature_out]
        Fy = tf.reshape(Fy, (nfeature_in * L, L * nfeature_out))  # [feature_in * n, m * feature_out]

        Fy = tf.math.conj(Fy)
        Fz = tf.matmul(Fx, Fy)  # [batch * m_x, m_y * feature_out, complex] m_x -> m, m_y -> n
        Fz = tf.reshape(Fz, (-1, L * L, nfeature_out))  # [batch, m * n, feature_out]
        Fz = tf.transpose(Fz, (1, 0, 2))  # [m * n, batch, feature_out]

        Fz_list.append(Fz)

        begin += size

    z = tf.concat(Fz_list, 0)  # [l * m * n, batch, feature_out]
    return z
