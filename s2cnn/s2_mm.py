# pylint: disable=R,C,E1101
from functools import lru_cache
import tensorflow as tf
from string import Template

# TODO simplify the cuda code like it was done in SO3_mm using only one code for the kernel


@tf.function
def s2_mm(x, y):
    '''
    :param x: [l * m,     batch,      feature_in,  complex]
    :param y: [l * m,     feature_in, feature_out, complex]
    :return:  [l * m * n, batch,      feature_out, complex]
    '''
    nfeature_in = x.shape[2]
    nfeature_out = y.shape[2]
    assert y.shape[1] == nfeature_in
    nspec = x.shape[0]
    assert y.shape[0] == nspec

    nl = round(nspec**0.5)

    Fz_list = []
    begin = 0
    for l in range(nl):
        L = 2 * l + 1
        size = L

        Fx = x[begin:begin+size]  # [m, batch,      feature_in]
        Fy = y[begin:begin+size]  # [m, feature_in, feature_out]

        Fx = tf.reshape(Fx, (-1, nfeature_in))  # [m * batch, feature_in]

        Fy = tf.transpose(Fy, (1, 0, 2))  # [feature_in, m, feature_out]
        Fy = tf.reshape(Fy, (nfeature_in, -1))  # [feature_in, m * feature_out]

        Fy = tf.math.conj(Fy)
        Fz = tf.matmul(Fx, Fy)  # [m_x * batch, m_y * feature_out] m_x -> m, m_y -> n

        Fz = tf.reshape(Fz, (L, -1, L, nfeature_out))  # [m, batch, n, feature_out]
        Fz = tf.transpose(Fz, (0, 2, 1, 3))  # [m, n, batch, feature_out]
        Fz = tf.reshape(Fz, (L * L, -1, nfeature_out))  # [m * n, batch, feature_out]

        Fz_list.append(Fz)

        begin += size

    z = tf.concat(Fz_list, axis=0)  # [l * m * n, batch, feature_out]
    return z
