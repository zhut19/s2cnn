# pylint: disable=R,C,E1101
import tensorflow as tf
import numpy as np
from functools import lru_cache
from s2cnn.utils.decorator import cached_dirpklgz


@tf.function
def s2_rft(x, b, grid):
    """
    Real Fourier Transform
    :param x: [..., beta_alpha]
    :param b: output bandwidth signal
    :param grid: tuple of (beta, alpha) tuples
    :return: [l * m, ..., complex]
    """
    # F is the Fourier matrix
    F = _setup_s2_ft(b, grid)  # [beta_alpha, l * m]

    assert x.shape[-1] == F.shape[0]

    sz = x.shape
    x = tf.complex(x, tf.zeros_like(x))
    x = tf.einsum("ia,af->fi", tf.reshape(x, (-1, x.shape[-1])), tf.identity(F))  # [l * m, ...]
    x = tf.reshape(x, (-1, *sz[:-1]))
    return x


@cached_dirpklgz("cache/setup_s2_ft")
def __setup_s2_ft(b, grid):
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    # Note: optionally get quadrature weights for the chosen grid and use them to weigh the D matrices below.
    # This is optional because we can also view the filter coefficients as having absorbed the weights already.

    # Sample the Wigner-D functions on the local grid
    n_spatial = len(grid)
    n_spectral = np.sum([(2 * l + 1) for l in range(b)])
    F = np.zeros((n_spatial, n_spectral), dtype=complex)
    for i, (beta, alpha) in enumerate(grid):
        Dmats = [(2 * b) * wigner_D_matrix(l, alpha, beta, 0,
                                           field='complex', normalization='quantum', order='centered', condon_shortley='cs')
                 .conj()
                 for l in range(b)]
        F[i] = np.hstack([Dmats[l][:, l] for l in range(b)])

    # F is a complex matrix of shape (n_spatial, n_spectral)
    # If we view it as float, we get a real matrix of shape (n_spatial, 2 * n_spectral)
    # In the so3_local_ft, we will multiply a batch of real (..., n_spatial) vectors x with this matrix F as xF.
    # The result is a (..., 2 * n_spectral) array that can be interpreted as a batch of complex vectors.
    # F = F.view('float').reshape((-1, n_spectral, 2))
    return F


@tf.function
def _setup_s2_ft(b, grid):
    F = __setup_s2_ft(b, grid)

    # convert to Tensor
    F = tf.convert_to_tensor(F, dtype=tf.complex64)  # pylint: disable=E1102

    return F
