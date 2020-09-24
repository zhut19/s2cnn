# pylint: disable=R,C,E1101,E1102
from functools import lru_cache
import tensorflow as tf
from s2cnn.utils.decorator import cached_dirpklgz


# inspired by https://gist.github.com/szagoruyko/89f83b6f5f4833d3c8adf81ee49f22a8
# modified from https://github.com/jonas-koehler/s2cnn/blob/d000c925f343427e612bae8f69ef5cf31bdf8a5b


@tf.function
def s2_fft(x, for_grad=False, b_out=None):
    '''
    :param x: [batch, feature_in, beta, alpha]
    :return:  [l * m, batch, feature_in]
    '''
    b_in = x.shape[-1] // 2
    assert x.shape[-1] == 2 * b_in
    assert x.shape[-2] == 2 * b_in
    if b_out is None:
        b_out = b_in
    nfeature_in = x.shape[1]

    x = tf.reshape(x, (-1, 2 * b_in, 2 * b_in))  # [batch, beta, alpha]

    '''
    :param x: [batch, beta, alpha] (nbatch, 2 * b_in, 2 * b_in)
    :return: [l * m, batch] (b_out**2, nbatch)
    '''
    nspec = b_out ** 2

    wigner = _setup_s2_wigner(b_in, b_out, not for_grad)  # [beta, l * m ]
    assert wigner.shape[0] == 2 * b_in
    assert wigner.shape[1] == nspec

    x = tf.signal.fft(x)  # [batch, beta, m]
    output_list = []
    for l in range(b_out):
        s = slice(l ** 2, l ** 2 + 2 * l + 1)
        l1 = min(l, b_in)

        rx = tf.roll(x, shift=l1, axis=-1)
        if l1 == b_in:
            rx = tf.pad(rx, [[0, 0], [0, 0], [0, 1]], 'CONSTANT')

        xx = rx[:, :, :2 * l1 + 1]
        xx = tf.pad(xx, [[0, 0], [0, 0], [l-l1, l-l1]], 'CONSTANT')
        output_list.append(tf.einsum('bm,zbm->mz', wigner[:, s], xx))

    output = tf.concat(output_list, axis=0)
    output = tf.reshape(output, (nspec, -1, nfeature_in))

    return output


@tf.function
def s2_ifft(x, for_grad=True, b_out=None):
    '''
    :param x: [l * m, batch, feature_out]
    '''
    nspec = x.shape[0]
    b_in = round(nspec ** 0.5)
    assert nspec == b_in ** 2
    if b_out is None:
        b_out = b_in
    nfeature_out = x.shape[-1]

    x = tf.reshape(x, (nspec, -1))  # [l * m, batch] (nspec, nbatch)

    '''
    :param x: [l * m, batch] (b_in**2, nbatch)
    :return: [batch, beta, alpha] (nbatch, 2 * b_out, 2 * b_out)
    '''
    wigner = _setup_s2_wigner(b_out, b_in, not for_grad)
    assert wigner.shape[0] == 2 * b_out
    assert wigner.shape[1] == nspec

    output_list = []
    for l in range(b_in):
        s = slice(l ** 2, l ** 2 + 2 * l + 1)
        out = tf.einsum("mz,bm->zbm", x[s], wigner[:, s])
        l1 = min(l, b_out)
        if l < b_out:
            out = tf.pad(out, [[0, 0,], [0, 0], [0, 2 * b_out - 2 * l - 1]], "CONSTANT")
        else:
            out = out[:, :, l-b_out: l+b_out]

        out = tf.roll(out, shift=-l1, axis=-1)
        output_list.append(out)

    output = tf.stack(output_list)
    output = tf.math.reduce_sum(output, axis=0)
    output = tf.signal.ifft(output) * output.shape[-1] # [batch, beta, alpha]
    output = tf.reshape(output, (-1, nfeature_out, 2 * b_out, 2 * b_out))
    return output


@tf.function
def _setup_s2_wigner(b, nl, weighted):
    dss = _setup_s2_fft(b, nl, weighted)
    dss = tf.convert_to_tensor(dss, dtype=tf.complex64)  # [beta, l * m] # pylint: disable=E1102
    return dss


@cached_dirpklgz("cache/setup_s2_fft")
def _setup_s2_fft(b, nl, weighted):
    from lie_learn.representations.SO3.wigner_d import wigner_d_matrix
    import lie_learn.spaces.S3 as S3
    import numpy as np
    import logging

    betas = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    w = S3.quadrature_weights(b) * 2 * b
    assert len(w) == len(betas)

    logging.getLogger("trainer").info("Compute Wigner (only columns): b=%d nbeta=%d nl=%d nspec=%d", b, len(betas), nl,
                                      nl ** 2)

    dss = []
    for b, beta in enumerate(betas):
        ds = []
        for l in range(nl):
            d = wigner_d_matrix(l, beta,
                                field='complex', normalization='quantum', order='centered', condon_shortley='cs')
            d = d[:, l]  # d[m=:, n=0]

            if weighted:
                d *= w[b]
            else:
                d *= 2 * l + 1

            ds.append(d)  # [m]
        dss.append(np.concatenate(ds))  # [l * m]

    dss = np.stack(dss)  # [beta, l * m]
    return dss


class S2_fft_real():
    @staticmethod
    @tf.function
    def forward(x, b_out=None):
        x = tf.complex(x, tf.zeros_like(x))
        return s2_fft(x, b_out=b_out)


class S2_ifft_real():
    @staticmethod
    @tf.function
    def forward(x, b_out=None):
        return tf.math.real(s2_ifft(x, b_out=b_out))
