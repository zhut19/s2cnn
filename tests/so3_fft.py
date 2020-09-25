# pylint: disable=C,R,E1101,E1102,W0621
'''
Compare so3_ft with so3_fft
'''
import tensorflow as tf
from functools import partial

def test_so3_rfft(b_in, b_out):
    x = tf.random.normal((1, 1, 2 * b_in, 2 * b_in, 2 * b_in),
    dtype=tf.float32)  # [beta, alpha, gamma]

    from s2cnn.soft.so3_fft import SO3_fft_real
    y1 = SO3_fft_real.forward(x, b_out=b_out)
    y1 = tf.squeeze(y1, [-2, -1])

    from s2cnn import so3_rft, so3_soft_grid
    import lie_learn.spaces.S3 as S3

    # so3_ft computes a non weighted Fourier transform
    weights = tf.convert_to_tensor(S3.quadrature_weights(b_in), dtype=tf.float32)
    x = tf.squeeze(x, [0, 1])
    x2 = tf.einsum("bac,b->bac", x, weights)

    y2 = so3_rft(tf.reshape(x2, (-1)), b_out, so3_soft_grid(b_in))
    assert abs((y1 - y2).numpy()).max() < 1e-4 * abs(y1.numpy()).mean()

test_so3_rfft(7, 5)
# test_so3_rfft(5, 7, torch.device("cpu"))  # so3_rft introduce aliasing




def test_inverse(f, g, b_in, b_out):
    x = tf.random.normal((1, 1, 2 * b_in, 2 * b_in, 2 * b_in),
        dtype=tf.float32)  # [beta, alpha, gamma]
    x = tf.complex(x, tf.zeros_like(x))

    x = g(f(x, b_out=b_out), b_out=b_in)

    y = g(f(x, b_out=b_out), b_out=b_in)

    assert abs((x - y).numpy()).max() < 1e-4 * abs(y.numpy()).mean()


def test_inverse2(f, g, b_in, b_out):
    x = tf.random.normal(((b_in * (4 * b_in**2 - 1) // 3), 1, 1),
        dtype=tf.float32)  # [beta, alpha, gamma]
    x = tf.complex(x, tf.zeros_like(x))

    x = g(f(x, b_out=b_out), b_out=b_in)

    y = g(f(x, b_out=b_out), b_out=b_in)

    assert abs((x - y).numpy()).max() < 1e-4 * abs(y.numpy()).mean()


from s2cnn.soft.so3_fft import so3_fft, so3_ifft
test_inverse(so3_fft, so3_ifft, 7, 7)
test_inverse(so3_fft, so3_ifft, 5, 4)
test_inverse(so3_fft, so3_ifft, 7, 4)

test_inverse2(so3_ifft, so3_fft, 7, 7)
test_inverse2(so3_ifft, so3_fft, 5, 5)
test_inverse2(so3_ifft, so3_fft, 4, 7)
