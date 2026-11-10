"""PAM position attention: 1x1 conv Q/K/V -> bmm(Q^T, K) -> softmax -> bmm(V, A^T) -> gamma * out + x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PAM_Module_4494',
    'batch': 8,
    'channels': 512,
    'height': 64,
    'width': 64,
    'gamma': 1.0,
}


def create_inputs(dtype=jnp.float32):
    b, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    kx, kq, kk, kv = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (b, c, h, w), dtype)
    wq = jax.random.normal(kq, (1, 1, c, c), dtype) * 0.02
    wk = jax.random.normal(kk, (1, 1, c, c), dtype) * 0.02
    wv = jax.random.normal(kv, (1, 1, c, c), dtype) * 0.02
    bq = jnp.zeros((c,), dtype)
    bk = jnp.zeros((c,), dtype)
    bv = jnp.zeros((c,), dtype)
    # torch inits gamma to 0 (output == x); use a nonzero constant so the attention path matters.
    gamma = jnp.full((1,), CONFIG['gamma'], dtype)
    return x, wq, bq, wk, bk, wv, bv, gamma


def _conv1x1(x_nhwc, w_hwio, b):
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, wq, bq, wk, bk, wv, bv, gamma):
    bsz, c, h, w = x.shape
    n = h * w
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    q = _conv1x1(x_nhwc, wq, bq).reshape(bsz, n, c)
    k = _conv1x1(x_nhwc, wk, bk).reshape(bsz, n, c)
    v = _conv1x1(x_nhwc, wv, bv).reshape(bsz, n, c)
    energy = jnp.einsum('bnc,bmc->bnm', q, k)
    attention = jax.nn.softmax(energy, axis=-1)
    out = jnp.einsum('bnm,bmc->bnc', attention, v)
    out = jnp.transpose(out.reshape(bsz, h, w, c), (0, 3, 1, 2))
    return gamma[0] * out + x
