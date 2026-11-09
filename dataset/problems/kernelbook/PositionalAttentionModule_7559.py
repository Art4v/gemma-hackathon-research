"""[bf16 variant] Positional attention: three 1x1 convs -> softmax(C^T B) -> D @ S^T -> alpha * DS + A."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PositionalAttentionModule_7559',
    'batch': 8,
    'channels': 512,
    'height': 64,
    'width': 64,
    'alpha': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c = CONFIG['batch'], CONFIG['channels']
    h, w = CONFIG['height'], CONFIG['width']
    keys = jax.random.split(jax.random.key(0), 7)
    a = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    w_b = jax.random.normal(keys[1], (c, c), dtype) * 0.02
    b_b = jax.random.normal(keys[2], (c,), dtype) * 0.02
    w_c = jax.random.normal(keys[3], (c, c), dtype) * 0.02
    b_c = jax.random.normal(keys[4], (c,), dtype) * 0.02
    w_d = jax.random.normal(keys[5], (c, c), dtype) * 0.02
    b_d = jax.random.normal(keys[6], (c,), dtype) * 0.02
    # torch inits alpha to 0, which would make E == A; nonzero so the attention path is exercised.
    alpha = jnp.full((1,), CONFIG['alpha'], dtype)
    return a, w_b, b_b, w_c, b_c, w_d, b_d, alpha


def _conv1x1(x, w, bias):
    # x: (b, c_in, n), w: (c_in, c_out) -> (b, c_out, n)
    return jnp.einsum('bcn,co->bon', x, w) + bias[None, :, None]


def workload(a, w_b, b_b, w_c, b_c, w_d, b_d, alpha):
    bsz, ch, h, w = a.shape
    x = a.reshape(bsz, ch, h * w)
    b_feat = _conv1x1(x, w_b, b_b)
    c_feat = _conv1x1(x, w_c, b_c)
    d_feat = _conv1x1(x, w_d, b_d)
    s = jax.nn.softmax(jnp.einsum('bci,bcj->bij', c_feat, b_feat), axis=-1)
    ds = jnp.einsum('bcj,bij->bci', d_feat, s).reshape(bsz, ch, h, w)
    return alpha[0] * ds + a
