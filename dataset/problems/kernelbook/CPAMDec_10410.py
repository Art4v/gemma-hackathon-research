"""[bf16 variant] CPAMDec: 1x1 conv query + linear key/value on centers -> softmax attention over K -> bmm -> scale * out + x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CPAMDec_10410',
    'batch': 8,
    'in_channels': 512,
    'height': 64,
    'width': 64,
    'num_centers': 256,
    'reduction': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    n = CONFIG['batch']
    c = CONFIG['in_channels']
    h = CONFIG['height']
    w = CONFIG['width']
    k = CONFIG['num_centers']
    cq = c // CONFIG['reduction']
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(keys[0], (n, c, h, w), dtype)
    y = jax.random.uniform(keys[1], (n, k, c), dtype)
    wq = jax.random.normal(keys[2], (1, 1, c, cq), dtype) * 0.02
    bq = jnp.zeros((cq,), dtype)
    wk = jax.random.normal(keys[3], (c, cq), dtype) * 0.02
    bk = jnp.zeros((cq,), dtype)
    wv = jax.random.normal(keys[4], (c, c), dtype) * 0.02
    bv = jnp.zeros((c,), dtype)
    # ponytail: torch inits scale to 0 (attention path dead); ones keeps the reference meaningful.
    scale = jnp.ones((1,), dtype)
    return x, y, wq, bq, wk, bk, wv, bv, scale


def workload(x, y, wq, bq, wk, bk, wv, bv, scale):
    n, c, h, w = x.shape
    xh = jnp.transpose(x, (0, 2, 3, 1))
    q = lax.conv_general_dilated(
        xh, wq, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + bq
    q = q.reshape(n, h * w, -1)                     # (N, HW, C/4)
    k = y @ wk + bk                                 # (N, K, C/4)
    energy = jnp.einsum('npc,nkc->npk', q, k)       # (N, HW, K)
    attention = jax.nn.softmax(energy, axis=-1)
    v = y @ wv + bv                                 # (N, K, C)
    out = jnp.einsum('nkc,npk->ncp', v, attention)  # (N, C, HW)
    out = out.reshape(n, c, h, w)
    return scale * out + x
