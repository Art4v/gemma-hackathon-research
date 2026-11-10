"""scSE: x*sigmoid(conv1x1(x)) + x*sigmoid(conv1x1(relu6(conv1x1(avgpool(x)))))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'scSE_3996',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'reduced': 64,  # max(1, channels // 2)
    'relu6_max': 6.0,
}


def create_inputs(dtype=jnp.float32):
    n, c, h, w, r = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'],
                     CONFIG['width'], CONFIG['reduced'])
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(ks[0], (n, c, h, w), dtype)
    ws = jax.random.normal(ks[1], (c,), dtype) * 0.02
    bs = jax.random.normal(ks[2], (1,), dtype) * 0.02
    w1 = jax.random.normal(ks[3], (c, r), dtype) * 0.02
    b1 = jax.random.normal(ks[4], (r,), dtype) * 0.02
    w2 = jax.random.normal(ks[5], (r, c), dtype) * 0.02
    b2 = jax.random.normal(ks[6], (c,), dtype) * 0.02
    return x, ws, bs, w1, b1, w2, b2


def workload(x, ws, bs, w1, b1, w2, b2):
    # sSE: 1x1 conv C->1 is a channel contraction.
    s = jnp.einsum('nchw,c->nhw', x, ws, precision='highest') + bs[0]
    s = jax.nn.sigmoid(s)[:, None, :, :]
    # cSE: global avg pool, 1x1 convs on a (N, C) vector are matmuls.
    pooled = jnp.mean(x, axis=(2, 3))
    hid = jnp.clip(jnp.matmul(pooled, w1, precision='highest') + b1,
                   0.0, CONFIG['relu6_max'])
    c = jax.nn.sigmoid(jnp.matmul(hid, w2, precision='highest') + b2)
    c = c[:, :, None, None]
    return x * s + x * c
