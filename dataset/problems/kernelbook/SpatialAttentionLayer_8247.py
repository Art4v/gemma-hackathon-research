"""[bf16 variant] SpatialAttentionLayer: channel max/mean -> width/height softmax-gated tanh MLPs -> outer-product tanh attention -> x * (att + 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SpatialAttentionLayer_8247',
    'batch': 8,
    'channels': 128,
    'spatial_size': 256,
}

_HI = jax.lax.Precision.HIGHEST


def create_inputs(dtype=jnp.bfloat16):
    b, c, s = CONFIG['batch'], CONFIG['channels'], CONFIG['spatial_size']
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (b, c, s, s), dtype)
    ww = [jax.random.normal(keys[1 + i], (s, 1), dtype) * 0.02 for i in range(3)]
    hw = [jax.random.normal(keys[4 + i], (s, 1), dtype) * 0.02 for i in range(3)]
    wb = [jnp.zeros((1, s), dtype) for _ in range(3)]
    hb = [jnp.zeros((1, s), dtype) for _ in range(3)]
    return (x, ww[0], ww[1], ww[2], wb[0], wb[1], wb[2],
            hw[0], hw[1], hw[2], hb[0], hb[1], hb[2])


def _branch(v_mean, v_max, w0, w1, w2, b0, b1, b2):
    s = jax.nn.softmax(v_mean, axis=1)
    y0 = jnp.matmul(v_mean, w0, precision=_HI)
    y1 = jnp.matmul(v_max, w1, precision=_HI)
    y0_t = jnp.tanh(y0 * s + b0)
    y1_t = jnp.tanh(y1 * s + b1)
    y2 = jnp.matmul(y1_t, w2, precision=_HI)
    return y2 * y0_t + b2


def workload(x, ww0, ww1, ww2, wb0, wb1, wb2, hw0, hw1, hw2, hb0, hb1, hb2):
    sp_max = jnp.max(x, axis=1)
    sp_mean = jnp.mean(x, axis=1)
    y_w = _branch(jnp.mean(sp_mean, axis=1), jnp.max(sp_max, axis=1),
                  ww0, ww1, ww2, wb0, wb1, wb2)
    y_h = _branch(jnp.mean(sp_mean, axis=2), jnp.max(sp_max, axis=2),
                  hw0, hw1, hw2, hb0, hb1, hb2)
    spatial = jnp.tanh(y_h[:, :, None] * y_w[:, None, :])[:, None, :, :]
    return x * (spatial + 1)
