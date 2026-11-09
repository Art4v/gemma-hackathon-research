"""[bf16 variant] FCUDown: 1x1 Conv2d -> AvgPool2d -> flatten/transpose -> LayerNorm -> GELU -> concat cls token from x_t."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FCUDown_16471',
    'batch': 8,
    'c1': 512,
    'c2': 512,
    'height': 64,
    'width': 64,
    'dw_stride': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    s = c['dw_stride']
    n_tokens = (c['height'] // s) * (c['width'] // s) + 1
    k_x, k_t, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(k_x, (c['batch'], c['c1'], c['height'], c['width']), dtype)
    x_t = jax.random.uniform(k_t, (c['batch'], n_tokens, c['c2']), dtype)
    w = jax.random.normal(k_w, (c['c1'], c['c2']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['c2'],), dtype) * 0.02
    gamma = jnp.ones((c['c2'],), dtype)
    beta = jnp.zeros((c['c2'],), dtype)
    return x, x_t, w, b, gamma, beta


def workload(x, x_t, w, b, gamma, beta):
    s = CONFIG['dw_stride']
    # ponytail: 1x1 stride-1 conv is a channel einsum; rank-generic so unbatched (C,H,W) matches torch too.
    y = jnp.einsum('...chw,co->...ohw', x, w, precision=lax.Precision.HIGHEST)
    y = y + b[:, None, None]
    lead = (1,) * (y.ndim - 2)
    y = lax.reduce_window(y, 0.0, lax.add, lead + (s, s), lead + (s, s), 'VALID') / (s * s)
    y = y.reshape(y.shape[:2] + (-1,))  # torch flatten(2)
    y = jnp.swapaxes(y, 1, 2)
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=-1, keepdims=True)
    y = (y - mean) * lax.rsqrt(var + CONFIG['eps']) * gamma + beta
    y = jax.nn.gelu(y, approximate=False)
    return jnp.concatenate([x_t[:, :1, :], y], axis=1)
