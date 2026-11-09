"""[bf16 variant] fChannelAttention: spatial mean/max pool -> shared 2-layer MLP (no bias, ReLU) -> sum -> sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'fChannelAttention_15287',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'ratio': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    b, c = CONFIG['batch'], CONFIG['channels']
    hidden = c // CONFIG['ratio']
    x = jax.random.uniform(k_x, (b, c, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k_w1, (c, hidden), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (hidden, c), dtype) * 0.02
    return x, w1, w2


def workload(x, w1, w2):
    b, c = x.shape[0], x.shape[1]
    avg = jnp.mean(x, axis=(-2, -1))
    mx = jnp.max(x, axis=(-2, -1))

    def mlp(v):
        h = jax.nn.relu(jnp.matmul(v, w1, precision=lax.Precision.HIGHEST))
        return jnp.matmul(h, w2, precision=lax.Precision.HIGHEST)

    out = jax.nn.sigmoid(mlp(avg) + mlp(mx))
    return out.reshape(b, c, 1, 1)
