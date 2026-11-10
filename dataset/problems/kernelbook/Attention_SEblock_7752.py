"""[bf16 variant] Global average pool -> Linear -> ReLU -> Linear -> hard Gumbel-softmax (SE-style gate)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Attention_SEblock_7752',
    'batch': 8,
    'channels': 512,
    'height': 128,
    'width': 128,
    'reduction': 4,
    'num_outputs': 2,
    'tau': 1.0,
    'fc2_bias': (0.1, 2.0),
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    batch = CONFIG['batch']
    channels = CONFIG['channels']
    hidden = channels // CONFIG['reduction']
    num_outputs = CONFIG['num_outputs']
    scale = CONFIG['weight_scale']

    key = jax.random.key(0)
    k_x, k_w1, k_b1, k_w2, k_g = jax.random.split(key, 5)

    x = jax.random.uniform(k_x, (batch, channels, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k_w1, (channels, hidden), dtype) * scale
    b1 = jax.random.normal(k_b1, (hidden,), dtype) * scale
    w2 = jax.random.normal(k_w2, (hidden, num_outputs), dtype) * scale
    b2 = jnp.array(CONFIG['fc2_bias'], dtype)
    gumbels = jax.random.gumbel(k_g, (batch, num_outputs), dtype)
    return x, w1, b1, w2, b2, gumbels


def workload(x, w1, b1, w2, b2, gumbels):
    pooled = jnp.mean(x, axis=(2, 3)).reshape(-1, w1.shape[0])
    h = jnp.maximum(pooled @ w1 + b1, 0.0)
    logits = h @ w2 + b2

    y_soft = jax.nn.softmax((logits + gumbels) / CONFIG['tau'], axis=-1)
    index = jnp.argmax(y_soft, axis=-1)
    y_hard = jax.nn.one_hot(index, y_soft.shape[-1], dtype=y_soft.dtype)
    return y_hard - lax.stop_gradient(y_soft) + y_soft
