"""[bf16 variant] AlphaVectorMultiplication: x * sigmoid(alpha expanded to (batch, size_alpha))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AlphaVectorMultiplication_13791',
    'batch': 16,
    'channels': 256,
    'size_alpha': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_a = jax.random.split(jax.random.key(0))
    b, c, a = CONFIG['batch'], CONFIG['channels'], CONFIG['size_alpha']
    # Axis 2 must equal batch: the expanded (batch, size_alpha) alpha broadcasts onto x's last two axes.
    x = jax.random.uniform(k_x, (b, c, b, a), dtype)
    alpha = jax.random.normal(k_a, (1, a), dtype) * 0.02
    return x, alpha


def workload(x, alpha):
    bsz = x.shape[0]
    gate = jax.nn.sigmoid(jnp.broadcast_to(alpha, (bsz, alpha.shape[1])))
    return x * gate
