"""[bf16 variant] AlphaChooser: clamp(1 + sigmoid(pre_alpha), 1.01, 2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AlphaChooser_16276',
    'head_count': 33554432,
    'alpha_min': 1.01,
    'alpha_max': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    # Unscaled normal to match torch.randn init, so the min clamp is exercised.
    pre_alpha = jax.random.normal(key, (CONFIG['head_count'],), dtype)
    return (pre_alpha,)


def workload(pre_alpha):
    alpha = 1 + jax.nn.sigmoid(pre_alpha)
    return jnp.clip(alpha, CONFIG['alpha_min'], CONFIG['alpha_max'])
