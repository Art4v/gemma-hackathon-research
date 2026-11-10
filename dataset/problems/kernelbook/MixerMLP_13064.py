"""[bf16 variant] MixerMLP_13064: swapaxes(1,2) -> LayerNorm -> Linear -> GELU (exact) -> Linear -> swapaxes(1,2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MixerMLP_13064',
    'batch': 8,
    'channels': 128,
    'patches': 128,
    'dim': 512,
    'eps': 1e-5,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    dim = CONFIG['dim']
    scale = CONFIG['weight_scale']
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['channels'], CONFIG['patches'], dim), dtype
    )
    ln_gamma = jnp.ones((dim,), dtype)
    ln_beta = jnp.zeros((dim,), dtype)
    fc1_w = jax.random.normal(k_w1, (dim, dim), dtype) * scale
    fc1_b = jax.random.normal(k_b1, (dim,), dtype) * scale
    fc2_w = jax.random.normal(k_w2, (dim, dim), dtype) * scale
    fc2_b = jax.random.normal(k_b2, (dim,), dtype) * scale
    return x, ln_gamma, ln_beta, fc1_w, fc1_b, fc2_w, fc2_b


def workload(x, ln_gamma, ln_beta, fc1_w, fc1_b, fc2_w, fc2_b):
    x = jnp.swapaxes(x, 1, 2)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + CONFIG['eps']) * ln_gamma + ln_beta
    x = jax.nn.gelu(x @ fc1_w + fc1_b, approximate=False)
    x = x @ fc2_w + fc2_b
    return jnp.swapaxes(x, 1, 2)
