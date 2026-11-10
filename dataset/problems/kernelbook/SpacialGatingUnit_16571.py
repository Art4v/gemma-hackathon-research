"""[bf16 variant] Spatial gating unit: chunk -> LayerNorm(z2) -> seq-axis linear (W z2 + b) -> z1 * z2."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpacialGatingUnit_16571',
    'seq_len': 1024,
    'batch': 16,
    'd_z': 2048,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    s, b, d = CONFIG['seq_len'], CONFIG['batch'], CONFIG['d_z']
    k_z, k_w = jax.random.split(jax.random.key(0))
    z = jax.random.uniform(k_z, (s, b, d), dtype)
    weight = jax.random.normal(k_w, (s, s), dtype) * 0.02
    bias = jnp.ones((s,), dtype)
    ln_gamma = jnp.ones((d // 2,), dtype)
    ln_beta = jnp.zeros((d // 2,), dtype)
    return z, weight, bias, ln_gamma, ln_beta


def workload(z, weight, bias, ln_gamma, ln_beta):
    seq_len = z.shape[0]
    z1, z2 = jnp.split(z, 2, axis=-1)
    mean = jnp.mean(z2, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(z2 - mean), axis=-1, keepdims=True)
    z2 = (z2 - mean) * lax.rsqrt(var + CONFIG['eps']) * ln_gamma + ln_beta
    w = weight[:seq_len, :seq_len]
    z2 = jnp.einsum('ij,jbd->ibd', w, z2, precision=lax.Precision.HIGHEST)
    z2 = z2 + bias[:seq_len, None, None]
    return z1 * z2
