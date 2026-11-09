"""[bf16 variant] NIN4d: weight-normalized channel einsum (bc...,oc->bo...) plus 5D bias broadcast."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NIN4d_14612',
    'batch': 8,
    'in_features': 128,
    'out_features': 16,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_v = jax.random.split(jax.random.key(0))
    b, c, o = CONFIG['batch'], CONFIG['in_features'], CONFIG['out_features']
    x = jax.random.uniform(k_x, (b, c, CONFIG['height'], CONFIG['width']), dtype)
    weight_v = jax.random.normal(k_v, (o, c), dtype) * 0.02
    weight_g = jnp.ones((o, 1), dtype)
    bias = jnp.zeros((o, 1, 1, 1, 1), dtype)
    return x, weight_v, weight_g, bias


def workload(x, weight_v, weight_g, bias):
    v_norm = jnp.sqrt(jnp.sum(weight_v * weight_v, axis=1, keepdims=True))
    weight = weight_v * (weight_g / v_norm)
    out = jnp.einsum('bc...,oc->bo...', x, weight, precision=lax.Precision.HIGHEST)
    # ponytail: 5D bias on a 4D output broadcasts to (O,B,O,H,W), same as PyTorch.
    return out + bias
