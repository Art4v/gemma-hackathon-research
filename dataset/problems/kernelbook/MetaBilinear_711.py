"""[bf16 variant] Bilinear: y = x1^T W x2 + b over the last feature axis (F.bilinear)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MetaBilinear_711',
    'lead_shape': (8, 2, 128),
    'in1_features': 256,
    'in2_features': 256,
    'out_features': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    lead = CONFIG['lead_shape']
    in1, in2, out = CONFIG['in1_features'], CONFIG['in2_features'], CONFIG['out_features']
    x1 = jax.random.uniform(k1, (*lead, in1), dtype)
    x2 = jax.random.uniform(k2, (*lead, in2), dtype)
    w = jax.random.normal(k3, (out, in1, in2), dtype) * 0.02
    b = jax.random.normal(k4, (out,), dtype) * 0.02
    return x1, x2, w, b


def workload(x1, x2, w, b):
    return jnp.einsum('...i,oij,...j->...o', x1, w, x2) + b
