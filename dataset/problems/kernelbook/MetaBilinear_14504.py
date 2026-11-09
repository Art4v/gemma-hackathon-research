"""MetaBilinear: F.bilinear(x1, x2, W, b) = einsum('...i,oij,...j->...o') + b."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MetaBilinear_14504',
    'batch': 8,
    'seq_len': 2048,
    'in1_features': 256,
    'in2_features': 256,
    'out_features': 256,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    b, s = CONFIG['batch'], CONFIG['seq_len']
    i1, i2, o = CONFIG['in1_features'], CONFIG['in2_features'], CONFIG['out_features']
    x1 = jax.random.uniform(k1, (b, s, i1), dtype)
    x2 = jax.random.uniform(k2, (b, s, i2), dtype)
    weight = jax.random.normal(k3, (o, i1, i2), dtype) * 0.02
    bias = jax.random.normal(k4, (o,), dtype) * 0.02
    return x1, x2, weight, bias


def workload(x1, x2, weight, bias):
    return jnp.einsum('...i,oij,...j->...o', x1, weight, x2) + bias
