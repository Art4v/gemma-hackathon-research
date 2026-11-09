"""[bf16 variant] Bilinear: y = x1^T W x2 + b over the last feature dim (F.bilinear)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MetaBilinear_9554',
    'batch': 8,
    'seq': 2048,
    'in1_features': 256,
    'in2_features': 256,
    'out_features': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    b, s = CONFIG['batch'], CONFIG['seq']
    i1, i2, o = CONFIG['in1_features'], CONFIG['in2_features'], CONFIG['out_features']
    x1 = jax.random.uniform(k1, (b, s, i1), dtype)
    x2 = jax.random.uniform(k2, (b, s, i2), dtype)
    w = jax.random.normal(k3, (o, i1, i2), dtype) * 0.02
    bias = jax.random.normal(k4, (o,), dtype) * 0.02
    return x1, x2, w, bias


def workload(x1, x2, w, bias):
    # Sizes come from array shapes so any leading dims work, as in PyTorch.
    t = jnp.einsum('...i,oij->...oj', x1, w, precision=lax.Precision.HIGHEST)
    y = jnp.einsum('...oj,...j->...o', t, x2, precision=lax.Precision.HIGHEST)
    return y + bias
