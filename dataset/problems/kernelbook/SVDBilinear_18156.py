"""[bf16 variant] SVDBilinear: usv = (left * diag) @ right, then bilinear(in1, in2, usv) without bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SVDBilinear_18156',
    'batch_shape': (8, 4, 64),
    'num_basis': 128,
    'in1_features': 512,
    'in2_features': 512,
    'out_features': 128,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    b = CONFIG['batch_shape']
    i1, i2 = CONFIG['in1_features'], CONFIG['in2_features']
    o, k = CONFIG['out_features'], CONFIG['num_basis']
    s = CONFIG['weight_scale']
    in1 = jax.random.uniform(k1, (*b, i1), dtype)
    in2 = jax.random.uniform(k2, (*b, i2), dtype)
    left = jax.random.normal(k3, (o, i1, k), dtype) * s
    right = jax.random.normal(k4, (o, k, i2), dtype) * s
    # diag keeps its torch init scale (std 1/sqrt(K)) so outputs are not vanishingly small.
    diag = jax.random.normal(k5, (o, 1, k), dtype) * (1.0 / k ** 0.5)
    return in1, in2, left, right, diag


def workload(in1, in2, left, right, diag):
    hp = lax.Precision.HIGHEST
    usv = jnp.matmul(left * diag, right, precision=hp)  # (O, I1, I2)
    t = jnp.einsum('...i,oij->...oj', in1, usv, precision=hp)
    return jnp.einsum('...oj,...j->...o', t, in2, precision=hp)
