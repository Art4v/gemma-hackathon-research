"""ProbabilityBilinear: softmax(W, axis=0) -> bilinear(x1, x2, W) -> L1-normalize over last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ProbabilityBilinear_14907',
    'batch': 16,
    'seq': 512,
    'in1_features': 256,
    'in2_features': 256,
    'out_features': 256,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b, s = CONFIG['batch'], CONFIG['seq']
    i1, i2, o = CONFIG['in1_features'], CONFIG['in2_features'], CONFIG['out_features']
    x1 = jax.random.uniform(k1, (b, s, i1), dtype)
    x2 = jax.random.uniform(k2, (b, s, i2), dtype)
    w = jax.random.normal(k3, (o, i1, i2), dtype) * 0.02
    return x1, x2, w


def workload(x1, x2, w):
    w = jax.nn.softmax(w, axis=0)
    out = jnp.einsum('...i,oij,...j->...o', x1, w, x2)
    return out / jnp.sum(out, axis=-1, keepdims=True)
