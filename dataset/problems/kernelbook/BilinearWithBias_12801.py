"""[bf16 variant] Bilinear form x1^T W_k x2 plus bias, plus linear terms x1 @ V1 and x2 @ V2."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BilinearWithBias_12801',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'in1_features': 128,
    'in2_features': 128,
    'out_features': 128,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x1, k_x2, k_w, k_v1, k_v2, k_bias = jax.random.split(key, 6)
    prefix = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'])
    in1 = CONFIG['in1_features']
    in2 = CONFIG['in2_features']
    out = CONFIG['out_features']
    scale = CONFIG['weight_scale']
    x1 = jax.random.uniform(k_x1, prefix + (in1,), dtype)
    x2 = jax.random.uniform(k_x2, prefix + (in2,), dtype)
    W = jax.random.normal(k_w, (out, in1, in2), dtype) * scale
    V1 = jax.random.normal(k_v1, (in1, out), dtype) * scale
    V2 = jax.random.normal(k_v2, (in2, out), dtype) * scale
    bias = jax.random.normal(k_bias, (out,), dtype) * scale
    return x1, x2, W, V1, V2, bias


def workload(x1, x2, W, V1, V2, bias):
    def bilinear_channel(W_k):
        return jnp.sum((x1 @ W_k) * x2, axis=-1)

    bilinear = jnp.moveaxis(lax.map(bilinear_channel, W), 0, -1)
    return bilinear + bias + x1 @ V1 + x2 @ V2
