"""[bf16 variant] CosNorm classifier: squash-normalize rows of x, L2-normalize weight columns, scaled matmul."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CosNorm_Classifier_5719',
    'batch': 8192,
    'in_dims': 4096,
    'out_dims': 4096,
    'scale': 16.0,
    'margin': 0.5,  # unused in forward, kept for parity with the torch module
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['in_dims']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_dims'], CONFIG['out_dims']), dtype) * 0.02
    return x, w


def workload(x, w):
    # w is (in, out); torch normalizes each output row of its (out, in) weight -> axis 0 here.
    norm_x = jnp.linalg.norm(x, axis=1, keepdims=True)
    ex = norm_x / (1 + norm_x) * (x / norm_x)
    ew = w / jnp.linalg.norm(w, axis=0, keepdims=True)
    return (CONFIG['scale'] * ex) @ ew
