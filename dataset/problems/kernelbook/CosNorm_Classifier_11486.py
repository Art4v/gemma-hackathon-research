"""CosNorm classifier: row L2 norm squash of x, column-normalized weight, scaled matmul."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CosNorm_Classifier_11486',
    'batch': 16384,
    'in_dims': 2048,
    'out_dims': 4096,
    'scale': 16.0,
    'margin': 0.5,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['in_dims']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_dims'], CONFIG['out_dims']), dtype) * 0.02
    return x, w


def workload(x, w):
    norm_x = jnp.linalg.norm(x, axis=1, keepdims=True)
    ex = norm_x / (1 + norm_x) * (x / norm_x)
    # w is (in, out): PyTorch's per-output-row norm is a norm over axis 0 here.
    ew = w / jnp.linalg.norm(w, axis=0, keepdims=True)
    return (CONFIG['scale'] * ex) @ ew
