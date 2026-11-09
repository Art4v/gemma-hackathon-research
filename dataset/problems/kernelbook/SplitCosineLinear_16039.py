"""[bf16 variant] SplitCosineLinear: L2-normalize(x, dim=1) @ L2-normalize(W1/W2, in-dim), concat on dim 1, scale by sigma."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SplitCosineLinear_16039',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'in_features': 512,
    'out_features1': 512,
    'out_features2': 512,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['in_features']),
        dtype,
    )
    w1 = jax.random.normal(k_w1, (CONFIG['in_features'], CONFIG['out_features1']), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (CONFIG['in_features'], CONFIG['out_features2']), dtype) * 0.02
    sigma = jnp.ones((1,), dtype)
    return x, w1, w2, sigma


def _l2_normalize(a, axis):
    norm = jnp.sqrt(jnp.sum(a * a, axis=axis, keepdims=True))
    return a / jnp.maximum(norm, CONFIG['eps'])


def workload(x, w1, w2, sigma):
    # F.normalize(input, dim=1) normalizes over axis 1 (channels), not the feature axis.
    xn = _l2_normalize(x, 1)
    # Weights are (in, out); torch normalizes (out, in) over in -> axis 0 here.
    out1 = xn @ _l2_normalize(w1, 0)
    out2 = xn @ _l2_normalize(w2, 0)
    out = jnp.concatenate((out1, out2), axis=1)
    return sigma * out
