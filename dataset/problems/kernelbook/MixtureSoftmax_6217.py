"""[bf16 variant] Mixture softmax: weighted (per-dim-2) sum over dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MixtureSoftmax_6217',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'width': 512,
    'in_features': 128,  # must equal height: torch broadcast of w[None,:,None] hits dim 2
    'weight_mean': 10.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['in_features'],), dtype) + CONFIG['weight_mean']
    w = w / jnp.sum(w)
    return x, w


def workload(x, w):
    # torch: input * weight[None, :, None] on 4D input -> weight broadcast as (1, 1, F, 1)
    return jnp.sum(x * w[None, None, :, None], axis=1)
