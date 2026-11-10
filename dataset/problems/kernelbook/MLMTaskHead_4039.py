"""MLMTaskHead_4039: Linear -> GELU (exact) -> LayerNorm (eps=1e-12) -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLMTaskHead_4039',
    'batch': 8,
    'height': 128,
    'width': 128,
    'ninp': 512,
    'ntoken': 512,
    'eps': 1e-12,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_src, k_w_span, k_b_span, k_w_head, k_b_head = jax.random.split(key, 5)
    ninp = CONFIG['ninp']
    ntoken = CONFIG['ntoken']
    scale = CONFIG['weight_scale']
    src = jax.random.uniform(
        k_src, (CONFIG['batch'], CONFIG['height'], CONFIG['width'], ninp), dtype
    )
    w_span = jax.random.normal(k_w_span, (ninp, ninp), dtype) * scale
    b_span = jax.random.normal(k_b_span, (ninp,), dtype) * scale
    norm_scale = jnp.ones((ninp,), dtype)
    norm_shift = jnp.zeros((ninp,), dtype)
    w_head = jax.random.normal(k_w_head, (ninp, ntoken), dtype) * scale
    b_head = jax.random.normal(k_b_head, (ntoken,), dtype) * scale
    return src, w_span, b_span, norm_scale, norm_shift, w_head, b_head


def workload(src, w_span, b_span, norm_scale, norm_shift, w_head, b_head):
    x = src @ w_span + b_span
    x = jax.nn.gelu(x, approximate=False)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + CONFIG['eps'])
    x = x * norm_scale + norm_shift
    return x @ w_head + b_head
