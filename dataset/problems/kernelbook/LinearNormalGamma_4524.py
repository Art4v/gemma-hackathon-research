"""[bf16 variant] Linear -> reshape (batch, -1, 4) -> unbind into mu/logv/logalpha/logbeta -> evidence log(exp(x)+1) on the last three (alpha offset +1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearNormalGamma_4524',
    'batch': 8,
    'height': 128,
    'width': 128,
    'in_channels': 256,
    'out_channels': 64,
    'num_params': 4,
    'alpha_offset': 1.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    out_features = CONFIG['out_channels'] * CONFIG['num_params']
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['in_channels']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['in_channels'], out_features), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (out_features,), dtype) * CONFIG['weight_scale']
    return x, w, b


def _evidence(x):
    return jnp.log(jnp.exp(x) + 1)


def workload(x, w, b):
    num_params = CONFIG['num_params']
    pred = (x @ w + b).reshape(x.shape[0], -1, num_params)
    mu, logv, logalpha, logbeta = [pred[..., i] for i in range(num_params)]
    return (
        mu,
        _evidence(logv),
        _evidence(logalpha) + CONFIG['alpha_offset'],
        _evidence(logbeta),
    )
