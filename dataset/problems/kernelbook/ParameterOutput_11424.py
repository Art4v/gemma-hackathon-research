"""Linear -> softmax over dim 1 (PyTorch implicit dim for 4D input) -> affine rescale to [low, high]."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ParameterOutput_11424',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'in_features': 512,
    'out_features': 512,
    'low': -1.0,
    'high': 1.0,
    # F.softmax without dim on a 4D tensor uses dim=1 (PyTorch _get_softmax_dim rule).
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['in_features']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    y = x @ w + b
    y = jax.nn.softmax(y, axis=CONFIG['softmax_axis'])
    low, high = CONFIG['low'], CONFIG['high']
    return (high - low) * y + low
