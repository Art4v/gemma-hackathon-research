"""[bf16 variant] SplitDim: softplus on one column plus softplus(var), other columns passed through."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SplitDim_6311',
    'batch': 65536,
    'dim': 1024,
    'col': 1,
    'softplus_threshold': 20.0,
}


def _softplus(x):
    # Matches torch.nn.functional.softplus defaults (beta=1, threshold=20).
    return jnp.where(x > CONFIG['softplus_threshold'], x, jnp.log1p(jnp.exp(x)))


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['dim']), dtype)
    var = jnp.zeros((1,), dtype)
    return x, var


def workload(x, var):
    col = CONFIG['col']
    transformed = _softplus(x[:, col]) + _softplus(var)
    return x.at[:, col].set(transformed)
