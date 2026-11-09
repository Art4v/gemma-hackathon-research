"""SplitDim: softplus on one column plus softplus(var), other columns passed through."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SplitDim_10398',
    'batch': 65536,
    'features': 512,
    'nonlin_col': 1,
    'softplus_threshold': 20.0,
}


def _softplus(x):
    # Matches torch.nn.functional.softplus (beta=1, threshold=20).
    t = CONFIG['softplus_threshold']
    return jnp.where(x > t, x, jnp.log1p(jnp.exp(jnp.minimum(x, t))))


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['features']), dtype)
    var = jnp.zeros((1,), dtype)
    return x, var


def workload(x, var):
    col = CONFIG['nonlin_col']
    transformed = _softplus(x[:, col]) + _softplus(var)[0]
    return x.at[:, col].set(transformed)
