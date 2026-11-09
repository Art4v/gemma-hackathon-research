"""KL divergence (batchmean) between log_softmax(input) and softmax(target) over the last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KL_5741',
    'batch': 16,
    'channels': 32,
    'height': 128,
    'width': 512,
    'softmax_axis': -1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    t = jax.random.uniform(k2, shape, dtype)
    return x, t


def workload(x, t):
    axis = CONFIG['softmax_axis']
    x = x.astype(jnp.float32)
    t = t.astype(jnp.float32)
    logq = jax.nn.log_softmax(x, axis=axis)
    p = jax.nn.softmax(t, axis=axis)
    # xlogy convention: terms with p == 0 contribute 0, as in torch.kl_div.
    safe_p = jnp.where(p > 0, p, 1.0)
    pointwise = jnp.where(p > 0, p * (jnp.log(safe_p) - logq), 0.0)
    return jnp.sum(pointwise) / x.shape[0]
