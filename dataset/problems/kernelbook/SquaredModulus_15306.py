"""[bf16 variant] SquaredModulus: x**2 -> AvgPool1d(k=2, s=2) over dim 1 -> *2."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SquaredModulus_15306',
    'batch': 16,
    'length': 4096,
    'channels': 1024,
    'pool_kernel': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['length'], CONFIG['channels']), dtype)
    return (x,)


def workload(x):
    k = CONFIG['pool_kernel']
    n, length, c = x.shape
    out_len = length // k
    # The torch transpose puts dim 1 last, so AvgPool1d pools adjacent rows along dim 1.
    pairs = x[:, :out_len * k, :].reshape(n, out_len, k, c)
    return 2 * jnp.mean(pairs ** 2, axis=2)
