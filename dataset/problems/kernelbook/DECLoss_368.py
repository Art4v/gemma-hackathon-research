"""[bf16 variant] DECLoss: target distribution p = row-normalized(q**2 / colsum(q)), then KL = sum(p * log(p / q))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DECLoss_368',
    'num_samples': 16384,
    'num_clusters': 2048,
    'min_val': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    # ponytail: minval avoids exact zeros, where log(p / q) would be 0/0 = NaN
    q = jax.random.uniform(
        key, (CONFIG['num_samples'], CONFIG['num_clusters']), dtype,
        minval=CONFIG['min_val'], maxval=1.0)
    return (q,)


def workload(q):
    weight = q ** 2 / jnp.sum(q, axis=0)
    p = jax.lax.stop_gradient(weight / jnp.sum(weight, axis=1, keepdims=True))
    return jnp.sum(p * jnp.log(p / q))
