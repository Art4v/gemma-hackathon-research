"""[bf16 variant] SELU activation: scale * (max(0,x) + min(0, alpha*(exp(x)-1)))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '27_SELU_',
    'batch': 4096,
    'dim': 16384,
    'alpha': 1.6732632423543772848170429916717,
    'scale': 1.0507009873554804934193349852946,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    alpha = jnp.asarray(CONFIG['alpha'], dtype=x.dtype)
    scale = jnp.asarray(CONFIG['scale'], dtype=x.dtype)
    neg = alpha * jnp.expm1(jnp.minimum(x, 0))
    return scale * jnp.where(x > 0, x, neg)
