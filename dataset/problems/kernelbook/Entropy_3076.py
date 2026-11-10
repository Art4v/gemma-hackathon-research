"""[bf16 variant] Entropy: softmax*log_softmax along dim 1 and dim 2, summed, negated, normalized by ms1*ms2."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Entropy_3076',
    'num': 32,
    'ms1': 1024,
    'ms2': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['num'], CONFIG['ms1'], CONFIG['ms2']), dtype)
    return (x,)


def workload(x):
    _, ms1, ms2 = x.shape
    ent_p2g = jax.nn.softmax(x, axis=1) * jax.nn.log_softmax(x, axis=1)
    ent_g2p = jax.nn.softmax(x, axis=2) * jax.nn.log_softmax(x, axis=2)
    ent_sum = -1.0 * jnp.sum(ent_p2g) - jnp.sum(ent_g2p)
    return ent_sum / (ms1 * ms2)
