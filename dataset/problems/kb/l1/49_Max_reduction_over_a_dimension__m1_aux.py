"""[m1 M2_second_output variant of kb/l1/49_Max_reduction_over_a_dimension] Max reduction over dim 1, also returning the logsumexp over the same axis computed from that max (the online-softmax normalizer)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '49_Max_reduction_over_a_dimension_m1_aux',
    'batch_size': 16,
    'dim1': 2048,
    'dim2': 2048,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch_size'], CONFIG['dim1'], CONFIG['dim2']), dtype=dtype
    )
    return (x,)


def workload(x):
    dim = CONFIG['dim']
    m = jnp.max(x, axis=dim, keepdims=True)
    lse = m + jnp.log(jnp.sum(jnp.exp(x - m), axis=dim, keepdims=True))
    return jnp.squeeze(m, axis=dim), jnp.squeeze(lse, axis=dim)
