"""Join: gather unary rows by two index vectors and concatenate with binary along features."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Join_17327',
    'num_unary': 65536,
    'unary_dim': 512,
    'num_pairs': 65536,
    'binary_dim': 512,
}


def create_inputs(dtype=jnp.float32):
    k_u, k_b, k_i1, k_i2 = jax.random.split(jax.random.key(0), 4)
    unary = jax.random.normal(k_u, (CONFIG['num_unary'], CONFIG['unary_dim']), dtype)
    binary = jax.random.uniform(k_b, (CONFIG['num_pairs'], CONFIG['binary_dim']), dtype)
    index1 = jax.random.randint(k_i1, (CONFIG['num_pairs'],), 0, CONFIG['num_unary'], jnp.int32)
    index2 = jax.random.randint(k_i2, (CONFIG['num_pairs'],), 0, CONFIG['num_unary'], jnp.int32)
    return unary, binary, index1, index2


def workload(unary, binary, index1, index2):
    # reshape(-1) covers torch's squeeze + unsqueeze-if-scalar handling.
    index1 = jnp.reshape(index1, (-1,))
    index2 = jnp.reshape(index2, (-1,))
    u1 = unary[index1]
    u2 = unary[index2]
    return jnp.concatenate([u1, u2, binary.astype(unary.dtype)], axis=1)
