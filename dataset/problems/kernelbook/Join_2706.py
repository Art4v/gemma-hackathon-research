"""[bf16 variant] Join: gather unary rows by index1 and index2, then concat with binary along features."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Join_2706',
    'num_unary': 65536,
    'unary_dim': 256,
    'num_binary': 131072,
    'binary_dim': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    k_u, k_b, k_i1, k_i2 = jax.random.split(jax.random.key(0), 4)
    unary = jax.random.normal(k_u, (CONFIG['num_unary'], CONFIG['unary_dim']), dtype)
    binary = jax.random.normal(k_b, (CONFIG['num_binary'], CONFIG['binary_dim']), dtype)
    index1 = jax.random.randint(k_i1, (CONFIG['num_binary'],), 0, CONFIG['num_unary'], jnp.int32)
    index2 = jax.random.randint(k_i2, (CONFIG['num_binary'],), 0, CONFIG['num_unary'], jnp.int32)
    return unary, binary, index1, index2


def workload(unary, binary, index1, index2):
    # torch squeeze plus the scalar unsqueeze both reduce to a 1-D index vector.
    index1 = index1.reshape(-1)
    index2 = index2.reshape(-1)
    u1 = jnp.take(unary, index1, axis=0)
    u2 = jnp.take(unary, index2, axis=0)
    return jnp.concatenate([u1, u2, binary], axis=1)
