"""[bf16 variant] InnerProductDecoder: sigmoid(sum(z[src] * z[dst], axis=1)) over an edge list."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'InnerProductDecoder_2030',
    'num_nodes': 262144,
    'latent_dim': 128,
    'num_edges': 2097152,
}


def create_inputs(dtype=jnp.bfloat16):
    k_z, k_e = jax.random.split(jax.random.key(0))
    z = jax.random.normal(k_z, (CONFIG['num_nodes'], CONFIG['latent_dim']), dtype)
    edge_index = jax.random.randint(
        k_e, (2, CONFIG['num_edges']), 0, CONFIG['num_nodes'], dtype=jnp.int32)
    return z, edge_index


def workload(z, edge_index):
    value = jnp.sum(z[edge_index[0]] * z[edge_index[1]], axis=1)
    return jax.nn.sigmoid(value)
