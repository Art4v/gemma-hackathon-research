"""HighwayLayer: sigmoid(Linear) gate * tanh(Linear) + (1 - gate) * x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HighwayLayer_8017',
    'batch': 8,
    'height': 64,
    'width': 64,
    'dim': 1024,
    'gate_bias_init': -1.0,
}


def create_inputs(dtype=jnp.float32):
    kx, kg, kn = jax.random.split(jax.random.key(0), 3)
    dim = CONFIG['dim']
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['height'], CONFIG['width'], dim), dtype)
    w_gate = jax.random.normal(kg, (dim, dim), dtype) * 0.02
    b_gate = jnp.full((dim,), CONFIG['gate_bias_init'], dtype)
    w_nlin = jax.random.normal(kn, (dim, dim), dtype) * 0.02
    b_nlin = jnp.zeros((dim,), dtype)
    return x, w_gate, b_gate, w_nlin, b_nlin


def workload(x, w_gate, b_gate, w_nlin, b_nlin):
    gate = jax.nn.sigmoid(x @ w_gate + b_gate)
    nlin = jnp.tanh(x @ w_nlin + b_nlin)
    return gate * nlin + (1.0 - gate) * x
