"""[bf16 variant] Bilinear form (nn.Bilinear, out_features=1, no bias) followed by exp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LBM_9785',
    'lead_shape': (8, 128, 128),
    'l_dim': 128,
    'r_dim': 128,
    'out_features': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    lead = CONFIG['lead_shape']
    e1 = jax.random.uniform(k1, (*lead, CONFIG['l_dim']), dtype)
    e2 = jax.random.uniform(k2, (*lead, CONFIG['r_dim']), dtype)
    w = jax.random.normal(
        k3, (CONFIG['out_features'], CONFIG['l_dim'], CONFIG['r_dim']), dtype) * 0.02
    return e1, e2, w


def workload(e1, e2, w):
    # w keeps PyTorch's Bilinear layout (out, l_dim, r_dim).
    y = jnp.einsum('...i,oij,...j->...o', e1, w, e2)
    return jnp.exp(y)
