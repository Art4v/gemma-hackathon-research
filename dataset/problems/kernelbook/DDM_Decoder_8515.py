"""[bf16 variant] DDM_Decoder: Linear(dim->288) -> ELU -> Linear(288->dim) -> ELU -> Linear(dim->obs_space)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DDM_Decoder_8515',
    'batch': 16,
    'height': 64,
    'width': 64,
    'dim': 512,
    'hidden': 32 * 3 * 3,
    'obs_space': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(ks[0], (c['batch'], c['height'], c['width'], c['dim']), dtype)
    fc_w = jax.random.normal(ks[1], (c['dim'], c['hidden']), dtype) * 0.02
    fc_b = jax.random.normal(ks[2], (c['hidden'],), dtype) * 0.02
    l1_w = jax.random.normal(ks[3], (c['hidden'], c['dim']), dtype) * 0.02
    l1_b = jax.random.normal(ks[4], (c['dim'],), dtype) * 0.02
    l2_w = jax.random.normal(ks[5], (c['dim'], c['obs_space']), dtype) * 0.02
    l2_b = jax.random.normal(ks[6], (c['obs_space'],), dtype) * 0.02
    return x, fc_w, fc_b, l1_w, l1_b, l2_w, l2_b


def workload(x, fc_w, fc_b, l1_w, l1_b, l2_w, l2_b):
    h = jax.nn.elu(x @ fc_w + fc_b)
    h = jax.nn.elu(h @ l1_w + l1_b)
    return h @ l2_w + l2_b
