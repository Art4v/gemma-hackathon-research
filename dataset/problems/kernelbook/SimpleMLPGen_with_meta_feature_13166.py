"""[bf16 variant] Linear -> ELU -> Linear (SimpleMLPGen forward)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SimpleMLPGen_with_meta_feature_13166',
    'lead_shape': (8, 8, 512),
    'num_in_features': 1024,
    'neurons_per_layer': 1024,
    'num_out_features': 1024,
    'elu_alpha': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_wi, k_bi, k_wo, k_bo = jax.random.split(jax.random.key(0), 5)
    d_in = CONFIG['num_in_features']
    h = CONFIG['neurons_per_layer']
    d_out = CONFIG['num_out_features']
    x = jax.random.uniform(k_x, (*CONFIG['lead_shape'], d_in), dtype)
    w_in = jax.random.normal(k_wi, (d_in, h), dtype) * 0.02
    b_in = jax.random.normal(k_bi, (h,), dtype) * 0.02
    w_out = jax.random.normal(k_wo, (h, d_out), dtype) * 0.02
    b_out = jax.random.normal(k_bo, (d_out,), dtype) * 0.02
    return x, w_in, b_in, w_out, b_out


def workload(x, w_in, b_in, w_out, b_out):
    h = jax.nn.elu(x @ w_in + b_in, alpha=CONFIG['elu_alpha'])
    return h @ w_out + b_out
