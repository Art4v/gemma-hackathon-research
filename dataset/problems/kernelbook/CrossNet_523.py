"""CrossNet (DCN vector parameterization): repeated tensordot -> broadcast matmul -> bias add -> residual, on a 4D input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CrossNet_523',
    'batch': 64,
    'in_features': 16,  # contracted dim 1; dim 2 must also equal this (torch broadcast constraint)
    'inner': 1024,
    'layer_num': 2,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    f = CONFIG['in_features']
    x = jax.random.uniform(k_x, (CONFIG['batch'], f, f, CONFIG['inner']), dtype)
    kernels = jax.random.normal(k_w, (CONFIG['layer_num'], f, 1), dtype) * 0.02
    bias = jax.random.normal(k_b, (CONFIG['layer_num'], f, 1), dtype) * 0.02
    return x, kernels, bias


def workload(inputs, kernels, bias):
    x_0 = jnp.expand_dims(inputs, 2)
    x_l = x_0
    for i in range(kernels.shape[0]):
        xl_w = jnp.tensordot(x_l, kernels[i], axes=([1], [0]))
        dot_ = jnp.matmul(x_0, xl_w)
        x_l = dot_ + bias[i] + x_l
    # torch.squeeze(dim=2) is a no-op when that dim is not size 1
    if x_l.shape[2] == 1:
        x_l = jnp.squeeze(x_l, axis=2)
    return x_l
