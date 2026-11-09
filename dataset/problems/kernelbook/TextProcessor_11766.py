"""[bf16 variant] TextProcessor: Linear(desc_dim -> hid_dim) over flattened rows, reshape, ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TextProcessor_11766',
    'bs': 64,
    'num_classes': 512,
    'desc_dim': 1024,
    'hid_dim': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    desc = jax.random.uniform(
        k_x, (CONFIG['bs'], CONFIG['num_classes'], CONFIG['desc_dim']), dtype)
    w = jax.random.normal(k_w, (CONFIG['desc_dim'], CONFIG['hid_dim']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['hid_dim'],), dtype) * 0.02
    return desc, w, b


def workload(desc, w, b):
    bs, num_classes, desc_dim = desc.shape
    out = desc.reshape(-1, desc_dim) @ w + b
    out = out.reshape(bs, num_classes, -1)
    return jax.nn.relu(out)
