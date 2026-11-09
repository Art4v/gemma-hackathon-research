"""[bf16 variant] DuelDQNet: Linear -> ReLU -> (value head, advantage head) -> val + adv - mean(adv, axis=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DuelDQNet_1507',
    'batch': 16,
    'dim1': 16,
    'dim2': 128,
    'obs_size': 1024,
    'hid_size': 256,
    'act_size': 128,
    # torch reduces dim=1 of the 4-D activation (not the action axis); kept as-is.
    'mean_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['obs_size']), dtype)
    base_w = jax.random.normal(keys[1], (c['obs_size'], c['hid_size']), dtype) * 0.02
    base_b = jax.random.normal(keys[2], (c['hid_size'],), dtype) * 0.02
    val_w = jax.random.normal(keys[3], (c['hid_size'], 1), dtype) * 0.02
    val_b = jax.random.normal(keys[4], (1,), dtype) * 0.02
    adv_w = jax.random.normal(keys[5], (c['hid_size'], c['act_size']), dtype) * 0.02
    adv_b = jax.random.normal(keys[6], (c['act_size'],), dtype) * 0.02
    return x, base_w, base_b, val_w, val_b, adv_w, adv_b


def workload(x, base_w, base_b, val_w, val_b, adv_w, adv_b):
    h = jax.nn.relu(x @ base_w + base_b)
    val = h @ val_w + val_b
    adv = h @ adv_w + adv_b
    return val + (adv - jnp.mean(adv, axis=CONFIG['mean_axis'], keepdims=True))
