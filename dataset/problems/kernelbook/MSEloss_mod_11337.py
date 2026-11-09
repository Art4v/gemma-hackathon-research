"""[bf16 variant] MSEloss_mod: (y_gt[:,:,0:2] transposed - y_pred[:,:,0:2])^2 summed over 2 channels, then mean."""
import jax
import jax.numpy as jnp

CONFIG = {'name': 'MSEloss_mod_11337', 'T': 2048, 'N': 4096, 'C': 4}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    t, n, c = CONFIG['T'], CONFIG['N'], CONFIG['C']
    y_pred = jax.random.uniform(k1, (t, n, c), dtype)
    y_gt = jax.random.uniform(k2, (n, t, c), dtype)
    return y_pred, y_gt


def workload(y_pred, y_gt):
    mu_x = y_pred[:, :, 0]
    mu_y = y_pred[:, :, 1]
    x = y_gt[:, :, 0].T
    y = y_gt[:, :, 1].T
    out = (x - mu_x) ** 2 + (y - mu_y) ** 2
    return jnp.sum(out) / (out.shape[0] * out.shape[1])
