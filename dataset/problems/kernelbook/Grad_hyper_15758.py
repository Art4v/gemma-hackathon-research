"""[bf16 variant] Grad_hyper: N-D gradient loss (abs finite diffs along H and W, per-sample mean, scaled by wts, global mean)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Grad_hyper_15758',
    'batch': 128,
    'channels': 4,
    'height': 256,
    # The torch op broadcasts d (shape (B,)) against wts' last axis, so width must equal batch.
    'width': 128,
    'penalty': 'l1',
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    y_pred = jax.random.uniform(k1, shape, dtype)
    wts = jax.random.uniform(k2, shape, dtype)
    return y_pred, wts


def workload(y_pred, wts):
    dy = jnp.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = jnp.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    if CONFIG['penalty'] == 'l2':
        dy = dy * dy
        dx = dx * dx
    d = jnp.mean(dx, axis=(1, 2, 3)) + jnp.mean(dy, axis=(1, 2, 3))
    # Same trailing-axis broadcast as PyTorch: d (B,) aligns with wts' last dim W.
    grad = d / 2.0 * wts
    return jnp.mean(grad)
