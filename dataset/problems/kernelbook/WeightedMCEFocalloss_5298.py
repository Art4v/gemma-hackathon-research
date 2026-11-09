"""[bf16 variant] Weighted multi-class focal loss: centercrop -> log_softmax/softmax(dim=1) -> (1-p)^gamma * weight * logp * y_true -> sum(C) -> -mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WeightedMCEFocalloss_5298',
    'batch': 8,
    'channels': 32,
    'height': 256,
    'width': 256,
    'gamma': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    y_pred = jax.random.uniform(k1, shape, dtype)
    y_true = jax.random.uniform(k2, shape, dtype)
    weight = jax.random.uniform(k3, shape, dtype)
    return y_pred, y_true, weight


def _centercrop(image, w, h):
    ht, wt = image.shape[2], image.shape[3]
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


def workload(y_pred, y_true, weight):
    h, w = y_pred.shape[2], y_pred.shape[3]
    y_true = _centercrop(y_true, w, h)
    weight = _centercrop(weight, w, h)
    y_pred_log = jax.nn.log_softmax(y_pred, axis=1)
    fweight = (1.0 - jax.nn.softmax(y_pred, axis=1)) ** CONFIG['gamma']
    logpy = jnp.sum(weight * fweight * y_pred_log * y_true, axis=1)
    return -jnp.mean(logpy)
