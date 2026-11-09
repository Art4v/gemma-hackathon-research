"""[bf16 variant] SigLIP sigmoid contrastive loss: L2-normalize -> scaled img @ txt^T + bias -> -sum logsigmoid(labels * logits) / B."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'siglip_sigmoid_contrastive_loss',
    'batch': 8192,
    'dim': 1152,
    'init_logit_scale': 10.0,
    'init_logit_bias': -10.0,
    'norm_eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    k_img, k_txt = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['dim'])
    image_features = jax.random.normal(k_img, shape, dtype)
    text_features = jax.random.normal(k_txt, shape, dtype)
    logit_scale = jnp.asarray(math.log(CONFIG['init_logit_scale']), dtype)
    logit_bias = jnp.asarray(CONFIG['init_logit_bias'], dtype)
    return image_features, text_features, logit_scale, logit_bias


def _l2_normalize(x):
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True))
    return x / jnp.maximum(norm, CONFIG['norm_eps'])


def workload(image_features, text_features, logit_scale, logit_bias):
    img = _l2_normalize(image_features)
    txt = _l2_normalize(text_features)
    sim = jnp.matmul(img, txt.T, precision=lax.Precision.HIGHEST)
    logits = (jnp.exp(logit_scale) * sim + logit_bias).astype(jnp.float32)
    n = img.shape[0]
    labels = 2.0 * jnp.eye(n, dtype=jnp.float32) - 1.0
    return -jnp.sum(jax.nn.log_sigmoid(labels * logits)) / n
