"""PatchApplier: sequentially overwrite image pixels with each nonzero adversarial patch (torch.where over unbind(adv, 1))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PatchApplier_5711',
    'batch': 8,
    'num_patches': 8,
    'channels': 3,
    'height': 512,
    'width': 512,
    'patch_zero_fraction': 0.7,
}


def create_inputs(dtype=jnp.float32):
    k_img, k_adv, k_mask = jax.random.split(jax.random.key(0), 3)
    b, n, c = CONFIG['batch'], CONFIG['num_patches'], CONFIG['channels']
    h, w = CONFIG['height'], CONFIG['width']
    img_batch = jax.random.uniform(k_img, (b, c, h, w), dtype)
    adv_vals = jax.random.uniform(k_adv, (b, n, c, h, w), dtype)
    keep = jax.random.uniform(k_mask, (b, n, c, h, w)) >= CONFIG['patch_zero_fraction']
    adv_batch = jnp.where(keep, adv_vals, jnp.zeros_like(adv_vals))
    return img_batch, adv_batch


def workload(img_batch, adv_batch):
    out = img_batch
    for i in range(adv_batch.shape[1]):
        adv = adv_batch[:, i]
        out = jnp.where(adv == 0, out, adv)
    return out
