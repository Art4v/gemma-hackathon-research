"""[bf16 variant] Soft MoE: RMSNorm -> slot logits -> masked dispatch/combine softmax -> per-expert GELU FFN -> combine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'soft_moe',
    'batch': 8,
    'seq_len': 1024,
    'dim': 768,
    'num_experts': 16,
    'num_slots': 16,
    'mult': 4,
    'min_len': 512,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    b, n, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim']
    e, s = CONFIG['num_experts'], CONFIG['num_slots']
    h = d * CONFIG['mult']
    k = jax.random.split(jax.random.key(0), 7)
    x = jax.random.normal(k[0], (b, n, d), dtype)
    lengths = jax.random.randint(k[1], (b, 1), CONFIG['min_len'], n + 1)
    mask = jnp.arange(n)[None, :] < lengths
    gamma = jnp.ones((d,), dtype)
    slot_gamma = jnp.ones((d,), dtype)
    slot_embeds = jax.random.normal(k[2], (e, s, d), dtype)
    w1 = jax.random.normal(k[3], (e, d, h), dtype) * 0.02
    b1 = jax.random.normal(k[4], (e, h), dtype) * 0.02
    w2 = jax.random.normal(k[5], (e, h, d), dtype) * 0.02
    b2 = jax.random.normal(k[6], (e, d), dtype) * 0.02
    return x, mask, gamma, slot_gamma, slot_embeds, w1, b1, w2, b2


def _rms_norm(x, gamma):
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True))
    return x / jnp.maximum(norm, CONFIG['eps']) * (x.shape[-1] ** 0.5) * gamma


def workload(x, mask, gamma, slot_gamma, slot_embeds, w1, b1, w2, b2):
    x = _rms_norm(x, gamma)
    slots_e = _rms_norm(slot_embeds, slot_gamma)
    logits = jnp.einsum('bnd,esd->bnes', x, slots_e)
    logits = jnp.where(mask[:, :, None, None], logits, -jnp.finfo(logits.dtype).max)
    dispatch = jax.nn.softmax(logits, axis=1)
    b, n, e, s = logits.shape
    combine = jax.nn.softmax(logits.reshape(b, n, e * s), axis=-1)
    slots = jnp.einsum('bnd,bnes->besd', x, dispatch)
    hid = jax.nn.gelu(jnp.einsum('besd,edh->besh', slots, w1) + b1[None, :, None], approximate=False)
    out = jnp.einsum('besh,ehd->besd', hid, w2) + b2[None, :, None]
    out = out.reshape(b, e * s, -1)
    return jnp.einsum('bsd,bns->bnd', out, combine)
