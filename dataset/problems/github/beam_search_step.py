"""[bf16 variant] Beam search step: log_softmax over vocab + cumulative beam score, top-(2*beam) over flattened beam x vocab, split into (beam, token)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'beam_search_step',
    'batch': 64,
    'beam': 8,
    'vocab': 50304,
    'max_len': 16,
    'step': 5,
    'candidate_multiple': 2,
    'logit_scale': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    b, beam, v, t = CONFIG['batch'], CONFIG['beam'], CONFIG['vocab'], CONFIG['max_len']
    logits = jax.random.normal(k1, (b, beam, v), dtype) * CONFIG['logit_scale']
    scores = -jnp.cumsum(jax.random.uniform(k2, (b, beam, t), dtype), axis=-1)
    return logits, scores


def workload(logits, scores):
    bsz, beam_size, vocab_size = logits.shape
    step = CONFIG['step']
    lprobs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    if step == 0:
        lprobs = lprobs[:, ::beam_size, :]
    else:
        lprobs = lprobs + scores[:, :, step - 1][:, :, None].astype(jnp.float32)
    flat = lprobs.reshape(bsz, -1)
    k = min(CONFIG['candidate_multiple'] * beam_size, flat.shape[1] - 1)
    scores_buf, indices = lax.top_k(flat, k)
    beams_buf = indices // vocab_size
    indices_buf = indices % vocab_size
    return scores_buf, indices_buf, beams_buf
