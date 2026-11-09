"""[bf16 variant] Residual vector quantization (EnCodec): per-codebook nearest-codeword argmax, embedding lookup, residual subtract."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'encodec_residual_vq',
    'batch': 16,
    'dim': 128,
    'time': 16384,
    'num_quantizers': 8,
    'codebook_size': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_e = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k_x, (CONFIG['batch'], CONFIG['dim'], CONFIG['time']), dtype)
    # ponytail: unit-normal codebook like torch randn init; 0.02 scale would make every codeword ~equidistant
    embed = jax.random.normal(
        k_e, (CONFIG['num_quantizers'], CONFIG['codebook_size'], CONFIG['dim']), dtype)
    return x, embed


def _quantize(flat, embed):
    scaled = jnp.sum(flat * flat, axis=1, keepdims=True)
    cross = jnp.matmul(flat, embed.T, precision=lax.Precision.HIGHEST)
    e2 = jnp.sum(embed * embed, axis=1)[None, :]
    dist = -(scaled - 2 * cross + e2)
    return jnp.argmax(dist, axis=-1).astype(jnp.int32)


def workload(embeddings, embed):
    x = jnp.transpose(embeddings, (0, 2, 1))  # [B, T, D]
    b, t, d = x.shape
    residual = x.reshape(b * t, d)
    quantized_out = jnp.zeros_like(residual)
    all_indices = []
    for q in range(embed.shape[0]):
        idx = _quantize(residual, embed[q])
        quantized = jnp.take(embed[q], idx, axis=0)
        residual = residual - quantized
        quantized_out = quantized_out + quantized
        all_indices.append(idx.reshape(b, t))
    codes = jnp.stack(all_indices)
    return codes, jnp.transpose(quantized_out.reshape(b, t, d), (0, 2, 1))
