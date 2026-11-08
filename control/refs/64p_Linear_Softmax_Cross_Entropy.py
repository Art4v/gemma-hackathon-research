import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


CONFIGS = {
    "ce_small": {
        "name": "ce_small",
        "model": "Generic-Small",
        "operator": "linear_softmax_cross_entropy_loss",
        "batch_tokens": 1024,
        "hidden_dim": 1024,
        "vocab_size": 32768,
    },
    "llama3_8b": {
        "name": "llama3_8b",
        "model": "Llama-3.1-8B",
        "operator": "linear_softmax_cross_entropy_loss",
        "batch_tokens": 4096,
        "hidden_dim": 4096,
        "vocab_size": 128256,
    },
    "llama3_8b_large": {
        "name": "llama3_8b_large",
        "model": "Llama-3.1-8B",
        "operator": "linear_softmax_cross_entropy_loss",
        "batch_tokens": 8192,
        "hidden_dim": 4096,
        "vocab_size": 128256,
    },
    "gemma2_9b": {
        "name": "gemma2_9b",
        "model": "Gemma-2-9B",
        "operator": "linear_softmax_cross_entropy_loss",
        "batch_tokens": 4096,
        "hidden_dim": 3584,
        "vocab_size": 256000,
    },
    "qwen2_5_7b": {
        "name": "qwen2_5_7b",
        "model": "Qwen-2.5-7B",
        "operator": "linear_softmax_cross_entropy_loss",
        "batch_tokens": 4096,
        "hidden_dim": 3584,
        "vocab_size": 152064,
    },
    "deepseek_v3": {
        "name": "deepseek_v3",
        "model": "DeepSeek-V3",
        "operator": "linear_softmax_cross_entropy_loss",
        "batch_tokens": 4096,
        "hidden_dim": 7168,
        "vocab_size": 129280,
    },
}


CONFIG = CONFIGS["ce_small"]


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (x, labels, w)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3 = jax.random.split(key, 3)
  B = cfg["batch_tokens"]
  H = cfg["hidden_dim"]
  V = cfg["vocab_size"]

  x = jax.random.normal(k1, (B, H), dtype=dtype)
  w = jax.random.normal(k2, (H, V), dtype=dtype) * 0.02
  labels = jax.random.randint(k3, (B,), 0, V)

  return x, labels, w


import jax
import jax.numpy as jnp


def workload(x, labels, w):
  """Plain JAX implementation of Linear Softmax Cross-Entropy Loss."""
  if x.dtype == jnp.float16:
    x = x.astype(jnp.float32)
  if w.dtype == jnp.float16:
    w = w.astype(jnp.float32)

  b_dim = x.shape[0]
  v_dim = w.shape[1]

  # Choose a batch block size to keep the (b_block_size, V) float32 logits tile
  # comfortably within memory (e.g. 512 * 256000 * 4 bytes = 512 MB).
  b_block_size = 512
  while b_dim % b_block_size != 0 and b_block_size > 1:
    b_block_size //= 2

  num_b_blocks = b_dim // b_block_size
  x_blocks = x.reshape(num_b_blocks, b_block_size, x.shape[1])
  labels_blocks = labels.reshape(num_b_blocks, b_block_size)

  def scan_body(carry, batch_slice):
    del carry
    x_b, labels_b = batch_slice
    logits = jax.lax.dot_general(
        x_b,
        w,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    lse_b = jax.nn.logsumexp(logits, axis=-1)

    valid_mask = (labels_b >= 0) & (labels_b < v_dim)
    safe_labels = jnp.where(valid_mask, labels_b, 0)
    target_logits = jnp.take_along_axis(
        logits, safe_labels[:, None], axis=-1
    ).squeeze(-1)
    target_logits = jnp.where(valid_mask, target_logits, 0.0)

    loss_b = lse_b - target_logits
    return None, (loss_b, lse_b)

  _, (loss_blocks, lse_blocks) = jax.lax.scan(
      scan_body, None, (x_blocks, labels_blocks)
  )

  loss = loss_blocks.reshape(b_dim)
  lse = lse_blocks.reshape(b_dim)

  return jnp.mean(loss), lse
