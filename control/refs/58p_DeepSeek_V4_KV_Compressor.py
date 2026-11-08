import jax
import jax.numpy as jnp
import numpy as np
import dataclasses
import enum
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import numpy as np
import dataclasses
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


def _jaxbench_cdiv(a, b):
  """Ceiling division: JAX's own cdiv, copied so these inputs import no kernel library."""
  if isinstance(a, int) and isinstance(b, int):
    return (a + b - 1) // b
  return jax.lax.div(a + (b - 1), b)


LANE = 128  # bytes per sub-slot / TPU lane width


SLOT_PACK = 4  # sub-slots packed into one physical HBM slot row


N_FIELDS = 2  # values stored per token: kv + score


FP32_BYTES = 4


class Mode(enum.Enum):
  HCA = "hca"
  CSA = "csa"
  CSA_INDEXER = "csa_indexer"


_MODE_DEFAULTS = {
    Mode.HCA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=128,
        quant_block=0,
        overlap=False,
        has_rope_cache=False,
    ),
    Mode.CSA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=64,
        overlap=True,
        has_rope_cache=True,
    ),
    Mode.CSA_INDEXER: dict(
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=128,
        overlap=True,
        has_rope_cache=False,
    ),
}


def physical_page_size(
    mode: Mode, kv_cache_block_size: int, compress_ratio: int
) -> int:
  """Rows per page of the uint8 cache array allocated for `mode`.

  This must mirror the shapes ``KVCacheManager._create_dsv4_kv_caches``
  allocates, because ``compress_norm_rope_store`` re-derives the whole page
  geometry from ``cache.shape[1]``. Anything that computes slot indices for
  that array (the compressor wrapper's state-cache ``block_size``, and
  ``derive_metadata``) has to agree with it or writes land on the wrong page.
  """
  # Compressed KV tokens per page (vLLM's ``spec.storage_block_size``).
  storage_block_size = kv_cache_block_size // compress_ratio
  if mode is Mode.CSA_INDEXER:
    # (N, T // 4, 4, 256) u8
    return storage_block_size // SLOT_PACK
  if mode is Mode.CSA:
    # (N, T, 4, 128) u8
    return storage_block_size
  # HCA (N, T * 2, 4, 128) u8
  return storage_block_size * 2


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TileSizes:
  """Tile sizes for the kernel."""
  tile_n: int


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dimensions:
  """True inputs only. Anything derivable is a property below."""

  mode: Mode = dataclasses.field(metadata=dict(static=True))
  size_n: int  # num_tokens
  head_dim: int
  rope_head_dim: int
  compress_ratio: int
  physical_page_size: int
  state_physical_page_size: int
  quant_block: int
  overlap: bool
  has_rope_cache: bool
  rms_eps: float = 1e-6

  cos_sin_dtype: jax.typing.DTypeLike = jnp.float32
  rope_width: int = 128

  @property
  def is_quantized(self) -> bool:
    return self.quant_block > 0

  @property
  def has_rope(self) -> bool:
    return self.rope_head_dim > 0

  @property
  def nope_dtype(self) -> jax.typing.DTypeLike:
    return jnp.uint8 if self.is_quantized else jnp.bfloat16

  @property
  def rope_dtype(self) -> jax.typing.DTypeLike:
    return self.nope_dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Configs:
  """Configuration for the kernel."""
  tile_sizes: TileSizes
  dims: Dimensions

  # --- factory ---------------------------------------------------------------
  @classmethod
  def make(
      cls,
      mode: Mode,
      *,
      size_n,
      physical_page_size,
      state_physical_page_size=None,
      rms_eps=1e-6,
      tile_n=4,
      **overrides,
  ) -> "Configs":
    """Build a config for `mode`; `overrides` replace per-mode defaults.

    `state_physical_page_size` defaults to `physical_page_size`, i.e. the
    state and the compressed KV share one buffer.
    """
    actual_overrides = {**_MODE_DEFAULTS[mode], **overrides}
    if mode == Mode.HCA:
      actual_overrides["quant_block"] = 0

    dims = Dimensions(
        mode=mode,
        size_n=size_n,
        physical_page_size=physical_page_size,
        state_physical_page_size=(
            physical_page_size
            if state_physical_page_size is None
            else state_physical_page_size
        ),
        rms_eps=rms_eps,
        **actual_overrides,
    )
    return cls(tile_sizes=TileSizes(tile_n=tile_n), dims=dims)

  # --- compute tiling (logical, in LANE-wide tiles) --------------------------
  @property
  def overlap_factor(self) -> int:
    """State copies kept per step: 1, or 2 when windows overlap (prev+curr)."""
    return 1 + int(self.dims.overlap)

  @property
  def state_width(self) -> int:
    """Width of one stored field (head_dim, doubled when overlapping)."""
    return self.overlap_factor * self.dims.head_dim

  @property
  def head_tiles(self) -> int:
    """head_dim split into LANE-wide compute tiles (old: slots_per_part)."""
    return self.dims.head_dim // LANE

  @property
  def window(self) -> int:
    """Timesteps compressed together (x overlap_factor when overlapping)."""
    return self.dims.compress_ratio * self.overlap_factor

  # --- rope sub-layout -------------------------------------------------------
  @property
  def nope_dim(self) -> int:
    """Width of the non-rope (nope) part of a head."""
    return self.dims.head_dim - self.dims.rope_head_dim

  @property
  def nope_store_dim(self) -> int:
    """Dimension of the nope storage (contains rope if no separate rope cache)."""
    return (
        self.dims.head_dim - self.dims.rope_head_dim
        if self.dims.has_rope_cache
        else self.dims.head_dim
    )

  @property
  def half_rope(self) -> int:
    """cos/sin split point (half the rope dim)."""
    return self.dims.rope_head_dim // 2

  @property
  def rope_slot(self) -> int:
    """Head-tile holding the rope channels (the last one)."""
    return self.head_tiles - 1

  # --- output record size ----------------------------------------------------
  @property
  def record_bytes(self) -> int:
    """Bytes in one packed output record (old: total_bytes_out)."""
    if not self.dims.is_quantized:
      return self.dims.head_dim * 2  # bf16
    # fp8 payload + scale + padding, capped to a fixed record width.
    return 256 if self.dims.head_dim == LANE else 512

  @property
  def record_rows(self) -> int:
    """Packed physical rows per output record."""
    return _jaxbench_cdiv(self.record_bytes, self.row_size_bytes)

  # --- HBM storage packing ---------------------------------------------------
  @property
  def row_size_bytes(self) -> int:
    """Size of one physical HBM row in bytes."""
    return self.hbm_pack * self.last_dim_size

  @property
  def hbm_pack(self) -> int:
    """Sub-slots physically packed per slot row, <= SLOT_PACK."""
    return SLOT_PACK

  @property
  def last_dim_size(self) -> int:
    """Size of the last HBM cache dimension in bytes."""
    if self.dims.mode == Mode.CSA_INDEXER:
      return 2 * LANE
    return LANE

  @property
  def cache_last_dims(self) -> tuple[int, ...]:
    return (self.hbm_pack, self.last_dim_size)

  # --- slot translation helpers -----------------------------------------
  @property
  def tokens_in_second_minor(self) -> int:
    """Divisor to map wrapper row offset to new physical row offset."""
    tokens_per_row = self.row_size_bytes // self.record_bytes
    return max(1, tokens_per_row)

  @property
  def kv_stride(self) -> int:
    """Token stride in logical slot units."""
    if self.record_bytes <= self.row_size_bytes:
      return 1
    return self.record_rows

  # --- block sizes and physical page size ------------------------------------
  @property
  def physical_page_size(self) -> int:
    """Number of physical HBM rows per page of the compressed-KV array."""
    return self.dims.physical_page_size

  @property
  def state_physical_page_size(self) -> int:
    """Number of physical HBM rows per page of the *state* array."""
    return self.dims.state_physical_page_size

  @property
  def state_rows_per_token(self) -> int:
    """Number of physical HBM rows occupied by one token's full state (kv + score)."""
    state_bytes = N_FIELDS * self.state_width * FP32_BYTES
    return state_bytes // self.row_size_bytes

  @property
  def field_rows(self) -> int:
    """Number of physical HBM rows spanned by one field during gather."""
    field_bytes = self.dims.head_dim * FP32_BYTES
    return _jaxbench_cdiv(field_bytes, self.row_size_bytes)

  @property
  def state_block_size(self) -> int:
    """Number of state tokens per page of the state array."""
    return self.state_physical_page_size // self.state_rows_per_token

  @property
  def kv_block_size(self) -> int:
    """Number of compressed KV tokens per page."""
    page_bytes = self.physical_page_size * self.row_size_bytes
    return page_bytes // self.record_bytes

  @property
  def rope_page_size(self) -> int:
    """Number of physical HBM rows per page in RoPE cache."""
    return self.physical_page_size // self.hbm_pack

  @property
  def pages_to_buffer_per_token(self) -> int:
    """Pages that must be resident to cover one token's window (+1 guard)."""
    return _jaxbench_cdiv(self.window, self.state_block_size) + 1

  # --- shapes (single source of truth for every reshape / BlockSpec) ---------
  @property
  def _tile_n(self) -> int:
    return self.tile_sizes.tile_n

  def window_shape(self) -> tuple[int, ...]:
    """f32 window scratch: (fields, tile, window, head_tiles, lane)."""
    return (N_FIELDS, self._tile_n, self.window, self.head_tiles, LANE)

  def window_bytes_shape(self) -> tuple[int, ...]:
    """uint8 view of the window scratch; each f32 lane -> FP32_BYTES rows.

    Note: that trailing FP32_BYTES (4) is bytes-per-f32, NOT the HBM SLOT_PACK
    (also 4) -- they're numerically equal but mean different things.

    Returns:
      The shape of the window bytes scratch.
    """
    return (
        N_FIELDS,
        self._tile_n,
        self.window,
        self.head_tiles,
        FP32_BYTES,
        LANE,
    )

  def output_shape(self) -> tuple[int, ...]:
    """Packed output tile / VMEM block: (tile, record_rows) + cache_last_dims."""
    return (self._tile_n, self.record_rows) + self.cache_last_dims

  def page_buffer_shape(self) -> tuple[int, ...]:
    """Page-buffer VMEM block: (tile, pages, state page rows) + cache_last_dims.

    Pages are DMA'd whole out of the *state* array, so this is sized by
    `state_physical_page_size`, not by the compressed-KV page.
    """
    return (
        self._tile_n,
        self.pages_to_buffer_per_token,
        self.state_physical_page_size,
    ) + self.cache_last_dims

  def rope_output_shape(self) -> tuple[int, ...]:
    """RoPE output VMEM block: (tile, 4, lane)."""
    assert self.dims.has_rope_cache
    return (self._tile_n, 4, LANE)

  def cos_sin_shape(self) -> tuple[int, ...]:
    """cos/sin VMEM block: (tile, rope_head_dim)."""
    return (self._tile_n, self.dims.rope_head_dim)

  def cache_shape(self, num_pages: int) -> tuple[int, ...]:
    """Shape of the global HBM KV cache."""
    return (num_pages, self.physical_page_size) + self.cache_last_dims

  def state_cache_shape(self, num_pages: int) -> tuple[int, ...]:
    """Shape of the global HBM array hosting the f32 compressor state."""
    return (num_pages, self.state_physical_page_size) + self.cache_last_dims

  def rope_cache_shape(self, num_pages: int) -> tuple[int, ...]:
    """Shape of the global HBM RoPE cache."""
    assert self.dims.has_rope_cache
    return (num_pages, self.rope_page_size, 4, 128)


def generate_kv_slot_mapping(
    num_boundary_tokens: int,
    num_pages: int,
    page_size: int,
    slots_per_part_out: int,
) -> jax.Array:
  """Generates kv_slot_mapping for boundary tokens."""
  kv_slot_mapping_np = np.full((num_boundary_tokens,), -1, dtype=np.int32)
  boundary_count = 0
  total_slots_needed = num_boundary_tokens * slots_per_part_out
  pages_needed = (total_slots_needed + page_size - 1) // page_size
  start_page = max(0, num_pages - pages_needed)

  for t in range(num_boundary_tokens):
    kv_slot_mapping_np[t] = start_page * page_size + boundary_count
    boundary_count += slots_per_part_out
  return jnp.array(kv_slot_mapping_np, dtype=jnp.int32)


CONFIGS = {
    "csa_small": {
        "name": "csa_small",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa",
        "batch_size": 16,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 256,
        "quant_block": 64,
        "overlap": True,
        "mode": "prefill",
    },
    "csa_decode_small": {
        "name": "csa_decode_small",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa",
        "batch_size": 16,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 256,
        "quant_block": 64,
        "overlap": True,
        "mode": "decode",
        "history_len": 64,
    },
    "csa_prefill_bs512": {
        "name": "csa_prefill_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 256,
        "quant_block": 64,
        "overlap": True,
        "mode": "prefill",
    },
    "csa_decode_bs512": {
        "name": "csa_decode_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 256,
        "quant_block": 64,
        "overlap": True,
        "mode": "decode",
        "history_len": 1024,
    },
    "hca_prefill_bs512": {
        "name": "hca_prefill_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "hca",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 128,
        "physical_page_size": 128,
        "quant_block": 0,
        "overlap": False,
        "mode": "prefill",
    },
    "hca_decode_bs512": {
        "name": "hca_decode_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "hca",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 128,
        "physical_page_size": 128,
        "quant_block": 0,
        "overlap": False,
        "mode": "decode",
        "history_len": 1024,
    },
    "csa_indexer_prefill_bs512": {
        "name": "csa_indexer_prefill_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa_indexer",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 128,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 32,
        "quant_block": 128,
        "overlap": True,
        "mode": "prefill",
    },
    "csa_indexer_decode_bs512": {
        "name": "csa_indexer_decode_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa_indexer",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 128,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 32,
        "quant_block": 128,
        "overlap": True,
        "mode": "decode",
        "history_len": 1024,
    },
}


CONFIG = CONFIGS["csa_small"]


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (cache, positions, block_table, token_to_req_indices, kv_slot_mapping, rms_weight, cos_sin_cache, rope_cache)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  case_type = cfg.get("case_type", "csa")
  batch_size = cfg.get("batch_size", 512)
  head_dim = cfg.get("head_dim", 512)
  compress_ratio = cfg.get("compress_ratio", 4)
  physical_page_size = cfg.get("physical_page_size", 256)
  rope_head_dim = cfg.get("rope_head_dim", 64)
  quant_block = cfg.get(
      "quant_block",
      64 if case_type == "csa" else (128 if case_type == "csa_indexer" else 0),
  )
  mode = cfg.get("mode", "prefill")
  history_len = cfg.get("history_len", 1024)
  rms_eps = cfg.get("rms_eps", 1e-6)
  tile_n = 4

  if case_type == "hca":
    config_mode = Mode.HCA
  elif case_type == "csa":
    config_mode = Mode.CSA
  elif case_type == "csa_indexer":
    config_mode = Mode.CSA_INDEXER
  else:
    raise ValueError(f"Unknown case_type: {case_type}")

  cfgs = Configs.make(
      config_mode,
      size_n=batch_size,
      physical_page_size=physical_page_size,
      rms_eps=rms_eps,
      tile_n=tile_n,
      head_dim=head_dim,
      rope_head_dim=rope_head_dim,
      compress_ratio=compress_ratio,
      quant_block=quant_block,
  )

  tokens_per_page = cfgs.state_block_size
  rng = np.random.default_rng(42)

  if mode == "prefill":
    num_boundary_tokens = batch_size // compress_ratio
    assert (
        num_boundary_tokens % 4 == 0
    ), f"num_boundary_tokens ({num_boundary_tokens}) must be divisible by 4"
    max_pos = num_boundary_tokens * compress_ratio
    pages_for_state = max(1, (max_pos + tokens_per_page - 1) // tokens_per_page)

    total_rows_needed = num_boundary_tokens * cfgs.record_rows
    pages_for_kv_cache = max(
        1,
        (total_rows_needed + cfgs.physical_page_size - 1)
        // cfgs.physical_page_size,
    )
    num_pages = pages_for_state + pages_for_kv_cache

    block_table = jnp.arange(pages_for_state, dtype=jnp.int32).reshape(
        1, pages_for_state
    )

    boundary_positions = (
        np.arange(num_boundary_tokens) + 1
    ) * compress_ratio - 1
    positions_np = np.pad(
        boundary_positions,
        (0, batch_size - num_boundary_tokens),
        constant_values=0,
    )
    positions = jnp.array(positions_np, dtype=jnp.int32)
    token_to_req_indices = jnp.zeros((batch_size,), dtype=jnp.int32)

    kv_slot_mapping_filtered = generate_kv_slot_mapping(
        num_boundary_tokens,
        num_pages,
        cfgs.kv_block_size * cfgs.kv_stride,
        cfgs.kv_stride,
    )
    kv_slot_mapping = jnp.pad(
        kv_slot_mapping_filtered,
        (0, batch_size - num_boundary_tokens),
        constant_values=-1,
    )

  elif mode == "decode":
    boundary_ratio = cfg.get("boundary_ratio", 1.0 / compress_ratio)
    num_boundary_tokens = int(batch_size * boundary_ratio)
    if boundary_ratio > 0 and num_boundary_tokens == 0:
      num_boundary_tokens = 1
    num_boundary_tokens = max(
        4, ((num_boundary_tokens + tile_n - 1) // tile_n) * tile_n
    )
    num_boundary_tokens = min(num_boundary_tokens, batch_size)

    max_pos = history_len
    pages_for_state = max(1, (max_pos + tokens_per_page - 1) // tokens_per_page)

    total_rows_needed = num_boundary_tokens * cfgs.record_rows
    pages_for_kv_cache = max(
        1,
        (total_rows_needed + cfgs.physical_page_size - 1)
        // cfgs.physical_page_size,
    )
    num_pages = batch_size * pages_for_state + pages_for_kv_cache

    block_table = jnp.arange(
        batch_size * pages_for_state, dtype=jnp.int32
    ).reshape(batch_size, pages_for_state)

    max_boundary_idx = max(1, history_len // compress_ratio)
    boundary_reqs = rng.choice(
        batch_size,
        size=num_boundary_tokens,
        replace=(batch_size < num_boundary_tokens),
    )

    kv_slot_mapping_filtered = generate_kv_slot_mapping(
        num_boundary_tokens,
        num_pages,
        cfgs.kv_block_size * cfgs.kv_stride,
        cfgs.kv_stride,
    )

    positions_active = np.zeros((num_boundary_tokens,), dtype=np.int32)
    for idx in range(num_boundary_tokens):
      q = rng.integers(0, max_boundary_idx)
      positions_active[idx] = (q + 1) * compress_ratio - 1

    positions_np = np.pad(
        positions_active,
        (0, batch_size - num_boundary_tokens),
        constant_values=0,
    )
    token_to_req_indices_np = np.pad(
        boundary_reqs,
        (0, batch_size - num_boundary_tokens),
        constant_values=0,
    )
    kv_slot_mapping_np = np.pad(
        kv_slot_mapping_filtered,
        (0, batch_size - num_boundary_tokens),
        constant_values=-1,
    )

    positions = jnp.array(positions_np, dtype=jnp.int32)
    token_to_req_indices = jnp.array(token_to_req_indices_np, dtype=jnp.int32)
    kv_slot_mapping = jnp.array(kv_slot_mapping_np, dtype=jnp.int32)
  else:
    raise ValueError(f"Unknown mode: {mode}")

  key = jax.random.key(42)
  k1, k2, k3 = jax.random.split(key, 3)
  rms_weight = jax.random.uniform(
      k1, (head_dim,), minval=0.5, maxval=1.5, dtype=jnp.float32
  )
  max_p = int(np.max(positions_np)) + 1 if len(positions_np) > 0 else batch_size
  cos_sin_len = max(max_p, batch_size)
  cos_sin_cache = jax.random.uniform(
      k2,
      (cos_sin_len, rope_head_dim),
      minval=0.1,
      maxval=0.3,
      dtype=jnp.float32,
  )

  num_state_pages = num_pages - pages_for_kv_cache
  state_rows = cfgs.physical_page_size // cfgs.state_block_size
  sample_pages = min(num_state_pages, 64)
  state_f32 = jax.random.uniform(
      k3,
      (sample_pages, cfgs.state_block_size, state_rows, cfgs.last_dim_size),
      minval=0.5,
      maxval=1.5,
      dtype=jnp.float32,
  )
  state_u8 = (
      jax.lax.bitcast_convert_type(state_f32, jnp.uint8)
      .transpose(0, 1, 2, 4, 3)
      .reshape(
          sample_pages,
          cfgs.physical_page_size,
          cfgs.hbm_pack,
          cfgs.last_dim_size,
      )
  )
  if num_state_pages > sample_pages:
    reps = (num_state_pages + sample_pages - 1) // sample_pages
    state_pages = jnp.tile(state_u8, (reps, 1, 1, 1))[:num_state_pages]
  else:
    state_pages = state_u8
  kv_pages = jnp.zeros(
      (
          pages_for_kv_cache,
          cfgs.physical_page_size,
          cfgs.hbm_pack,
          cfgs.last_dim_size,
      ),
      dtype=jnp.uint8,
  )
  cache = jnp.concatenate([state_pages, kv_pages], axis=0)
  if cfgs.dims.has_rope_cache:
    rope_cache = jnp.zeros(cfgs.rope_cache_shape(num_pages), dtype=jnp.uint8)
  else:
    rope_cache = None

  return (
      cache,
      positions,
      block_table,
      token_to_req_indices,
      kv_slot_mapping,
      rms_weight,
      cos_sin_cache,
      rope_cache,
  )


import jax
import jax.numpy as jnp


def _interleaved_rope_vector(x, cos_val_32, sin_val_32):
  # x: (N, 128), cos_val_32: (N, 32), sin_val_32: (N, 32)
  N = x.shape[0]
  iota = jnp.arange(128)
  swap_indices = jnp.bitwise_xor(iota, 1)
  x_swapped = x[:, swap_indices]

  cos_32 = cos_val_32.astype(x.dtype)
  sin_32 = sin_val_32.astype(x.dtype)
  ones_32 = jnp.ones((N, 32), dtype=x.dtype)
  zeros_32 = jnp.zeros((N, 32), dtype=x.dtype)

  cos_pairs = jnp.concatenate([ones_32, cos_32], axis=-1)
  sin_pairs = jnp.concatenate([zeros_32, sin_32], axis=-1)

  dup_indices = iota // 2
  cos_dup = cos_pairs[:, dup_indices]
  sin_dup = sin_pairs[:, dup_indices]

  alt_mask = ((iota % 2) * 2 - 1).astype(x.dtype)[None, :]
  sin_alt = sin_dup * alt_mask

  return x * cos_dup + x_swapped * sin_alt


def _quantize_fp8_tiled(x, block_size):
  # x: (N, S, 128) f32
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  N, S, width = x.shape
  num_blocks = width // block_size

  x_blocks = x.reshape(N, S, num_blocks, block_size)
  amax = jnp.clip(
      jnp.max(jnp.abs(x_blocks), axis=-1, keepdims=True), 1e-4, None
  )
  log2_val = jnp.log2(amax / fp8_max)
  scale = jnp.exp2(jnp.ceil(log2_val))  # (N, S, num_blocks, 1) f32

  q_blocks = (x_blocks * (1.0 / scale)).astype(jnp.float8_e4m3fn)
  q = q_blocks.reshape(N, S, width)
  scale = jnp.squeeze(scale, axis=-1)  # (N, S, num_blocks)

  scale_u32 = jax.lax.bitcast_convert_type(scale, jnp.uint32)
  scale_exp = scale_u32 >> 23
  scale_u8 = scale_exp.astype(jnp.uint8)
  return q, scale_u8


def _pack_nope_tiled(
    q, scale, nope_dim, block_size, nope_width_bytes=512, last_dim_size=128
):
  # q: (N, S, 128) fp8, scale: (N, S, num_blocks) u8
  N = q.shape[0]
  q_bytes = jax.lax.bitcast_convert_type(q, jnp.uint8)
  scale_bytes = (
      scale
      if scale.dtype == jnp.uint8
      else jax.lax.bitcast_convert_type(scale, jnp.uint8)
  )

  q_flat = q_bytes.reshape(N, -1)
  scale_flat = scale_bytes.reshape(N, -1)

  if nope_dim < q_flat.shape[1]:
    q_nope = q_flat[:, :nope_dim]
  else:
    q_nope = q_flat

  nope_blocks = (nope_dim + block_size - 1) // block_size
  if nope_blocks < scale_flat.shape[1]:
    scale_nope = scale_flat[:, :nope_blocks]
  else:
    scale_nope = scale_flat

  pad_size = nope_width_bytes - (nope_dim + nope_blocks)
  zeros = jnp.zeros((N, pad_size), dtype=jnp.uint8)

  nope_record_padded = jnp.concatenate([q_nope, scale_nope, zeros], axis=1)
  return nope_record_padded.reshape(N, -1, last_dim_size)


def _pack_rope_tiled(rope_slot_ropped, rope_head_dim_actual, rope_width=128):
  # rope_slot_ropped: (N, 128) f32
  start = 128 - rope_head_dim_actual
  rope_val = rope_slot_ropped[:, start:]  # (N, rope_head_dim)
  rope_bf16 = rope_val.astype(jnp.bfloat16)
  rope_width_bf16 = rope_width // 2

  if rope_head_dim_actual < rope_width_bf16:
    rope_padded_bf16 = jnp.pad(
        rope_bf16,
        ((0, 0), (0, rope_width_bf16 - rope_head_dim_actual)),
    )
  else:
    rope_padded_bf16 = rope_bf16

  rope_u16 = jax.lax.bitcast_convert_type(rope_padded_bf16, jnp.uint16)
  rope_i32 = rope_u16.astype(jnp.int32)
  rope_hi = ((rope_i32 >> 8) & 0xFF).astype(jnp.uint8)
  rope_lo = (rope_i32 & 0xFF).astype(jnp.uint8)
  rope_uint8_2d = jnp.concatenate([rope_hi, rope_lo], axis=-1)  # (N, 128)
  return rope_uint8_2d


def workload(
    cache,
    positions,
    block_table,
    token_to_req_indices,
    kv_slot_mapping,
    rms_weight,
    cos_sin_cache,
    rope_cache=None,
):
  head_dim = rms_weight.shape[0]
  rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0
  num_tokens = positions.shape[0]
  physical_page_size = cache.shape[1]

  if head_dim == 128:
    case_type = "csa_indexer"
    compress_ratio = 4
    overlap = True
    quant_block = 128
    has_rope_cache = False
    last_dim_size = 256
    record_bytes = 256
    record_rows = 1
    state_rows_per_token = 2
    field_rows = 1
    kv_block_size = (physical_page_size * 4 * 256) // 256
  elif rope_cache is None:
    case_type = "hca"
    compress_ratio = 128
    overlap = False
    quant_block = 0
    has_rope_cache = False
    last_dim_size = 128
    record_bytes = 1024
    record_rows = 2
    state_rows_per_token = 8
    field_rows = 4
    kv_block_size = (physical_page_size * 4 * 128) // 1024
  else:
    case_type = "csa"
    compress_ratio = 4
    overlap = True
    quant_block = 64
    has_rope_cache = True
    last_dim_size = 128
    record_bytes = 512
    record_rows = 1
    state_rows_per_token = 16
    field_rows = 4
    kv_block_size = (physical_page_size * 4 * 128) // 512

  if block_table.ndim == 2:
    block_table_stride = block_table.shape[1]
    flat_block_table = block_table.reshape(-1)
  else:
    block_table_stride = block_table.shape[0]
    flat_block_table = block_table

  rms_eps = 1e-6
  overlap_factor = 1 + int(overlap)
  window = compress_ratio * overlap_factor
  head_tiles = head_dim // 128
  state_block_size = physical_page_size // state_rows_per_token

  # Gather window positions and valid masks
  w_arange = jnp.arange(window, dtype=jnp.int32)
  pos_start = positions - window + 1  # (N,)
  pos_w = pos_start[:, None] + w_arange[None, :]  # (N, W)
  valid_w = pos_w >= 0  # (N, W)

  block_idx_w = pos_w // state_block_size
  safe_block_idx_w = jnp.maximum(block_idx_w, 0)
  offset_in_block = pos_w % state_block_size  # (N, W)

  req_idx = token_to_req_indices[:, None]  # (N, 1)
  bt_idx = req_idx * block_table_stride + safe_block_idx_w  # (N, W)
  pages_w = flat_block_table[bt_idx]  # (N, W)

  if case_type == "csa_indexer":
    token_row_start = offset_in_block * 2
    kv_row = token_row_start
    score_row = token_row_start + 1

    row_kv = cache[pages_w, kv_row]  # (N, W, 4, 256)
    row_score = cache[pages_w, score_row]  # (N, W, 4, 256)

    is_prev = w_arange < (window // 2)  # (W,)
    is_prev_b = is_prev[None, :, None, None]

    val_kv_u8 = jnp.where(
        is_prev_b, row_kv[:, :, :, 0:128], row_kv[:, :, :, 128:256]
    )
    val_score_u8 = jnp.where(
        is_prev_b, row_score[:, :, :, 0:128], row_score[:, :, :, 128:256]
    )

    val_kv_u8 = jnp.where(valid_w[:, :, None, None], val_kv_u8, 0)
    val_score_u8 = jnp.where(valid_w[:, :, None, None], val_score_u8, 0)

    kv_window_u8 = val_kv_u8[:, :, None, :, :]  # (N, W, 1, 4, 128)
    score_window_u8 = val_score_u8[:, :, None, :, :]  # (N, W, 1, 4, 128)
  else:
    d_arange = jnp.arange(head_tiles, dtype=jnp.int32)
    if overlap:
      is_prev = w_arange < (window // 2)
      kv_slot_start_row = jnp.where(is_prev, 0, field_rows)[None, :, None]
      score_slot_start_row = jnp.where(is_prev, 2 * field_rows, 3 * field_rows)[
          None, :, None
      ]
    else:
      kv_slot_start_row = 0
      score_slot_start_row = field_rows

    base_row = (offset_in_block * state_rows_per_token)[:, :, None]
    kv_src_row = (
        base_row + kv_slot_start_row + d_arange[None, None, :]
    )  # (N, W, head_tiles)
    score_src_row = (
        base_row + score_slot_start_row + d_arange[None, None, :]
    )  # (N, W, head_tiles)

    pages_b = pages_w[:, :, None]
    kv_window_u8 = cache[pages_b, kv_src_row]  # (N, W, head_tiles, 4, 128)
    score_window_u8 = cache[
        pages_b, score_src_row
    ]  # (N, W, head_tiles, 4, 128)

  # Convert byte-transposed (..., 4, 128) uint8 to (..., 128) float32
  kv_val = jax.lax.bitcast_convert_type(
      jnp.swapaxes(kv_window_u8, -2, -1), jnp.float32
  )
  scores_val = jax.lax.bitcast_convert_type(
      jnp.swapaxes(score_window_u8, -2, -1), jnp.float32
  )

  # Windowed softmax
  mask_b = valid_w[:, :, None, None]
  neg_inf = jnp.array(-jnp.inf, dtype=jnp.float32)
  masked_scores = jnp.where(mask_b, scores_val, neg_inf)
  weights = jax.nn.softmax(masked_scores, axis=1)
  kv_val = jnp.where(mask_b, kv_val, 0.0)
  compressed = jnp.sum(weights * kv_val, axis=1)  # (N, head_tiles, 128)

  # RMS norm
  rms_weight_tiled = rms_weight.astype(jnp.float32).reshape(head_tiles, 128)
  variance = jnp.mean(jnp.square(compressed), axis=(1, 2), keepdims=True)
  normed = (
      compressed
      * jax.lax.rsqrt(variance + rms_eps)
      * rms_weight_tiled[None, :, :]
  )

  # RoPE
  rope_ropped = None
  if rope_head_dim > 0:
    rope_slot = head_tiles - 1
    rope_val = normed[:, rope_slot]  # (N, 128)

    compressed_pos = (positions // compress_ratio) * compress_ratio
    cos_sin = cos_sin_cache[compressed_pos]  # (N, rope_head_dim)
    half_rope = rope_head_dim // 2
    cos_val = cos_sin[:, :half_rope]
    sin_val = cos_sin[:, half_rope:]

    rope_ropped = _interleaved_rope_vector(rope_val, cos_val, sin_val)
    if head_tiles > 1:
      normed = jnp.concatenate(
          [normed[:, :rope_slot], rope_ropped[:, None, :]], axis=1
      )
    else:
      normed = rope_ropped[:, None, :]

  # Active token mask mimicking dynamic grid_size and kv_slot >= 0
  tile_n = 4
  valid_slot = kv_slot_mapping >= 0
  token_indices = jnp.arange(num_tokens, dtype=jnp.int32)
  max_idx = jnp.max(jnp.where(valid_slot, token_indices, -1))
  grid_size = jnp.where(max_idx >= 0, (max_idx + 1 + tile_n - 1) // tile_n, 0)
  active_mask = valid_slot & (token_indices < grid_size * tile_n)

  # Pack and Store
  if quant_block > 0:
    nope_store_dim = head_dim - rope_head_dim if has_rope_cache else head_dim
    q, scale = _quantize_fp8_tiled(normed, quant_block)
    nope_val_padded = _pack_nope_tiled(
        q,
        scale,
        nope_store_dim,
        quant_block,
        nope_width_bytes=record_bytes,
        last_dim_size=last_dim_size,
    )

    if case_type == "csa_indexer":
      # nope_val_padded: (N, 1, 256)
      vals = nope_val_padded[:, 0, :]  # (N, 256)
      p = jnp.where(active_mask, kv_slot_mapping // kv_block_size, 0)
      s_row = jnp.where(active_mask, (kv_slot_mapping % kv_block_size) // 4, 0)
      sub_idx = jnp.where(active_mask, kv_slot_mapping % 4, 0)

      existing = cache[p, s_row, sub_idx]
      write_vals = jnp.where(active_mask[:, None], vals, existing)
      out_cache = cache.at[p, s_row, sub_idx].set(write_vals)
      return out_cache
    else:
      # CSA: nope_val_padded: (N, 4, 128)
      p = jnp.where(active_mask, kv_slot_mapping // physical_page_size, 0)
      s = jnp.where(active_mask, kv_slot_mapping % physical_page_size, 0)
      existing = cache[p, s]
      write_vals = jnp.where(active_mask[:, None, None], nope_val_padded, existing)
      out_cache = cache.at[p, s].set(write_vals)

      # RoPE cache update
      rope_val_padded = _pack_rope_tiled(rope_ropped, rope_head_dim, 128)  # (N, 128)
      rope_page_size = physical_page_size // 4
      rope_slot_idx = kv_slot_mapping // 4
      rope_p = jnp.where(active_mask, rope_slot_idx // rope_page_size, 0)
      rope_s_row = jnp.where(active_mask, rope_slot_idx % rope_page_size, 0)
      rope_sub = jnp.where(active_mask, kv_slot_mapping % 4, 0)

      existing_rope = rope_cache[rope_p, rope_s_row, rope_sub]
      write_rope = jnp.where(
          active_mask[:, None], rope_val_padded, existing_rope
      )
      out_rope = rope_cache.at[rope_p, rope_s_row, rope_sub].set(write_rope)
      return out_cache, out_rope
  else:
    # HCA: bf16 bitcast to uint8 -> (N, 2, 4, 128)
    out_bytes = jax.lax.bitcast_convert_type(
        normed.astype(jnp.bfloat16), jnp.uint8
    ).reshape(num_tokens, record_rows, 4, 128)

    p = jnp.where(active_mask, kv_slot_mapping // physical_page_size, 0)
    s = jnp.where(active_mask, kv_slot_mapping % physical_page_size, 0)

    existing_0 = cache[p, s]
    existing_1 = cache[p, s + 1]
    write_0 = jnp.where(active_mask[:, None, None], out_bytes[:, 0], existing_0)
    write_1 = jnp.where(active_mask[:, None, None], out_bytes[:, 1], existing_1)

    out_cache = cache.at[p, s].set(write_0)
    out_cache = out_cache.at[p, s + 1].set(write_1)
    return out_cache
