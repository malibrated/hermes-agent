"""TurboQuant KV cache compression for LLM inference on MLX.

Adapted from the video generation pipeline at:
  /Users/Shared/projects/mlx-video-with-audio-fork/mlx_video/kv_compression.py

Paper: arXiv:2504.19874

Per-head implementation of TurboQuant's two-stage asymmetric compression
optimized for autoregressive LLM inference:

**Stage 1 — MSE-Optimal Quantization:**
  Random orthogonal rotation (QR from Gaussian) maps each head's key vectors
  so each coordinate follows a known Beta distribution.  Lloyd-Max scalar
  quantization with precomputed centroids quantizes each coordinate.

**Stage 2 — QJL Error Correction (1-bit):**
  The quantization residual is projected through a random Gaussian matrix
  and reduced to sign bits.  This is an error correction code for the
  attention score, not reconstruction.

**Asymmetric attention scoring:**
  score = <q, k_mse> + γ · √(π/2)/d · <S·q, sign(S·r)>
  Queries stay full-precision.  Only keys are compressed.

**LLM-specific adaptations vs video pipeline:**
  - Incremental compression: keys compressed one token at a time as the cache
    grows during autoregressive generation
  - TurboQuantKVCache class wraps mlx_lm's cache interface
  - Automatic head dimension detection from model config
  - Shared rotation/JL matrices cached across the generation session

All operations are per-head (dim_head typically 64-128 for LLMs).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None  # type: ignore


PI = math.pi


# ---------------------------------------------------------------------------
# Bit-packing (vectorized, no Python loops over elements)
# ---------------------------------------------------------------------------

def pack_bits(values: mx.array, bits: int) -> mx.array:
    """Pack low-bit integer values into uint32 arrays."""
    vals_per_int = 32 // bits
    *batch_shape, n = values.shape
    pad_n = (vals_per_int - n % vals_per_int) % vals_per_int
    if pad_n > 0:
        padding = mx.zeros((*batch_shape, pad_n), dtype=values.dtype)
        values = mx.concatenate([values, padding], axis=-1)
    n_ints = values.shape[-1] // vals_per_int
    values = mx.reshape(values, (*batch_shape, n_ints, vals_per_int)).astype(mx.uint32)
    shifts = mx.array([i * bits for i in range(vals_per_int)], dtype=mx.uint32)
    shifted = mx.left_shift(values, shifts)
    packed = mx.sum(shifted, axis=-1).astype(mx.uint32)
    return packed


def unpack_bits(packed: mx.array, bits: int, count: int) -> mx.array:
    """Unpack uint32 array back to individual low-bit values."""
    vals_per_int = 32 // bits
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
    *batch_shape, m = packed.shape
    shifts = mx.array([i * bits for i in range(vals_per_int)], dtype=mx.uint32)
    expanded = mx.expand_dims(packed, axis=-1)
    shifted = mx.right_shift(expanded, shifts)
    values = mx.bitwise_and(shifted, mask).astype(mx.uint8)
    values = mx.reshape(values, (*batch_shape, m * vals_per_int))
    return values[..., :count]


def pack_sign_bits(signs: mx.array) -> mx.array:
    """Pack ±1 sign values into uint32 bit arrays."""
    bits = mx.where(signs > 0, mx.array(1, dtype=mx.uint8), mx.array(0, dtype=mx.uint8))
    return pack_bits(bits, 1)


def unpack_sign_bits(packed: mx.array, count: int) -> mx.array:
    """Unpack uint32 bit array back to ±1 sign values."""
    bits = unpack_bits(packed, 1, count)
    return mx.where(bits > 0, mx.array(1, dtype=mx.int8), mx.array(-1, dtype=mx.int8))


# ---------------------------------------------------------------------------
# Matrices: orthogonal rotation (Stage 1) and Gaussian projection (Stage 2)
# ---------------------------------------------------------------------------

def _make_orthogonal_matrix(dim: int, seed: int) -> mx.array:
    """Uniform random orthogonal matrix via QR decomposition of Gaussian."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, R = np.linalg.qr(G)
    Q *= np.sign(np.diag(R))  # Haar measure fix
    return mx.array(Q, dtype=mx.float16)


def _make_jl_matrix(dim: int, seed: int) -> mx.array:
    """Random Gaussian matrix for QJL (does NOT need to be orthogonal)."""
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((dim, dim)).astype(np.float32)
    return mx.array(S, dtype=mx.float16)


# ---------------------------------------------------------------------------
# Lloyd-Max codebook for Beta distribution
# ---------------------------------------------------------------------------

_codebook_cache: dict[tuple[int, int], mx.array] = {}


def get_codebook(dim_head: int, bits: int) -> mx.array:
    """Get or compute Lloyd-Max codebook for Beta((d-1)/2, (d-1)/2) on [-1,1]."""
    key = (dim_head, bits)
    if key not in _codebook_cache:
        _codebook_cache[key] = _lloyd_max_beta(dim_head, bits)
    return _codebook_cache[key]


def _lloyd_max_beta(dim_head: int, bits: int) -> mx.array:
    """Compute Lloyd-Max optimal centroids for per-head Beta distribution."""
    from scipy.stats import beta as beta_dist
    from scipy.integrate import quad

    n_levels = 1 << bits
    alpha = (dim_head - 1) / 2.0
    dist = beta_dist(alpha, alpha, loc=-1, scale=2)

    probs = np.linspace(0, 1, n_levels + 1)
    centroids = np.array([
        dist.ppf((probs[i] + probs[i + 1]) / 2) for i in range(n_levels)
    ])

    for _ in range(300):
        old = centroids.copy()
        boundaries = np.concatenate([[-1.0], (centroids[:-1] + centroids[1:]) / 2, [1.0]])
        new_c = []
        for j in range(n_levels):
            a, b = boundaries[j], boundaries[j + 1]
            num, _ = quad(lambda x: x * dist.pdf(x), a, b)
            den, _ = quad(lambda x: dist.pdf(x), a, b)
            new_c.append(num / den if den > 1e-15 else (a + b) / 2)
        centroids = np.array(new_c)
        if np.max(np.abs(centroids - old)) < 1e-8:
            break

    return mx.array(centroids, dtype=mx.float32)


# ---------------------------------------------------------------------------
# Metal kernel for fused unpack (indices + signs → k_combined)
# ---------------------------------------------------------------------------

_metal_kernel_cache: dict[str, object] = {}


def _build_unpack_combined_kernel(bits: int, dim_head: int) -> object:
    """Metal kernel: unpack indices + signs into k_combined [B,H,L,2d].

    First d elements  = codebook[indices] * key_norm   (Stage 1)
    Last  d elements  = ±1 * gamma * correction_scale  (Stage 2)

    Allows both stages in one wide GEMM:
      scores = q_combined @ k_combined^T
    where q_combined = [q@Pi | q@S].
    """
    vals_per_int = 32 // bits
    mask = (1 << bits) - 1
    packed_per_d = -(-dim_head // vals_per_int)
    sign_packed_per_d = -(-dim_head // 32)
    correction_scale = math.sqrt(PI / 2.0) / dim_head
    d_aligned = (dim_head % 4 == 0)
    vals_aligned = (vals_per_int % 4 == 0)

    header = f"""\
#include <metal_stdlib>
using namespace metal;
constant int D = {dim_head};
constant int D2 = {2 * dim_head};
constant int BITS = {bits};
constant int VALS_PER_INT = {vals_per_int};
constant uint MASK = {mask}u;
constant int PACKED_PER_D = {packed_per_d};
constant int SIGN_PACKED_PER_D = {sign_packed_per_d};
constant float CORRECTION_SCALE = {correction_scale}f;
"""

    source = """\
    uint tid = thread_position_in_grid.x;
    uint B_val = packed_indices_shape[0];
    uint H = packed_indices_shape[1];
    uint L = packed_indices_shape[2];
    uint total = B_val * H * L;
    if (tid >= total) return;

    float knorm = (float)key_norms[tid];
    float gam = (float)gamma[tid] * CORRECTION_SCALE;
    uint out_base = tid * D2;

    // Part 1: unpack indices → codebook values * key_norm
    uint idx_base = tid * PACKED_PER_D;
    for (int p = 0; p < PACKED_PER_D; p++) {
        uint packed_val = packed_indices[idx_base + p];
"""
    if vals_aligned and d_aligned:
        source += """\
        for (int g = 0; g < VALS_PER_INT; g += 4) {
            int base = p * VALS_PER_INT + g;
            if (base + 3 >= D) {
                for (int i = g; i < VALS_PER_INT; i++) {
                    int dim_idx = p * VALS_PER_INT + i;
                    if (dim_idx >= D) break;
                    uint code = (packed_val >> (i * BITS)) & MASK;
                    k_combined[out_base + dim_idx] = codebook[code] * knorm;
                }
                break;
            }
            float4 v;
            v.x = codebook[(packed_val >> ((g  ) * BITS)) & MASK] * knorm;
            v.y = codebook[(packed_val >> ((g+1) * BITS)) & MASK] * knorm;
            v.z = codebook[(packed_val >> ((g+2) * BITS)) & MASK] * knorm;
            v.w = codebook[(packed_val >> ((g+3) * BITS)) & MASK] * knorm;
            *reinterpret_cast<device float4*>(k_combined + out_base + base) = v;
        }
"""
    else:
        source += """\
        for (int i = 0; i < VALS_PER_INT; i++) {
            int dim_idx = p * VALS_PER_INT + i;
            if (dim_idx >= D) break;
            uint code = (packed_val >> (i * BITS)) & MASK;
            k_combined[out_base + dim_idx] = codebook[code] * knorm;
        }
"""
    source += """\
    }

    // Part 2: unpack sign bits → ±gamma * correction_scale
    uint sign_base = tid * SIGN_PACKED_PER_D;
    uint sign_out_base = out_base + D;
    for (int p = 0; p < SIGN_PACKED_PER_D; p++) {
        uint packed_s = packed_signs[sign_base + p];
"""
    if d_aligned:
        source += """\
        for (int g = 0; g < 32; g += 4) {
            int base = p * 32 + g;
            if (base + 3 >= D) {
                for (int i = g; i < 32; i++) {
                    int dim_idx = p * 32 + i;
                    if (dim_idx >= D) break;
                    k_combined[sign_out_base + dim_idx] =
                        ((packed_s >> i) & 1u) ? gam : -gam;
                }
                break;
            }
            float4 s;
            s.x = ((packed_s >> (g  )) & 1u) ? gam : -gam;
            s.y = ((packed_s >> (g+1)) & 1u) ? gam : -gam;
            s.z = ((packed_s >> (g+2)) & 1u) ? gam : -gam;
            s.w = ((packed_s >> (g+3)) & 1u) ? gam : -gam;
            *reinterpret_cast<device float4*>(k_combined + sign_out_base + base) = s;
        }
"""
    else:
        source += """\
        for (int i = 0; i < 32; i++) {
            int dim_idx = p * 32 + i;
            if (dim_idx >= D) break;
            k_combined[sign_out_base + dim_idx] =
                ((packed_s >> i) & 1u) ? gam : -gam;
        }
"""
    source += """\
    }
"""

    return mx.fast.metal_kernel(
        name=f"unpack_combined_{bits}b_d{dim_head}",
        input_names=["packed_indices", "codebook", "packed_signs",
                     "gamma", "key_norms"],
        output_names=["k_combined"],
        source=source,
        header=header,
    )


def _get_unpack_kernel(bits: int, dim_head: int):
    key = f"combined_{bits}_{dim_head}"
    if key not in _metal_kernel_cache:
        _metal_kernel_cache[key] = _build_unpack_combined_kernel(bits, dim_head)
    return _metal_kernel_cache[key]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression in LLM inference."""
    enabled: bool = False
    stage1_bits: int = 3            # Lloyd-Max bits; total = stage1_bits + 1 (QJL)
    rotation_seed: int = 42
    jl_seed: int = 137
    min_seq_len: int = 64           # Don't compress until cache exceeds this length
    use_metal: bool = True          # Use Metal-accelerated unpack kernel

    @property
    def total_bits(self) -> int:
        return self.stage1_bits + 1

    @classmethod
    def from_env(cls) -> "TurboQuantConfig":
        """Build config from environment variables."""
        import os
        enabled = os.getenv("TURBOQUANT_ENABLED", "").strip().lower() in {"1", "true", "yes"}
        bits = int(os.getenv("TURBOQUANT_BITS", "3"))
        min_seq = int(os.getenv("TURBOQUANT_MIN_SEQ", "64"))
        use_metal = os.getenv("TURBOQUANT_USE_METAL", "1").strip().lower() not in {"0", "false"}
        return cls(enabled=enabled, stage1_bits=bits, min_seq_len=min_seq, use_metal=use_metal)


# ---------------------------------------------------------------------------
# Compressed key storage (per-head, growing incrementally)
# ---------------------------------------------------------------------------

@dataclass
class CompressedKeyCache:
    """Incrementally-built TurboQuant compressed key cache for LLM generation.

    During autoregressive decoding, new keys are compressed one token at a time
    and appended to the packed arrays.  Shared matrices (Pi, S, codebook) are
    computed once at first use and reused for all subsequent tokens.
    """
    # Per-token compressed data (grow with each new token)
    packed_indices_list: list = field(default_factory=list)  # each [H, 1, packed_d]
    packed_signs_list: list = field(default_factory=list)    # each [H, 1, sign_packed_d]
    residual_norms_list: list = field(default_factory=list)  # each [H, 1]
    key_norms_list: list = field(default_factory=list)       # each [H, 1]

    # Shared per-head matrices (set on first compress)
    Pi: Optional[mx.array] = None           # [d, d] orthogonal rotation
    S: Optional[mx.array] = None            # [d, d] JL matrix
    codebook: Optional[mx.array] = None     # [2^bits] centroids
    PiS_scaled: Optional[mx.array] = None   # [d, 2d] * scale, cached

    # Metadata
    num_heads: int = 0
    dim_head: int = 0
    bits: int = 3
    seq_len: int = 0

    def append_token(
        self,
        keys: mx.array,
        config: TurboQuantConfig,
    ) -> None:
        """Compress and append one token's keys to the cache.

        Args:
            keys: [1, num_heads, 1, dim_head] — a single token's projected keys,
                  already in per-head layout from the model's attention module.
        """
        # keys shape: [B=1, H, 1, d]
        H = keys.shape[1]
        d = keys.shape[3]

        if self.Pi is None:
            # First token — initialize shared matrices
            self.num_heads = H
            self.dim_head = d
            self.bits = config.stage1_bits
            self.Pi = _make_orthogonal_matrix(d, config.rotation_seed)
            self.S = _make_jl_matrix(d, config.jl_seed)
            self.codebook = get_codebook(d, config.stage1_bits)

        k = keys.squeeze(0).astype(mx.float32)  # [H, 1, d]

        # Normalize to unit sphere
        k_norms = mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-12)  # [H, 1, 1]
        k_unit = k / k_norms

        # Stage 1: rotate + quantize
        Pi_f32 = self.Pi.astype(mx.float32)
        k_rotated = k_unit @ Pi_f32  # [H, 1, d]

        dists = mx.abs(
            mx.expand_dims(k_rotated, axis=-1)
            - mx.reshape(self.codebook, (1, 1, 1, -1))
        )
        indices = mx.argmin(dists, axis=-1).astype(mx.uint8)  # [H, 1, d]

        # Stage 1 reconstruction for residual
        k_rotated_q = mx.take(self.codebook, indices, axis=0)
        k_unit_recon = k_rotated_q @ Pi_f32.T
        k_recon = k_unit_recon * k_norms

        # Stage 2: QJL on residual
        S_f32 = self.S.astype(mx.float32)
        residual = k - k_recon  # [H, 1, d]
        r_norms = mx.sqrt(mx.sum(residual * residual, axis=-1) + 1e-12)  # [H, 1]
        projected = residual @ S_f32  # [H, 1, d]
        qjl_signs = mx.sign(projected).astype(mx.int8)

        # Bit-pack
        flat_idx = mx.reshape(indices, (-1, d))    # [H, d]
        packed_idx = pack_bits(flat_idx, config.stage1_bits)
        packed_idx = mx.reshape(packed_idx, (H, 1, -1))

        flat_signs = mx.reshape(qjl_signs, (-1, d))  # [H, d]
        packed_signs = pack_sign_bits(flat_signs)
        packed_signs = mx.reshape(packed_signs, (H, 1, -1))

        self.packed_indices_list.append(packed_idx)
        self.packed_signs_list.append(packed_signs)
        self.residual_norms_list.append(r_norms.astype(mx.float16))
        self.key_norms_list.append(mx.squeeze(k_norms, axis=-1).astype(mx.float16))
        self.seq_len += 1

    def compute_scores(
        self,
        q_heads: mx.array,
        config: TurboQuantConfig,
    ) -> mx.array:
        """Compute attention scores against the compressed key cache.

        Args:
            q_heads: [B=1, H, q_seq, d] queries in per-head layout.

        Returns:
            [B=1, H, q_seq, cache_seq_len] attention scores.
        """
        if self.seq_len == 0:
            B, H, q_seq, d = q_heads.shape
            return mx.zeros((B, H, q_seq, 0), dtype=mx.float32)

        d = self.dim_head
        scale = 1.0 / math.sqrt(d)

        # Concatenate all cached tokens: [H, seq_len, packed_d]
        packed_indices = mx.concatenate(self.packed_indices_list, axis=1)
        packed_signs = mx.concatenate(self.packed_signs_list, axis=1)
        residual_norms = mx.concatenate(self.residual_norms_list, axis=-1)  # [H, seq_len]
        key_norms = mx.concatenate(self.key_norms_list, axis=-1)  # [H, seq_len]

        # Add batch dim: [1, H, L, ...]
        packed_indices = mx.expand_dims(packed_indices, axis=0)
        packed_signs = mx.expand_dims(packed_signs, axis=0)
        residual_norms = mx.expand_dims(residual_norms, axis=0)
        key_norms = mx.expand_dims(key_norms, axis=0)

        B, H, q_seq, _ = q_heads.shape
        L = self.seq_len

        if config.use_metal and L >= 4:
            return self._scores_metal(
                q_heads, packed_indices, packed_signs,
                residual_norms, key_norms, scale,
            )
        else:
            return self._scores_python(
                q_heads, packed_indices, packed_signs,
                residual_norms, key_norms, scale,
            )

    def _scores_metal(
        self,
        q_heads: mx.array,
        packed_indices: mx.array,
        packed_signs: mx.array,
        residual_norms: mx.array,
        key_norms: mx.array,
        scale: float,
    ) -> mx.array:
        """Metal-accelerated three-op pipeline."""
        B, H, q_seq, d = q_heads.shape
        L = self.seq_len

        # Cache PiS_scaled
        if self.PiS_scaled is None:
            Pi = self.Pi.astype(mx.float32)
            S = self.S.astype(mx.float32)
            self.PiS_scaled = mx.concatenate([Pi, S], axis=1) * scale

        q_combined = q_heads @ self.PiS_scaled  # [B, H, q_seq, 2d]

        # Metal unpack
        kernel = _get_unpack_kernel(self.bits, d)
        total_vecs = B * H * L
        tg = min(256, total_vecs)
        grid = ((total_vecs + tg - 1) // tg) * tg

        k_combined = kernel(
            inputs=[packed_indices, self.codebook, packed_signs,
                    residual_norms, key_norms],
            grid=(grid, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(B, H, L, 2 * d)],
            output_dtypes=[mx.float32],
        )[0]

        # Single wide GEMM (scale already folded in)
        return q_combined @ mx.swapaxes(k_combined, -1, -2)

    def _scores_python(
        self,
        q_heads: mx.array,
        packed_indices: mx.array,
        packed_signs: mx.array,
        residual_norms: mx.array,
        key_norms: mx.array,
        scale: float,
    ) -> mx.array:
        """Pure-MLX fallback for small caches or when Metal is disabled."""
        B, H, q_seq, d = q_heads.shape
        L = self.seq_len
        Pi = self.Pi.astype(mx.float32)
        S = self.S.astype(mx.float32)

        # Stage 1: reconstruct quantized keys
        flat_packed = mx.reshape(packed_indices, (-1, packed_indices.shape[-1]))
        indices = unpack_bits(flat_packed, self.bits, d)
        indices = mx.reshape(indices, (B, H, L, d))
        k_rotated_q = mx.take(self.codebook, indices, axis=0)

        k_unit_recon = k_rotated_q @ Pi.T
        k_norms_exp = mx.expand_dims(key_norms.astype(mx.float32), axis=-1)
        k_mse = k_unit_recon * k_norms_exp

        score_stage1 = q_heads @ mx.swapaxes(k_mse, -1, -2)

        # Stage 2: QJL correction
        flat_signs_packed = mx.reshape(packed_signs, (-1, packed_signs.shape[-1]))
        signs = mx.reshape(
            unpack_sign_bits(flat_signs_packed, d).astype(mx.float32), (B, H, L, d)
        )
        gamma = residual_norms.astype(mx.float32)

        Sq = q_heads @ S
        qjl_dot = Sq @ mx.swapaxes(signs, -1, -2)
        correction_scale = math.sqrt(PI / 2.0) / d
        score_stage2 = correction_scale * qjl_dot * mx.expand_dims(gamma, axis=2)

        return (score_stage1 + score_stage2) * scale

    @property
    def memory_bytes(self) -> int:
        """Approximate compressed cache memory usage."""
        if self.seq_len == 0:
            return 0
        total = 0
        for t in self.packed_indices_list:
            total += t.nbytes
        for t in self.packed_signs_list:
            total += t.nbytes
        for t in self.residual_norms_list:
            total += t.nbytes
        for t in self.key_norms_list:
            total += t.nbytes
        return total

    @property
    def dense_equivalent_bytes(self) -> int:
        """What the same cache would cost in dense float16."""
        return self.seq_len * self.num_heads * self.dim_head * 2  # float16

    @property
    def compression_ratio(self) -> float:
        if self.memory_bytes == 0:
            return 0.0
        return self.dense_equivalent_bytes / self.memory_bytes


# ---------------------------------------------------------------------------
# TurboQuantKVCache — drop-in for mlx_lm's KVCache
# ---------------------------------------------------------------------------

class TurboQuantKVCache:
    """KV cache with TurboQuant-compressed keys and dense values.

    Drop-in replacement for mlx_lm's KVCache used in model attention layers.
    Implements the same interface: update_and_fetch, offset, state, trim,
    is_trimmable, to_quantized, make_mask, nbytes, empty.

    Keys are stored compressed (TurboQuant bit-packed).  When update_and_fetch
    is called, keys are decompressed on the fly so the existing SDPA path works
    unchanged.  The memory saving comes from the compressed storage — at 4-bit
    (3+1) a 128K context uses ~3x less RAM for keys than dense float16.

    For even faster attention, models can check isinstance(cache, TurboQuantKVCache)
    and call cache.compressed_attention_scores(queries) to skip decompression
    entirely, using the fused asymmetric scoring formula.

    Usage:
        config = TurboQuantConfig(enabled=True, stage1_bits=3)
        # Build prompt_cache list with TurboQuantKVCache per layer:
        prompt_cache = [TurboQuantKVCache(config=config) for _ in range(n_layers)]
    """

    step = 256  # Match KVCache.step for compatibility

    def __init__(self, config: Optional[TurboQuantConfig] = None) -> None:
        self.config = config or TurboQuantConfig()
        self._compressed = CompressedKeyCache()
        self.keys = None    # Dense keys (before compression threshold)
        self.values = None   # Dense values (always)
        self.offset = 0
        self._using_compression = False

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Update cache with new keys/values and return full cache.

        Matches mlx_lm KVCache.update_and_fetch interface exactly.

        Args:
            keys: [B, n_kv_heads, new_tokens, head_dim]
            values: [B, n_kv_heads, new_tokens, head_dim]

        Returns:
            (all_keys, all_values) — keys are decompressed if compressed.
        """
        prev = self.offset
        new_tokens = keys.shape[2]

        # --- Values: always dense, same as stock KVCache ---
        if self.values is None or (prev + new_tokens) > self.values.shape[2]:
            B, n_kv_heads, _, v_head_dim = values.shape
            n_steps = (self.step + new_tokens - 1) // self.step
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.values is not None:
                if prev % self.step != 0:
                    self.values = self.values[..., :prev, :]
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.values = new_v

        self.values[..., prev : prev + new_tokens, :] = values

        # --- Keys: compress after threshold, dense before ---
        threshold = self.config.min_seq_len

        if not self._using_compression and (prev + new_tokens) < threshold:
            # Below threshold — store keys dense (same as stock KVCache)
            if self.keys is None or (prev + new_tokens) > self.keys.shape[2]:
                B, n_kv_heads, _, k_head_dim = keys.shape
                n_steps = (self.step + new_tokens - 1) // self.step
                k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
                new_k = mx.zeros(k_shape, keys.dtype)
                if self.keys is not None:
                    if prev % self.step != 0:
                        self.keys = self.keys[..., :prev, :]
                    self.keys = mx.concatenate([self.keys, new_k], axis=2)
                else:
                    self.keys = new_k
            self.keys[..., prev : prev + new_tokens, :] = keys
            self.offset += new_tokens
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

        if not self._using_compression:
            # Crossing threshold — compress all existing + new keys
            self._using_compression = True
            if self.keys is not None:
                existing = self.keys[..., :prev, :]
                all_keys = mx.concatenate([existing, keys], axis=2)
            else:
                all_keys = keys
            for t in range(all_keys.shape[2]):
                token_key = all_keys[:, :, t : t + 1, :]
                self._compressed.append_token(token_key, self.config)
            self.keys = None  # Free dense storage
        else:
            # Already compressing — append incrementally
            for t in range(new_tokens):
                token_key = keys[:, :, t : t + 1, :]
                self._compressed.append_token(token_key, self.config)

        self.offset += new_tokens

        # Decompress for the standard SDPA path
        decompressed = self._decompress_keys()
        return decompressed, self.values[..., : self.offset, :]

    def _decompress_keys(self) -> mx.array:
        """Decompress all keys from compressed storage.

        Returns: [B=1, H, seq_len, d] decompressed keys.
        """
        ck = self._compressed
        if ck.seq_len == 0:
            return mx.zeros((1, ck.num_heads, 0, ck.dim_head), dtype=mx.float16)

        d = ck.dim_head
        H = ck.num_heads
        L = ck.seq_len

        packed_indices = mx.concatenate(ck.packed_indices_list, axis=1)  # [H, L, packed_d]
        packed_signs = mx.concatenate(ck.packed_signs_list, axis=1)
        residual_norms = mx.concatenate(ck.residual_norms_list, axis=-1)  # [H, L]
        key_norms = mx.concatenate(ck.key_norms_list, axis=-1)

        # Add batch dim
        packed_indices = mx.expand_dims(packed_indices, axis=0)  # [1, H, L, packed_d]
        packed_signs = mx.expand_dims(packed_signs, axis=0)
        residual_norms = mx.expand_dims(residual_norms, axis=0)
        key_norms = mx.expand_dims(key_norms, axis=0)

        Pi = ck.Pi.astype(mx.float32)
        S = ck.S.astype(mx.float32)

        # Stage 1: reconstruct quantized keys
        flat_packed = mx.reshape(packed_indices, (-1, packed_indices.shape[-1]))
        indices = unpack_bits(flat_packed, ck.bits, d)
        indices = mx.reshape(indices, (1, H, L, d))
        k_rotated_q = mx.take(ck.codebook, indices, axis=0)
        k_unit_recon = k_rotated_q @ Pi.T
        k_norms_exp = mx.expand_dims(key_norms.astype(mx.float32), axis=-1)
        k_mse = k_unit_recon * k_norms_exp

        # Stage 2: QJL correction
        flat_signs_packed = mx.reshape(packed_signs, (-1, packed_signs.shape[-1]))
        signs = mx.reshape(
            unpack_sign_bits(flat_signs_packed, d).astype(mx.float32), (1, H, L, d)
        )
        gamma = mx.expand_dims(residual_norms.astype(mx.float32), axis=-1)
        correction = (math.sqrt(PI / 2.0) / d) * gamma * (signs @ S.T)

        k_reconstructed = k_mse + correction
        return k_reconstructed.astype(mx.float16)

    def compressed_attention_scores(self, queries: mx.array) -> mx.array:
        """Compute attention scores directly from compressed keys (no decompression).

        This is the fast path — use when the model supports it.

        Args:
            queries: [B=1, H, q_seq, d] in per-head layout.

        Returns:
            [B=1, H, q_seq, cache_seq_len] attention scores (unscaled by 1/sqrt(d),
            matching what the model expects before softmax).
        """
        if not self._using_compression:
            # Dense path
            if self.keys is not None:
                k = self.keys[..., : self.offset, :]
                scale = 1.0 / math.sqrt(k.shape[-1])
                return (queries.astype(mx.float32) @ mx.swapaxes(
                    k.astype(mx.float32), -1, -2
                )) * scale
            B, H, q_seq, d = queries.shape
            return mx.zeros((B, H, q_seq, 0), dtype=mx.float32)

        return self._compressed.compute_scores(
            queries.astype(mx.float32), self.config
        )

    # ------------------------------------------------------------------
    # mlx_lm KVCache interface compatibility
    # ------------------------------------------------------------------

    def size(self):
        return self.offset

    @property
    def state(self):
        if self._using_compression:
            return self._decompress_keys(), self.values[..., : self.offset, :]
        if self.keys is not None:
            if self.offset == self.keys.shape[2]:
                return self.keys, self.values
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )
        return None, None

    @state.setter
    def state(self, v):
        # For loading from a saved state — always dense
        self.keys, self.values = v
        self.offset = self.keys.shape[2] if self.keys is not None else 0
        self._using_compression = False

    def is_trimmable(self):
        return not self._using_compression

    def trim(self, n):
        if self._using_compression:
            return 0  # Can't trim compressed cache
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        # Already quantized via TurboQuant — return self
        return self

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.llama import create_attention_mask
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        total = 0
        if self._using_compression:
            total += self._compressed.memory_bytes
        elif self.keys is not None:
            total += self.keys.nbytes
        if self.values is not None:
            total += self.values.nbytes
        return total

    def stats(self) -> dict:
        """Return compression statistics."""
        if not self._using_compression:
            return {
                "mode": "dense",
                "seq_len": self.offset,
                "compressed": False,
                "nbytes": self.nbytes,
            }
        ck = self._compressed
        return {
            "mode": "turboquant",
            "seq_len": ck.seq_len,
            "bits": f"{ck.bits}+1",
            "key_compressed_bytes": ck.memory_bytes,
            "key_dense_equiv_bytes": ck.dense_equivalent_bytes,
            "key_compression_ratio": f"{ck.compression_ratio:.1f}x",
            "total_nbytes": self.nbytes,
            "num_heads": ck.num_heads,
            "dim_head": ck.dim_head,
        }
