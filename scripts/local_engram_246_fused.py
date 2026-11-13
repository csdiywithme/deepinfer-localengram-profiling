"""Experimental owner-only NVLink gather for the pinned #246 LocalEngramBuffer.

Build after the shard mapping and writes, and warm the required layouts before
CUDA graph capture. Keep the bound fetch alive until all its work and graphs
have retired. For a separate CUDA stream, the caller orders input readiness and
output consumption with events and keeps the tensors alive across both streams.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_tokens", "sf_column_stride"])
def _owner_gather(
    indices, addresses, data, scales, num_tokens, sf_column_stride,
    ENTRIES: tl.constexpr, ROWS_PER_RANK: tl.constexpr, RANKS: tl.constexpr,
    DATA_OFFSET_WORDS: tl.constexpr, DATA_WORDS: tl.constexpr,
    SF_OFFSET_WORDS: tl.constexpr, SF_PACKS: tl.constexpr, TMA_SF: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_WORDS: tl.constexpr, BLOCK_SF: tl.constexpr,
):
    entry = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    real = entry < num_tokens * ENTRIES
    index = tl.load(indices + entry, mask=real, other=-1)
    valid = real & (index >= 0) & (index.to(tl.int64) < ROWS_PER_RANK * RANKS)
    safe_index = tl.where(valid, index, 0)
    owner = safe_index // ROWS_PER_RANK
    local_row = (safe_index - owner * ROWS_PER_RANK).to(tl.int64)
    address = tl.load(addresses + owner, mask=valid, other=0)

    # Copy integer words, preserving every FP8 or BF16 bit pattern.
    words = tl.arange(0, BLOCK_WORDS)
    table = address.to(tl.pointer_type(tl.int64)) + DATA_OFFSET_WORDS + local_row * DATA_WORDS
    values = tl.load(table[:, None] + words[None, :], mask=valid[:, None] & (words[None, :] < DATA_WORDS), other=0)
    data_offset = entry.to(tl.int64)[:, None] * DATA_WORDS + words[None, :]
    tl.store(data + data_offset, values, mask=real[:, None] & (words[None, :] < DATA_WORDS))

    if SF_PACKS > 0:
        pack = tl.arange(0, BLOCK_SF)
        scale = address.to(tl.pointer_type(tl.int32)) + SF_OFFSET_WORDS + local_row * SF_PACKS
        packed = tl.load(scale[:, None] + pack[None, :], mask=valid[:, None] & (pack[None, :] < SF_PACKS), other=0)
        if TMA_SF:
            token = entry // ENTRIES
            col = (entry % ENTRIES)[:, None] * SF_PACKS + pack[None, :]
            sf_offset = col.to(tl.int64) * sf_column_stride + token[:, None]
            # Only the final real token writes padding, once for each scale column.
            for pad in tl.static_range(1, 4):
                padding = (token == num_tokens - 1) & (token + pad < sf_column_stride)
                tl.store(scales + sf_offset + pad, 0, mask=padding[:, None] & (pack[None, :] < SF_PACKS))
        else:
            sf_offset = entry.to(tl.int64)[:, None] * SF_PACKS + pack[None, :]
        tl.store(scales + sf_offset, packed, mask=real[:, None] & (pack[None, :] < SF_PACKS))


class _FusedFetcher:
    def __init__(self, buffer):
        # Retain the mapped memory without a reference cycle if installed as buffer.fetch.
        self._peers = tuple(buffer._peers)
        self._mapping = buffer._keepalive
        if len(self._peers) != buffer.num_ranks or not self._peers:
            raise ValueError("Expected a live #246 LocalEngramBuffer")
        self.device = self._peers[buffer.rank_idx].device
        if self.device.type != "cuda" or any(peer.device != self.device for peer in self._peers):
            raise ValueError("Expected CUDA peer views mapped in this rank's address space")
        self.layout = buffer.layout
        self.rows_per_rank = tuple(buffer.num_entries_per_layer)
        self.entries = buffer.num_entries_per_token
        self.max_tokens = buffer.num_max_tokens
        self.hidden = buffer.hidden
        self.dtype = buffer.dtype
        self.sf_dtype = buffer.sf_dtype
        self.sf_packs = buffer.num_sf_packs
        self.row_words = self.layout.row_bytes // 8
        if self.layout.row_bytes % 8 or self.layout.sf_row_bytes != self.sf_packs * 4:
            raise ValueError("Expected 8-byte data words and 4-byte scale packs")
        if self.sf_packs and self.sf_dtype.itemsize != 4:
            raise ValueError("Expected a 4-byte scale dtype")
        pointers = [peer.data_ptr() for peer in self._peers]
        if any(not address or address % 8 for address in pointers):
            raise ValueError("Expected non-null, aligned peer pointers")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Build the fused Engram adapter before graph capture")
            self.addresses = torch.tensor(pointers, dtype=torch.int64, device=self.device)

    def _check_indices(self, indices, expected_ndim):
        if (indices.ndim != expected_ndim or indices.dtype != torch.int32 or not indices.is_contiguous()
                or indices.device != self.device or indices.shape[-1] != self.entries):
            raise ValueError("Expected contiguous int32 indices on the mapped device with the configured entry width")
        if indices.shape[-2] > self.max_tokens:
            raise ValueError(f"{indices.shape[-2]} tokens exceeds {self.max_tokens}")

    def _scale_layout(self, num_tokens, tma):
        width = self.entries * self.sf_packs
        if tma:
            aligned = (num_tokens + 3) // 4 * 4
            return width, width * aligned, (1, aligned)
        return width, width * num_tokens, (width, 1)

    def _launch_layer(self, indices, layer_index, scales, scale_stride, tma):
        num_tokens = indices.shape[0]
        data = torch.empty((num_tokens, self.entries * self.row_words), dtype=torch.int64, device=self.device)
        if num_tokens:
            block_rows = 4
            _owner_gather[(triton.cdiv(num_tokens * self.entries, block_rows),)](
                indices, self.addresses, data, scales if scales is not None else data, num_tokens, scale_stride,
                self.entries, self.rows_per_rank[layer_index], len(self._peers),
                self.layout.table_offsets[layer_index] // 8, self.row_words,
                self.layout.sf_offsets[layer_index] // 4, self.sf_packs, tma,
                block_rows, triton.next_power_of_2(self.row_words), triton.next_power_of_2(max(1, self.sf_packs)),
                num_warps=4,
            )
        return data.view(self.dtype).view(num_tokens, self.entries * self.hidden)

    def fetch(self, indices, num_qps=0, use_tma_aligned_col_major_sf=True):
        del num_qps
        self._check_indices(indices, 3)
        num_layers, num_tokens, _ = indices.shape
        if num_layers != len(self.rows_per_rank):
            raise ValueError(f"Expected {len(self.rows_per_rank)} Engram layers, got {num_layers}")
        tma = use_tma_aligned_col_major_sf
        width, layer_size, strides = self._scale_layout(num_tokens, tma)
        scales = torch.empty((num_layers, layer_size), dtype=torch.int32, device=self.device) if self.sf_packs else None
        hooks = []
        for layer in range(num_layers):
            data = self._launch_layer(indices[layer], layer, None if scales is None else scales[layer], strides[1], tma)
            result = data
            if scales is not None:
                scale_view = scales.view(self.sf_dtype).as_strided((num_tokens, width), strides, layer * layer_size)
                result = (data, scale_view)
            hooks.append(lambda result=result: result)
        return hooks

    def fetch_layer(self, indices_for_layer, layer_index, use_tma_aligned_col_major_sf=True):
        self._check_indices(indices_for_layer, 2)
        if not 0 <= layer_index < len(self.rows_per_rank):
            raise ValueError(f"Invalid Engram layer index: {layer_index}")
        num_tokens = indices_for_layer.shape[0]
        tma = use_tma_aligned_col_major_sf
        width, layer_size, strides = self._scale_layout(num_tokens, tma)
        scales = torch.empty(layer_size, dtype=torch.int32, device=self.device) if self.sf_packs else None
        data = self._launch_layer(indices_for_layer, layer_index, scales, strides[1], tma)
        if scales is None:
            return data
        return data, scales.view(self.sf_dtype).as_strided((num_tokens, width), strides)


def build_fused_fetch(buffer):
    """Return the #246-compatible fetch, including shared scale backing across layers."""
    return _FusedFetcher(buffer).fetch


def build_fused_layer_fetch(buffer):
    """Return a per-layer fetch with standalone, layout-compatible scale backing."""
    return _FusedFetcher(buffer).fetch_layer
