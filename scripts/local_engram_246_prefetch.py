"""Experimental side-stream adapter for PR #246's LocalEngramBuffer."""


def build_layered_fetch(buffer, fetch_layer=None):
    """Build a bound fetch before warmup and CUDA Graph capture.

    An optional fetch_layer(indices_for_layer, layer_index,
    use_tma_aligned_col_major_sf=True) runs on the current stream and returns
    the layer's data or (data, sf). Its implementation must be graph-capturable
    and must not reuse storage across simultaneous layer results. The default
    reproduces #246's shared scale backing, including strides and offsets.

    Fetch and all its hooks run on the same caller stream. Consume the hooks
    in model order, including the last hook, before ending capture. The last
    hook rejoins all prefetch work. Keep the adapter and its live symmetric
    mappings until outstanding forwards and graphs have finished.
    """
    import torch

    device = torch.device(buffer.device)
    side = None
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Build the Engram prefetch adapter before CUDA Graph capture")
            device = torch.device("cuda", torch.cuda.current_device())
            if buffer.num_layers > 1:
                side = torch.cuda.Stream(device=device)

    def fetch(indices, num_qps=0, use_tma_aligned_col_major_sf=True):
        del num_qps  # LocalEngram has no QPs.
        if indices.ndim != 3 or indices.dtype != torch.int32 or not indices.is_contiguous():
            raise ValueError("Expected contiguous int32 Engram indices [layers, tokens, entries]")
        if indices.device != device:
            raise ValueError(f"Engram indices are on {indices.device}, expected {device}")
        layers, tokens, entries = indices.shape
        if layers != buffer.num_layers or entries != buffer.num_entries_per_token or tokens > buffer.num_max_tokens:
            raise ValueError(f"Unexpected Engram indices shape: {tuple(indices.shape)}")

        scale_backing = None
        if fetch_layer is None and buffer.use_fp8:
            width = entries * buffer.num_sf_packs
            # #246 aligns the scale token dimension to four 4-byte packs.
            aligned_tokens = (tokens + 3) // 4 * 4
            if use_tma_aligned_col_major_sf:
                scale_strides, layer_stride = (1, aligned_tokens), width * aligned_tokens
            else:
                scale_strides, layer_stride = (width, 1), width * tokens
            scale_backing = torch.zeros((layers, layer_stride), dtype=buffer.sf_dtype, device=device)

        def read(layer):
            layer_indices = indices[layer]
            if fetch_layer is not None:
                return fetch_layer(layer_indices, layer, use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf)
            owner, local_row = buffer._route(layer_indices.reshape(tokens * entries), buffer.num_entries_per_layer[layer])
            rows = buffer._gather(buffer._peer_tables[layer], owner, local_row)
            data = rows.view(buffer.dtype).view(tokens, entries * buffer.hidden)
            if scale_backing is None:
                return data
            scales = buffer._gather(buffer._peer_sfs[layer], owner, local_row)
            scale = scale_backing.as_strided((tokens, width), scale_strides, layer * layer_stride)
            scale.copy_(scales.view(buffer.sf_dtype).view(tokens, width))
            return data, scale

        if side is None or tokens == 0:
            return [_ready_hook(read(layer)) for layer in range(layers)]

        caller = torch.cuda.current_stream(device)
        side.wait_stream(caller)  # Wait for the hashes and zeroed shared SF, before layer 0 work.
        try:
            indices.record_stream(side)
            if scale_backing is not None:
                scale_backing.record_stream(side)
            hooks = [_ready_hook(read(0))]
            with torch.cuda.stream(side):
                for layer in range(1, layers):
                    result = read(layer)
                    hooks.append(_deferred_hook(result, side.record_event(), caller))
        except BaseException as error:
            # If a callback aborts after forking, join the work already queued.
            try:
                caller.wait_stream(side)
            except RuntimeError as join_error:
                error.add_note(f"Engram prefetch stream could not be joined: {join_error}")
            raise
        return hooks

    return fetch


def _ready_hook(result):
    def consume():
        return result

    return consume


def _deferred_hook(result, ready, caller):
    def consume():
        import torch

        if torch.cuda.current_stream(caller.device) != caller:
            raise RuntimeError("Consume Engram prefetch on its caller stream")
        caller.wait_event(ready)
        for tensor in result if isinstance(result, tuple) else (result,):
            tensor.record_stream(caller)
        return result

    return consume
