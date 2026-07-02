"""AirLLM CPU patch — override utils.uncompress/compress_layer_state_dict.

AirLLM v2.11's `utils.py` hardcodes CUDA device for both directions of NF4
and int8 quantization: `device="cuda"` at line 92, `.cuda()` calls at 94,
105, 106, 107, 162, 169. On CPU-only torch these fail with
"Torch not compiled with CUDA enabled" (dequantize path) or
AssertionError (Linear4bit forward path).

`bitsandbytes` 0.49+ has a CPU backend for NF4/int8 that DOES work — verified
via `bnb.functional.quantize_nf4` and `bnb.nn.Linear4bit(...).to("cpu")`
round-trips. So the only real blocker is AirLLM's hardcoded device pin.

This module monkey-patches BOTH functions with device-detecting variants,
called once at package import time from `on_prem_llm_lab/__init__.py`. The
patch is a no-op on CUDA-capable machines (device auto-detects to "cuda").

Documented in the report methodology section — see README §Methodology for
the full disclosure. Upstream AirLLM issue: the compression code path was
authored on the assumption of a CUDA host; bitsandbytes CPU backend
predates that decision.
"""

from __future__ import annotations


def _cpu_uncompress_layer_state_dict(layer_state_dict):  # type: ignore[no-untyped-def]
    """CPU-aware replacement for airllm.utils.uncompress_layer_state_dict.

    Behaviour identical to the upstream function except the hardcoded
    ``device="cuda"`` and ``.cuda()`` calls (utils.py lines 92, 94, 105-107)
    are replaced with an auto-detected device string.
    """
    import bitsandbytes as bnb  # type: ignore[import-untyped]  # noqa: PLC0415
    import torch  # noqa: PLC0415

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    uncompressed = None
    if any("4bit" in k for k in layer_state_dict):
        uncompressed = {}
        for k, v in layer_state_dict.items():
            if "4bit" in k:
                continue
            qs_dict = {
                kk[len(k):]: kv
                for kk, kv in layer_state_dict.items()
                if kk.startswith(k) and k != kk
            }
            quant_state = bnb.functional.QuantState.from_dict(
                qs_dict=qs_dict, device=dev,
            )
            uncompressed[k] = bnb.functional.dequantize_nf4(v.to(dev), quant_state)
        del layer_state_dict
    elif any("8bit" in k for k in layer_state_dict):
        uncompressed = {}
        for k, v in layer_state_dict.items():
            if "8bit" in k:
                continue
            absmax = layer_state_dict[k + ".8bit.absmax"]
            code = layer_state_dict[k + ".8bit.code"]
            uncompressed[k] = bnb.functional.dequantize_blockwise(
                v.to(dev),
                bnb.functional.QuantState(
                    absmax=absmax.to(dev), code=code.to(dev),
                    blocksize=2048, dtype=torch.float16,
                ),
            )
        del layer_state_dict
    return layer_state_dict if uncompressed is None else uncompressed


def _cpu_compress_layer_state_dict(layer_state_dict, compression=None):  # type: ignore[no-untyped-def]
    """CPU-aware replacement for airllm.utils.compress_layer_state_dict.

    Called during initial layer-split when compression={'4bit', '8bit'}.
    Replaces `.cuda()` calls at utils.py lines 162, 169.
    """
    import bitsandbytes as bnb  # type: ignore[import-untyped]  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from airllm.utils import (
        save_quant_state_to_dict,  # type: ignore[import-untyped]  # noqa: PLC0415
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = None
    if compression == "4bit":
        out = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(v.to(dev), blocksize=64)
            out[k] = v_quant
            for qk, qv in save_quant_state_to_dict(quant_state).items():
                out[k + ".4bit." + qk] = qv
    elif compression == "8bit":
        out = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_blockwise(
                v.to(dev), blocksize=2048,
            )
            out[k] = v_quant
            out[k + ".8bit.absmax"] = quant_state.absmax.clone().contiguous()
            out[k + ".8bit.code"] = quant_state.code.clone().contiguous()
    return out if out is not None else layer_state_dict


def apply_cpu_patch() -> None:
    """Install the CPU-aware overrides into airllm.utils. Idempotent."""
    import airllm.utils as _u  # type: ignore[import-untyped]  # noqa: PLC0415

    _u.uncompress_layer_state_dict = _cpu_uncompress_layer_state_dict
    _u.compress_layer_state_dict = _cpu_compress_layer_state_dict


__all__ = ["apply_cpu_patch"]
