from __future__ import annotations

import threading
from typing import Callable

import torch.nn as nn
from transformers import modeling_utils

_PATCH_LOCK = threading.Lock()
_patch_applied = False


def _needs_guard(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    if weight is None:
        return False
    try:
        return weight.shape[0] == 0
    except Exception:  # pragma: no cover - defensive fallback
        return True


def apply_zero_size_embedding_guard() -> None:
    """
    Prevent HF weight initialization from touching ZeRO-3 partitioned embeddings whose storage size is zero.
    When DeepSpeed ZeRO-3 initializes a model under zero.Init, parameters are created with empty tensors on
    non-owner ranks. HF's default initializer attempts to zero-out the padding row in every embedding module,
    which raises IndexError for those zero-sized tensors. This guard skips the padding update until the
    parameter is materialized, relying on the checkpoint weights that will be loaded right after.
    """
    global _patch_applied
    if _patch_applied:
        return

    with _PATCH_LOCK:
        if _patch_applied:
            return

        original_init: Callable[[modeling_utils.PreTrainedModel, nn.Module], None] = (
            modeling_utils.PreTrainedModel._init_weights  # noqa
        )

        def _patched_init(
            self: modeling_utils.PreTrainedModel,
            module: nn.Module,  # type: ignore[name-defined]
        ) -> None:
            if isinstance(module, nn.Embedding) and _needs_guard(module):
                return
            original_init(self, module)

        modeling_utils.PreTrainedModel._init_weights = _patched_init  # noqa
        _patch_applied = True
