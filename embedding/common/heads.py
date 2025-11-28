from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn

try:
    from safetensors.torch import load_file, save_file
except ImportError:  # pragma: no cover - safetensors is expected in runtime env
    load_file = None  # type: ignore
    save_file = None  # type: ignore


class EmbeddingGemmaProjectionHead(nn.Module):
    """Mean-pooled → 3072 → 768 projection head used by EmbeddingGemma."""

    def __init__(
        self,
        base_path: Optional[str] = None,
        input_dim: int = 768,
        hidden_dim: int = 3072,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, input_dim, bias=False)
        if base_path:
            self.load_weights(base_path)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        hidden = self.linear1(embeddings)
        return self.linear2(hidden)

    def load_weights(self, base_path: str) -> None:
        """Load projection weights from a SentenceTransformer-style checkpoint."""
        if load_file is None:
            return
        base_dir = Path(base_path)
        if not base_dir.exists():
            raise ValueError(f"Weights not found at {base_path}")

        def _resolve_path(subdir: str) -> Optional[str]:
            local_file = base_dir / subdir / "model.safetensors"
            if local_file.exists():
                print(f"Loading weights from {local_file}")
                return str(local_file)
            raise ValueError(f"Weights not found at {local_file}")

        dense1_path = _resolve_path("2_Dense")
        if dense1_path:
            state = load_file(dense1_path)
            weight = state.get("linear.weight")
            if weight is not None and weight.shape == self.linear1.weight.shape:
                self.linear1.weight.data.copy_(weight)

        dense2_path = _resolve_path("3_Dense")
        if dense2_path:
            state = load_file(dense2_path)
            weight = state.get("linear.weight")
            if weight is not None and weight.shape == self.linear2.weight.shape:
                self.linear2.weight.data.copy_(weight)


def save_embeddinggemma_modules(
    head: nn.Module,
    output_dir: str | Path,
    input_dim: int = 768,
    hidden_dim: int = 3072,
) -> None:
    """Persist pooling/dense configs plus weights so evaluation can reload them."""
    if save_file is None:
        return
    base_head = head
    if isinstance(head, nn.DataParallel):
        base_head = head.module  # type: ignore[assignment]
    if not isinstance(base_head, EmbeddingGemmaProjectionHead):
        return

    output_path = Path(output_dir)
    pooling_dir = output_path / "1_Pooling"
    dense_1_dir = output_path / "2_Dense"
    dense_2_dir = output_path / "3_Dense"

    pooling_dir.mkdir(parents=True, exist_ok=True)
    dense_1_dir.mkdir(parents=True, exist_ok=True)
    dense_2_dir.mkdir(parents=True, exist_ok=True)

    pooling_cfg = {
        "word_embedding_dimension": input_dim,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
    }
    (pooling_dir / "config.json").write_text(json.dumps(pooling_cfg, indent=4))

    dense_1_cfg = {
        "in_features": input_dim,
        "out_features": hidden_dim,
        "bias": False,
        "activation_function": "torch.nn.modules.linear.Identity",
    }
    dense_2_cfg = {
        "in_features": hidden_dim,
        "out_features": input_dim,
        "bias": False,
        "activation_function": "torch.nn.modules.linear.Identity",
    }
    (dense_1_dir / "config.json").write_text(json.dumps(dense_1_cfg, indent=4))
    (dense_2_dir / "config.json").write_text(json.dumps(dense_2_cfg, indent=4))

    save_file(
        {"linear.weight": base_head.linear1.weight.detach().cpu()},
        str(dense_1_dir / "model.safetensors"),
    )
    save_file(
        {"linear.weight": base_head.linear2.weight.detach().cpu()},
        str(dense_2_dir / "model.safetensors"),
    )
