"""Utility functions for embedding models."""

from typing import List, Mapping

import torch
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerFast


def move_to_cuda(sample):
    """
    Move tensors to CUDA device recursively.

    Args:
        sample: Input data (tensor, dict, list, tuple, or mapping)

    Returns:
        Data moved to CUDA device
    """
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        if isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        if isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        if isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        if isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})  # type: ignore
        return maybe_tensor

    return _move_to_cuda(sample)


def pool(last_hidden_states: Tensor, attention_mask: Tensor, pool_type: str) -> Tensor:
    """
    Apply pooling strategy to obtain sentence embeddings.

    Args:
        last_hidden_states: Model outputs [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len]
        pool_type: Pooling strategy ('cls', 'avg', 'last', 'weightedavg')

    Returns:
        Pooled embeddings [batch_size, hidden_size]
    """
    mask = attention_mask.bool()
    last_hidden = last_hidden_states.masked_fill(~mask[..., None], 0.0)

    if pool_type == "avg":
        # Mean pooling with safe denominator
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        emb = last_hidden.sum(dim=1) / denom.to(last_hidden.dtype)
    elif pool_type == "weightedavg":
        # Position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
        weight_mask = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(last_hidden * weight_mask.unsqueeze(-1).float(), dim=1)
        d = weight_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        emb = s / d
    elif pool_type == "cls":
        # CLS token (first token)
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        # Last token (considering padding)
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def create_batch_dict(
    tokenizer: PreTrainedTokenizerFast,
    input_texts: List[str],
    always_add_eos: bool,
    max_length: int,
) -> BatchEncoding:
    """
    Create batch dictionary with proper tokenization.

    Args:
        tokenizer: HuggingFace tokenizer
        input_texts: List of input texts
        always_add_eos: Whether to always add EOS token
        max_length: Maximum sequence length

    Returns:
        Tokenized batch dictionary
    """
    if not always_add_eos:
        return tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            pad_to_multiple_of=8,
            return_token_type_ids=False,
            truncation=True,
            return_tensors="pt",
        )

    batch_dict = tokenizer(
        input_texts,
        max_length=max_length - 1,
        return_token_type_ids=False,
        return_attention_mask=False,
        padding=False,
        truncation=True,
    )

    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [
        input_ids + [tokenizer.eos_token_id]
        for input_ids in batch_dict["input_ids"]  # type: ignore
    ]

    return tokenizer.pad(
        batch_dict,
        padding=True,
        pad_to_multiple_of=8,
        return_attention_mask=True,
        return_tensors="pt",
    )


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Args:
        a: First tensor [batch_size_a, hidden_size]
        b: Second tensor [batch_size_b, hidden_size]

    Returns:
        Cosine similarity matrix [batch_size_a, batch_size_b]
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
