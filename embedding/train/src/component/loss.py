from typing import Optional, Sequence

import torch
from torch import nn


def contrastive_with_negative_loss(
    text: torch.Tensor, pos: torch.Tensor, neg: Optional[torch.Tensor] = None, tau: float = 20.0
) -> torch.Tensor:
    """
    Compute contrastive with negative loss

    :param text: torch.Tensor, text.
    :param pos: torch.Tensor, positive samples of text.
    :param neg: torch.Tensor, negative samples of text.
    :param tau: float, scale factor, default 20.0

    :return: torch.Tensor, loss value
    """
    # k = num of negative samples
    target = torch.cat((pos, neg), dim=0) if neg is not None else pos  # ((k+1)*B, D)
    q_norm = torch.nn.functional.normalize(text, p=2, dim=1)  # (B, D)
    t_norm = torch.nn.functional.normalize(target, p=2, dim=1)  # ((k+1)*B, D)
    scores = torch.mm(q_norm, t_norm.transpose(0, 1)) / tau  # (B, (k+1)*B)
    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
    return nn.CrossEntropyLoss()(scores, labels)


class E5EmbeddingLoss:
    """
    Configure E5EmbeddingLoss.

    only support info_nce loss for now.
    """

    def __init__(self, tau: float = 0.2, matryoshka_dims: Optional[Sequence[int]] = None):
        self.tau = tau
        if matryoshka_dims:
            filtered_dims = {int(dim) for dim in matryoshka_dims if int(dim) > 0}
            self.matryoshka_dims = sorted(filtered_dims, reverse=True)
        else:
            self.matryoshka_dims = None

    def _get_active_dims(self, embedding_dim: int) -> Optional[list[int]]:
        if not self.matryoshka_dims:
            return None
        dims = [dim for dim in self.matryoshka_dims if dim <= embedding_dim]
        return dims or None

    def __call__(self, outputs: torch.Tensor, num_sep_ids: int) -> torch.Tensor:
        # text,positive,negative

        text = outputs[::num_sep_ids]
        positive = outputs[1::num_sep_ids]
        negatives = [outputs[i::num_sep_ids] for i in range(2, num_sep_ids)]

        assert text.shape == positive.shape, (
            f"text.shape={text.shape}, postive.shape={positive.shape}, query and positive should have same shape"
        )

        assert all(text.shape == negative.shape for negative in negatives), (
            f"there is a shape mismatch between query and negative, or in between negatives, query.shape={text.shape}, negatives[0].shape={negatives[0].shape}"
        )

        negatives = torch.cat(negatives, dim=0)

        matryoshka_dims = self._get_active_dims(text.shape[-1])
        if not matryoshka_dims:
            return contrastive_with_negative_loss(text, positive, negatives, self.tau)

        losses = [
            contrastive_with_negative_loss(
                text[:, :dim],
                positive[:, :dim],
                negatives[:, :dim],
                self.tau,
            )
            for dim in matryoshka_dims
        ]
        return torch.stack(losses).mean()
