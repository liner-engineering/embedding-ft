from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class E5EmbeddingDataCollator:
    """Batch collator for contrastive E5-style inputs.

    The tokenizer produces a single flattened sequence that contains multiple
    segments (query, positive, negatives, ...). Each segment boundary is
    marked by the `seperate_ids` array. This collator splits every example into
    those segments, optionally appends the EOS token, and pads the resulting
    pieces so that the Hugging Face Trainer can consume them directly.

    Args:
        tokenizer: Tokenizer used to convert IDs back to padded tensors.
        padding: Padding strategy forwarded to ``tokenizer.pad``.
        max_length: Optional maximum length applied during padding when EOS is not appended.
        return_tensors: Tensor type to return (e.g. ``"pt"``).

    Returns:
        Dict[str, torch.Tensor]: Padded ``input_ids`` (and associated masks) ready for the model.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = "longest"
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        has_token_type_ids = "token_type_ids" in features[0]
        end_with_eos = features[0]["extra"]["end_with_eos"]

        new_features = []
        for feature in features:
            seperate_ids = feature["seperate_ids"]
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            assert len(seperate_ids) == len(input_ids) == len(attention_mask)

            has_token_type_ids = False
            if "token_type_ids" in feature:
                has_token_type_ids = True
                token_type_ids = feature["token_type_ids"]
                assert len(token_type_ids) == len(input_ids)

            max_seperate_id = max(seperate_ids)
            prev_start_idx = 0
            for seperate_id in range(1, max_seperate_id + 1):
                start_idx = seperate_ids.index(seperate_id)

                new_feature = {}
                new_feature["input_ids"] = input_ids[prev_start_idx:start_idx]
                new_feature["attention_mask"] = attention_mask[prev_start_idx:start_idx]
                if has_token_type_ids:
                    new_feature["token_type_ids"] = token_type_ids[prev_start_idx:start_idx]
                new_features.append(new_feature)
                prev_start_idx = start_idx

            # last
            new_feature = {}
            new_feature["input_ids"] = input_ids[prev_start_idx:]
            new_feature["attention_mask"] = attention_mask[prev_start_idx:]
            if has_token_type_ids:
                new_feature["token_type_ids"] = token_type_ids[prev_start_idx:]
            new_features.append(new_feature)

        # remove features
        del features

        if end_with_eos:
            features = {}  # type: ignore
            features["input_ids"] = [  # type: ignore
                feature["input_ids"] + [self.tokenizer.eos_token_id] for feature in new_features
            ]
            features = self.tokenizer.pad(  # type: ignore
                features,
                padding=self.padding,
                return_attention_mask=True,
                return_tensors=return_tensors,
            )
        else:
            features = self.tokenizer.pad(  # type: ignore
                {"input_ids": [feature["input_ids"] for feature in new_features]},
                padding=self.padding,
                max_length=self.max_length,
                return_tensors=return_tensors,
            )
            features["attention_mask"] = self.tokenizer.pad(  # type: ignore
                {"input_ids": [feature["attention_mask"] for feature in new_features]},
                padding=self.padding,
                max_length=self.max_length,
                return_tensors=return_tensors,
            )["input_ids"]
            if has_token_type_ids:
                features["token_type_ids"] = self.tokenizer.pad(  # type: ignore
                    {"input_ids": [feature["token_type_ids"] for feature in new_features]},
                    padding=self.padding,
                    max_length=self.max_length,
                    return_tensors=return_tensors,
                )["input_ids"]

        return features  # type: ignore
