# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Changes:
# 2021.12.10 - FeaturesCollatorWithPadding: added support for padding augmented data
#               Huawei Technologies Co., Ltd. <foss@huawei.com>

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from transformers import InputFeatures, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy

from .utils import InputFeaturesV2


def _as_dict(obj, exclude_fields: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    exclude_fields = exclude_fields or []
    return {k: v for k, v in asdict(obj).items() if v is not None and k not in exclude_fields}


@dataclass
class FeaturesCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding
            index) among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
              single sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
            >= 7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    group_features: bool = True
    num_augment_ranks: int = 0

    def __call__(
        self, features: Union[List[InputFeatures], List[InputFeaturesV2]]
    ) -> List[Tuple[Optional[torch.Tensor], ...]]:

        grouped_features = defaultdict(list)
        for f in features:
            if self.group_features and hasattr(f, "augmented_rank"):
                aug_rank = f.augmented_rank
            else:
                aug_rank = 0

            grouped_features[aug_rank].append(f)

        grouped_batch = [tuple()] * (self.num_augment_ranks + 1)

        for aug_rank, subfeatures in grouped_features.items():
            batch = self.tokenizer.pad(
                [
                    _as_dict(f, ["augmented_features", "augmented_indices", "example_index", "augmented_rank"])
                    for f in subfeatures
                ],
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            augmented_batch = {}
            augmented_mask = None
            augmented_ranks = None
            if any(hasattr(f, "augmented_features") and f.augmented_features for f in subfeatures):
                augmented_batch = self.tokenizer.pad(
                    [_as_dict(augf) for f in subfeatures for augf in f.augmented_features],
                    padding=self.padding,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt",
                )
                augmented_mask = torch.LongTensor(
                    [i for i, f in enumerate(subfeatures) for _ in range(len(f.augmented_features))]
                )
                augmented_ranks = torch.LongTensor(
                    [r + 1 for f in subfeatures for r in range(len(f.augmented_features))]
                )

            augmented_indices = None
            if self.group_features:
                if any(hasattr(f, "augmented_indices") and f.augmented_indices is not None for f in subfeatures):
                    augmented_indices = torch.LongTensor([aug_idx for f in subfeatures for aug_idx in f.augmented_indices])
            else:
                if any(hasattr(f, "augmented_rank") and f.augmented_rank is not None for f in subfeatures):
                    augmented_indices = torch.LongTensor([f.augmented_rank for f in subfeatures])

            example_indices = None
            if any(hasattr(f, "example_index") and f.example_index is not None for f in subfeatures):
                example_indices = torch.LongTensor([f.example_index for f in subfeatures])

            grouped_batch[aug_rank] = (
                batch["input_ids"],
                batch.get("attention_mask", None),
                batch.get("token_type_ids", None),
                batch.get("labels", None),
                augmented_batch.get("input_ids", None),
                augmented_batch.get("attention_mask", None),
                augmented_batch.get("token_type_ids", None),
                example_indices,
                augmented_indices,
                augmented_mask,
                augmented_ranks,
            )

        return grouped_batch


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """
    Copied from fairseq: https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/fairseq/data/data_utils.py#L34
    Convert a list of 1d tensors into a padded 2d tensor.
    """
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res
