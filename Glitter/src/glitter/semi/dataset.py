# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import torch
from torch.utils.data.dataset import Dataset
from transformers import (
    InputExample,
    InputFeatures,
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils_base import PaddingStrategy

from ..data.utils import AugmentedInputExample
from ..data.collator import _as_dict as as_dict

BatchTensor = Mapping[str, torch.Tensor]
DynamicAugBatchTensor = List[Tuple[Optional[torch.Tensor], ...]]


@dataclass(frozen=True)
class SemiInputFeatures(InputFeatures):
    is_unsup: bool = False
    example_index: Optional[int] = None
    augmented_rank: Optional[int] = None
    unsup_feature: Optional[InputFeatures] = None
    augmented_indices: Optional[List[int]] = None
    augmented_features: Sequence[InputFeatures] = ()


class SemiGLUEDataset(Dataset):
    def __init__(
        self,
        examples: List[AugmentedInputExample],
        labels: List[str],
        output_mode: str,
        max_length: int,
        tokenizer: PreTrainedTokenizerBase,
        num_augments: int,
        sup_ratio: float,
        sup_size: int = 0,
        allowed_augment_indices: Optional[List[List[int]]] = None,
        vanilla_augment: bool = False,
        padding: Optional[str] = None,
    ):
        self.examples = examples

        self.label_map = {label: i for i, label in enumerate(labels)}
        self.label_map[-100] = -100
        self.output_mode = output_mode

        self.max_length = max_length
        self.tokenizer = tokenizer

        self.sup_ratio = sup_ratio
        if sup_size > 0:
            self.sup_size = min(sup_size, len(self.examples))
        else:
            self.sup_size = int(len(self.examples) * self.sup_ratio)

        self.allowed_augment_indices = allowed_augment_indices
        if self.allowed_augment_indices is not None:
            self.sup_indices = self._generate_balanced_supervised_data()
            self.flattened_augment_ranks = [
                (i, j)
                for i, indices in enumerate(self.allowed_augment_indices)
                for j in range(1, 1 + min(num_augments, len(indices)))
            ]
            self.flattened_augment_ranks.extend(
                (i, 0) for i, indices in enumerate(self.allowed_augment_indices) if i in self.sup_indices
            )
        else:
            self.sup_indices = None
            self.flattened_augment_ranks = None

        self.vanilla_augment = vanilla_augment
        self.padding = padding or "max_length"

    def _generate_balanced_supervised_data(self) -> Set[int]:
        example_indices_by_label = defaultdict(list)
        sup_indices = set()
        for i, ex in enumerate(self.examples):
            example_indices_by_label[ex.label].append(i)

        if not sup_indices:
            num_labels = len(example_indices_by_label)
            for lbl, indices in example_indices_by_label.items():
                if self.sup_ratio == 1.0:
                    sample_size = len(indices)
                else:
                    sample_size = self.sup_size // num_labels
                if sample_size >= len(indices):
                    sup_indices.update(indices)
                else:
                    sup_indices.update(random.sample(indices, sample_size))

        return sup_indices

    def label_from_example(self, example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if self.output_mode == "classification":
            return self.label_map[example.label]
        elif self.output_mode == "regression":
            return float(example.label)
        raise KeyError(self.output_mode)

    def _default_label(self) -> Union[int, float]:
        if self.output_mode == "classification":
            return -100
        elif self.output_mode == "regression":
            return -100.0
        raise KeyError(self.output_mode)

    def __getitem__(self, index: int) -> SemiInputFeatures:
        if self.flattened_augment_ranks is not None:
            ex_index, aug_rank = self.flattened_augment_ranks[index]
        else:
            ex_index, aug_rank = index, 0

        is_unsup = aug_rank > 0

        example = self.examples[ex_index]
        label = self.label_from_example(example)

        encoded_example = self.tokenizer(
            example.text_a,
            example.text_b,
            max_length=self.max_length if self.max_length > 0 else None,
            padding=self.padding,
            truncation=True,
        )

        if is_unsup:
            if self.vanilla_augment:
                aug_index = self.allowed_augment_indices[ex_index][aug_rank - 1]

                unsup_ex = example.augmented_examples[aug_index]
                unsup_encoded_ex = self.tokenizer(
                    unsup_ex.text_a,
                    unsup_ex.text_b,
                    max_length=self.max_length if self.max_length > 0 else None,
                    padding=self.padding,
                    truncation=True,
                )

                return SemiInputFeatures(
                    **encoded_example,
                    label=label,
                    is_unsup=is_unsup,
                    example_index=ex_index,
                    augmented_rank=aug_rank,
                    augmented_indices=[aug_index + 1],
                    unsup_feature=InputFeatures(**unsup_encoded_ex, label=self._default_label()),
                )
            else:
                allowed_augmented_examples = [
                    example.augmented_examples[j] for j in self.allowed_augment_indices[ex_index]
                ]

                augmented_features = [
                    InputFeatures(
                        **self.tokenizer(
                            aug_ex.text_a,
                            aug_ex.text_b,
                            max_length=self.max_length if self.max_length > 0 else None,
                            padding=self.padding,
                            truncation=True,
                        ),
                        label=self._default_label(),
                    )
                    for aug_ex in allowed_augmented_examples
                ]

                return SemiInputFeatures(
                    **encoded_example,
                    label=label,
                    is_unsup=is_unsup,
                    example_index=ex_index,
                    augmented_rank=aug_rank,
                    augmented_indices=self.allowed_augment_indices[ex_index],
                    augmented_features=augmented_features,
                )
        else:
            return SemiInputFeatures(
                **encoded_example, label=label, is_unsup=is_unsup, example_index=ex_index, augmented_rank=aug_rank
            )

    def __len__(self) -> int:
        return len(self.flattened_augment_ranks or self.examples)


@dataclass
class SemiFeaturesCollator:
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
    group_features: bool = False
    num_augment_ranks: int = 0

    def _pad(self, features: List[Dict[str, Any]]) -> BatchTensor:
        batch = self.tokenizer.pad(
            features,
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

        return batch

    def _pad_input_features(self, features: List[SemiInputFeatures]) -> BatchTensor:
        converted_features = [
            as_dict(
                f,
                [
                    "is_unsup",
                    "augmented_features",
                    "augmented_indices",
                    "example_index",
                    "augmented_rank",
                    "unsup_feature",
                ],
            )
            for f in features
        ]

        return self._pad(converted_features)

    def _pad_unsup_features(self, features: List[SemiInputFeatures]) -> BatchTensor:
        return self._pad([as_dict(f.unsup_feature) for f in features])

    def _grouped_pad(self, features: List[SemiInputFeatures]) -> DynamicAugBatchTensor:
        assert self.group_features, "group_features must be True to invoke this method"

        grouped_features = defaultdict(list)
        feature_indices = defaultdict(list)
        for i, f in enumerate(features):
            grouped_features[f.augmented_rank].append(f)
            feature_indices[f.augmented_rank].append(i)

        grouped_batch = [tuple()] * (self.num_augment_ranks + 1)

        for aug_rank, subfeatures in grouped_features.items():
            augmented_batch = self._pad([as_dict(augf) for f in subfeatures for augf in f.augmented_features])
            augmented_mask = torch.LongTensor(
                [i for i, f in enumerate(subfeatures) for _ in range(len(f.augmented_features))]
            )
            augmented_ranks = torch.LongTensor([r + 1 for f in subfeatures for r in range(len(f.augmented_features))])
            augmented_indices = torch.LongTensor([aug_idx for f in subfeatures for aug_idx in f.augmented_indices])

            batch_indices = feature_indices[aug_rank]

            grouped_batch[aug_rank] = (
                batch_indices,
                augmented_indices,
                augmented_mask,
                augmented_ranks,
                augmented_batch["input_ids"],
                augmented_batch.get("attention_mask", None),
                augmented_batch.get("token_type_ids", None),
            )

        return grouped_batch

    def __call__(
        self, features: List[SemiInputFeatures]
    ) -> Tuple[Optional[BatchTensor], Optional[BatchTensor], Optional[BatchTensor], Optional[DynamicAugBatchTensor]]:
        sup_batch = None
        sup_features = [f for f in features if not f.is_unsup]
        if sup_features:
            sup_batch = self._pad_input_features(sup_features)

        unsup_orig_batch = None
        unsup_aug_batch = None
        unsup_grouped_aug_batch = None
        unsup_features = [f for f in features if f.is_unsup]
        if unsup_features:
            unsup_orig_batch = self._pad_input_features(unsup_features)

            if self.group_features:
                unsup_grouped_aug_batch = self._grouped_pad(unsup_features)
            else:
                unsup_aug_batch = self._pad_unsup_features(unsup_features)

        return (sup_batch, unsup_orig_batch, unsup_aug_batch, unsup_grouped_aug_batch)
