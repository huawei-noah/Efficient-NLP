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

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data.dataset import Dataset
from transformers import (
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils_base import PaddingStrategy

from glitter.mc.processors import MultiChoiceExample, AugmentedMultiChoice


@dataclass
class MultiChoiceBatch:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    token_type_ids: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    example_indices: Optional[torch.Tensor] = None
    augmented_input_ids: Optional[torch.Tensor] = None
    augmented_attention_mask: Optional[torch.Tensor] = None
    augmented_token_types: Optional[torch.Tensor] = None
    augmented_indices: Optional[torch.Tensor] = None
    augment_mask: Optional[torch.Tensor] = None
    augment_ranks: Optional[torch.Tensor] = None

    def select_subset(self, attr: Union[str, Optional[torch.Tensor]], selected_indices) -> Optional[torch.Tensor]:
        if attr is not None and isinstance(attr, str):
            val = getattr(self, attr, None)
        else:
            val = attr

        return val[selected_indices] if val is not None else val

    @classmethod
    def sub_batches(cls, batch: Dict[int, Dict[str, Optional[torch.Tensor]]]) -> Dict[int, "MultiChoiceBatch"]:
        return {r: cls(**sub_batch) for r, sub_batch in batch.items() if sub_batch}


@dataclass
class MultiChoiceFeatures:
    features: Dict[str, Any]
    example_index: int
    augment_rank: Optional[int] = None
    augmented_indices: Optional[List[int]] = None
    augmented_features: Sequence[Dict[str, Any]] = ()


class MultiChoiceDataset(Dataset):
    def __init__(
        self,
        examples: List[MultiChoiceExample],
        max_length: int,
        tokenizer: PreTrainedTokenizerBase,
        num_augments: int = 0,
        allowed_augment_indices: Optional[List[List[int]]] = None,
        vanilla_augment: bool = False,
        augment_with_label: bool = False,
        padding: Optional[str] = None,
        verbose: bool = True,
    ):
        self.examples = examples
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.vanilla_augment = vanilla_augment
        self.augment_with_label = augment_with_label
        self.padding = padding or "max_length"
        self.verbose = verbose

        self.allowed_augment_indices = allowed_augment_indices
        if self.allowed_augment_indices is not None:
            self.flattened_augment_ranks = [
                (i, j)
                for i, indices in enumerate(self.allowed_augment_indices)
                for j in range(1 + min(num_augments, len(indices)))
            ]
        else:
            self.flattened_augment_ranks = None

    def _prepare(self, ex: MultiChoiceExample, aug_ex: Optional[AugmentedMultiChoice] = None):
        first_sentences = [(ex if aug_ex is None else aug_ex).question] * ex.num_choices
        second_sentences = ex.choices

        tokenized_example = self.tokenizer(
            first_sentences,
            second_sentences,
            max_length=self.max_length if self.max_length > 0 else None,
            padding=self.padding,
            truncation=True,
        )

        if aug_ex is None or self.augment_with_label:
            tokenized_example["label"] = int(ex.label)

        return tokenized_example

    def __getitem__(self, index: int) -> MultiChoiceFeatures:
        if self.flattened_augment_ranks is not None:
            ex_index, aug_rank = self.flattened_augment_ranks[index]
        else:
            ex_index, aug_rank = index, 0
        example = self.examples[ex_index]

        encoded_example = self._prepare(example)

        if aug_rank > 0:
            if self.vanilla_augment:
                if self.allowed_augment_indices is not None:
                    aug_index = self.allowed_augment_indices[ex_index][aug_rank - 1]
                else:
                    aug_index = aug_rank - 1

                encoded_aug = self._prepare(example, example.augmented_examples[aug_index])
                return MultiChoiceFeatures(
                    encoded_aug,
                    ex_index,
                    augmented_indices=[aug_rank],
                )
            else:
                if self.allowed_augment_indices is not None:
                    allowed_augmented_examples = [
                        example.augmented_examples[j] for j in self.allowed_augment_indices[ex_index]
                    ]
                else:
                    allowed_augmented_examples = example.augmented_examples

                augmented_features = [self._prepare(example, aug_ex) for aug_ex in allowed_augmented_examples]

                return MultiChoiceFeatures(
                    encoded_example,
                    ex_index,
                    aug_rank,
                    self.allowed_augment_indices[ex_index] if self.allowed_augment_indices is not None else None,
                    augmented_features,
                )
        else:
            return MultiChoiceFeatures(
                encoded_example,
                ex_index,
            )

    def __len__(self) -> int:
        return len(self.flattened_augment_ranks or self.examples)


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def _flatten(self, features):
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        return sum(flattened_features, [])

    def __call__(self, batch: List[MultiChoiceFeatures]) -> Dict[int, Dict[str, torch.Tensor]]:
        grouped_features = defaultdict(list)
        for f in batch:
            if f.augment_rank is not None:
                aug_rank = f.augment_rank
            else:
                aug_rank = 0
            grouped_features[aug_rank].append(f)

        grouped_batch = {}

        for augment_rank, subfeatures in grouped_features.items():
            group_size = len(subfeatures)

            features_to_pad = [s.features for s in subfeatures]
            labels = [feature.pop("label", -100) for feature in features_to_pad]

            flattened_features = self._flatten(features_to_pad)

            padded_batch = self.tokenizer.pad(
                flattened_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            num_choices = len(features_to_pad[0]["input_ids"])
            padded_batch = {k: v.view(group_size, num_choices, -1) for k, v in padded_batch.items()}
            # Add back labels
            padded_batch["labels"] = torch.tensor(labels, dtype=torch.int64)
            padded_batch["example_indices"] = torch.LongTensor([s.example_index for s in subfeatures])

            if augment_rank > 0:
                augmented_features = [augf for f in subfeatures for augf in f.augmented_features]
                flattened_augmented_features = self._flatten(augmented_features)
                augmented_batch = self.tokenizer.pad(
                    flattened_augmented_features,
                    padding=self.padding,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt",
                )
                aug_size = sum(len(f.augmented_features) for f in subfeatures)
                augmented_batch = {k: v.view(aug_size, num_choices, -1) for k, v in augmented_batch.items()}
                padded_batch["augmented_input_ids"] = augmented_batch["input_ids"]
                padded_batch["augmented_attention_mask"] = augmented_batch["attention_mask"]
                padded_batch["augmented_token_types"] = augmented_batch.get("token_type_ids", None)

                padded_batch["augment_mask"] = torch.LongTensor(
                    [
                        i
                        for i, f in enumerate(subfeatures)
                        if f.augment_rank == augment_rank
                        for _ in range(len(f.augmented_features))
                    ]
                )

                padded_batch["augment_ranks"] = torch.LongTensor(
                    [
                        r + 1
                        for i, f in enumerate(subfeatures)
                        if f.augment_rank == augment_rank
                        for r in range(len(f.augmented_features))
                    ]
                )

                padded_batch["augmented_indices"] = torch.LongTensor(
                    [aug_idx for f in subfeatures for aug_idx in f.augmented_indices]
                )

            grouped_batch[augment_rank] = padded_batch

        return grouped_batch
