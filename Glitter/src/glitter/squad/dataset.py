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
from typing import Dict, Iterable, List, Optional, Union

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils_base import PaddingStrategy

from glitter.squad.processors import SquadExample, AugmentedQuestion, SquadFeaturesType


def _prepare_example(
    ex: Union[SquadExample, AugmentedQuestion],
    index,
    context,
    max_length,
    doc_stride,
    fold,
    tokenizer,
    aug_index=None,
    augment_with_labels=False,
):
    pad_on_right = tokenizer.padding_side == "right"
    context_index = 1 if pad_on_right else 0

    tokenized_examples = tokenizer(
        ex.question if pad_on_right else context,
        context if pad_on_right else ex.question,
        max_length=max_length if max_length > 0 else None,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        truncation="only_second" if pad_on_right else "only_first",
    )
    tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    for j, offsets in enumerate(offset_mapping):
        encoded_example = {k: v[j] for k, v in tokenized_examples.items()}
        encoded_example["start_positions"] = []
        encoded_example["end_positions"] = []
        encoded_example["example_index"] = index
        if aug_index is not None:
            encoded_example["augment_index"] = aug_index

        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][j]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(j)

        # One example can give several spans, this is the index of the example containing this span of text.
        answer_starts = ex.answer_starts

        if aug_index is None or augment_with_labels:
            if len(answer_starts) == 0:
                encoded_example["start_positions"] = cls_index
                encoded_example["end_positions"] = cls_index
            else:
                # Start/end character index of the answer in the text.
                start_char = answer_starts[0]
                end_char = start_char + len(ex.answers[0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    encoded_example["start_positions"] = cls_index
                    encoded_example["end_positions"] = cls_index
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    encoded_example["start_positions"] = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    encoded_example["end_positions"] = token_end_index + 1
        else:
            encoded_example["start_positions"] = -100
            encoded_example["end_positions"] = -100

        if fold != "train":
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            encoded_example["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None) for k, o in enumerate(offsets)
            ]

        yield encoded_example


def convert_to_features(
    examples: List[SquadExample],
    tokenizer: PreTrainedTokenizerBase,
    fold: Optional[str],
    max_length: int,
    doc_stride: int,
    augment_with_labels: bool = False,
    num_augments: int = 0,
    verbose: bool = True,
) -> Iterable[SquadFeaturesType]:
    for i, ex in enumerate(tqdm(examples, disable=not verbose, desc=fold)):
        if ex.question is not None:
            yield from _prepare_example(ex, i, ex.context, max_length, doc_stride, fold, tokenizer)

        if ex.augmented_examples and num_augments != 0:
            augmented_features = [
                f
                for j, aug_ex in enumerate(ex.augmented_examples)
                if j < num_augments or num_augments < 0
                for f in _prepare_example(
                    aug_ex, i, ex.context, max_length, doc_stride, fold, tokenizer, j, augment_with_labels
                )
            ]

            yield augmented_features


@dataclass
class SquadFeatures:
    augment_rank: int
    features: SquadFeaturesType


class SquadDataset(Dataset):
    def __init__(
        self,
        examples: List[SquadExample],
        max_length: int,
        tokenizer: PreTrainedTokenizerBase,
        doc_stride: int,
        fold: str,
        features: Optional[List[SquadFeaturesType]] = None,
        num_augments: int = 0,
        allowed_augment_indices: Optional[List[List[int]]] = None,
        vanilla_augment: bool = False,
        padding: Optional[str] = None,
        verbose: bool = True,
    ):
        self.examples = examples
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride
        self.fold = fold

        self.vanilla_augment = vanilla_augment
        self.padding = padding or "max_length"
        self.verbose = verbose

        if features is None:
            self.features = list(
                convert_to_features(
                    self.examples,
                    self.tokenizer,
                    self.fold,
                    self.max_length,
                    self.doc_stride,
                    num_augments=num_augments,
                    verbose=self.verbose,
                )
            )
        else:
            self.features = features

        self.allowed_augment_indices = allowed_augment_indices

        self.flattened_indices = []
        for i, f in enumerate(self.features):
            if not isinstance(f, (list, tuple)):
                self.flattened_indices.append((i, 0))
            elif num_augments > 0:
                self.flattened_indices.extend((i, j) for j in range(1, 1 + min(num_augments, len(f))))

    def __getitem__(self, index: int) -> SquadFeatures:
        ex_index, aug_rank = self.flattened_indices[index]
        augment_indices = self.allowed_augment_indices[ex_index] if self.allowed_augment_indices is not None else None

        if aug_rank == 0:
            features = dict(self.features[ex_index])
            features.pop("offset_mapping", None)
            features["augment_index"] = -1
            if augment_indices is not None:
                assert isinstance(augment_indices, int)
                features["cache_index"] = augment_indices
            else:
                features["cache_index"] = ex_index
        elif self.vanilla_augment:
            features = dict(self.features[ex_index][aug_rank - 1])
            features.pop("offset_mapping", None)
            features["cache_index"] = augment_indices[aug_rank - 1]
            aug_rank = 0
        else:
            features = [dict(f) for f in self.features[ex_index]]
            for i, f in enumerate(features):
                f.pop("offset_mapping", None)
                f["cache_index"] = augment_indices[i]

        return SquadFeatures(aug_rank, features)

    def __len__(self) -> int:
        return len(self.flattened_indices)


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

    def __call__(self, batch: List[SquadFeatures]) -> Dict[int, Dict[str, torch.Tensor]]:
        grouped_features = defaultdict(list)
        for f in batch:
            if isinstance(f.features, (list, tuple)):
                grouped_features[f.augment_rank].extend(f.features)
            else:
                grouped_features[f.augment_rank].append(f.features)

        grouped_batch = {}
        for augment_rank, subfeatures in grouped_features.items():
            padded_batch = self.tokenizer.pad(
                subfeatures,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            padded_batch["example_indices"] = padded_batch.pop("example_index", None)
            padded_batch["augment_indices"] = padded_batch.pop("augment_index", None)
            padded_batch["cache_indices"] = padded_batch.pop("cache_index", None)

            if augment_rank > 0:
                padded_batch["augment_mask"] = torch.LongTensor(
                    [
                        i
                        for i, f in enumerate([f for f in batch if f.augment_rank == augment_rank])
                        for _ in range(len(f.features))
                    ]
                )

                padded_batch["augment_ranks"] = torch.LongTensor(
                    [
                        r + 1
                        for i, f in enumerate(batch)
                        if f.augment_rank == augment_rank
                        for r in range(len(f.features))
                    ]
                )

            grouped_batch[augment_rank] = padded_batch

        return grouped_batch
