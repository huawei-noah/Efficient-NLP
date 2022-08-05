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

from typing import List, Optional, Union
from torch.utils.data.dataset import Dataset

from transformers import (
    InputExample,
    InputFeatures,
    PreTrainedTokenizerBase,
    glue_output_modes as hf_glue_output_modes,
    glue_tasks_num_labels as hf_glue_tasks_num_labels,
)

from ..data.utils import InputFeaturesV2, AugmentedInputExample
from .metrics import glue_metrics


class GLUEv2Dataset(Dataset):
    def __init__(
        self,
        examples: Union[List[InputExample], List[AugmentedInputExample]],
        labels: List[str],
        output_mode: str,
        max_length: int,
        tokenizer: PreTrainedTokenizerBase,
        max_augment_length: int = 0,
        num_augments: int = 0,
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

        self.max_augment_length = max_augment_length or max_length

        self.allowed_augment_indices = allowed_augment_indices
        if self.allowed_augment_indices is not None:
            self.flattened_augment_ranks = [
                (i, j)
                for i, indices in enumerate(self.allowed_augment_indices)
                for j in range(1 + min(num_augments, len(indices)))
            ]
        else:
            self.flattened_augment_ranks = None

        self.vanilla_augment = vanilla_augment
        self.padding = padding or "max_length"

    def label_from_example(self, example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if self.output_mode == "classification":
            return self.label_map[example.label]
        elif self.output_mode == "regression":
            return float(example.label)
        raise KeyError(self.output_mode)

    def __getitem__(self, index: int) -> Union[InputFeatures, InputFeaturesV2]:
        if self.flattened_augment_ranks is not None:
            ex_index, aug_rank = self.flattened_augment_ranks[index]
        else:
            ex_index, aug_rank = index, 0
        example = self.examples[ex_index]

        if self.vanilla_augment:
            if hasattr(example, "augmented_examples") and aug_rank > 0:
                if self.allowed_augment_indices is not None:
                    aug_index = self.allowed_augment_indices[ex_index][aug_rank - 1]
                else:
                    aug_index = aug_rank - 1

                ex = example.augmented_examples[aug_index]
                aug_index += 1
                label = -100
            else:
                aug_index = 0
                ex = example
                label = self.label_from_example(example)

            encoded_example = self.tokenizer(
                ex.text_a,
                ex.text_b,
                max_length=self.max_length if self.max_length > 0 else None,
                padding=self.padding,
                truncation=True,
            )

            return InputFeaturesV2(
                **encoded_example,
                label=label,
                example_index=ex_index,
                augmented_rank=aug_index,
            )

        encoded_example = self.tokenizer(
            example.text_a,
            example.text_b,
            max_length=self.max_length if self.max_length > 0 else None,
            padding=self.padding,
            truncation=True,
        )

        if hasattr(example, "augmented_examples") and aug_rank > 0:
            if self.allowed_augment_indices is not None:
                allowed_augmented_examples = [
                    example.augmented_examples[j] for j in self.allowed_augment_indices[ex_index]
                ]
            else:
                allowed_augmented_examples = example.augmented_examples

            augmented_features = [
                InputFeatures(
                    **self.tokenizer(
                        aug_ex.text_a,
                        aug_ex.text_b,
                        max_length=self.max_augment_length if self.max_augment_length > 0 else None,
                        padding=self.padding,
                        truncation=True,
                    )
                )
                for aug_ex in allowed_augmented_examples
            ]

            return InputFeaturesV2(
                **encoded_example,
                label=self.label_from_example(example),
                example_index=ex_index,
                augmented_indices=self.allowed_augment_indices[ex_index]
                if self.allowed_augment_indices is not None
                else None,
                augmented_features=augmented_features,
                augmented_rank=aug_rank,
            )
        else:
            return InputFeaturesV2(
                **encoded_example,
                label=self.label_from_example(example),
                example_index=ex_index,
                augmented_rank=aug_rank,
            )

    def __len__(self) -> int:
        return len(self.flattened_augment_ranks or self.examples)


glue_tasks_num_labels = {
    **hf_glue_tasks_num_labels,
    "mnli-mm": hf_glue_tasks_num_labels["mnli"],
    "mnli2": 3,  # to avoid reinitializing cls head
    "mnli2-mm": 3,  # to avoid reinitializing cls head
    "imdb": 2,
    "paws_qqp": 2,
}


glue_output_modes = {
    **hf_glue_output_modes,
    "mnli2": "classification",
    "mnli2-mm": "classification",
    "imdb": "classification",
    "paws_qqp": "classification",
}


glue_submission_names = {
    "cola": "CoLA",
    "mnli": "MNLI-m",
    "mnli-mm": "MNLI-mm",
    "mrpc": "MRPC",
    "sst-2": "SST-2",
    "sts-b": "STS-B",
    "qqp": "QQP",
    "qnli": "QNLI",
    "rte": "RTE",
    "mnli2": "MNLI2",
    "mnli2-mm": "MNLI2-mm",
    "imdb": "IMDb",
    "paws_qqp": "PAWS_QQP",
}


glue_submission_labels = {
    "cola": "integer",
    "mnli": "string",
    "mnli-mm": "string",
    "mrpc": "integer",
    "sst-2": "integer",
    "sts-b": "float",
    "qqp": "integer",
    "qnli": "string",
    "rte": "string",
    "mnli2": "integer",
    "mnli2-mm": "integer",
    "imdb": "string",
    "paws_qqp": "integer",
}


def glue_compute_metrics(task_name: str, preds, labels):
    assert task_name in glue_metrics, "Unknown task for computing metric"
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"

    if task_name in ("mnli2", "mnli2-mm"):
        preds = [0 if pred == 1 else 1 for pred in preds]

    metric_fn = glue_metrics[task_name]
    return metric_fn(preds, labels)
