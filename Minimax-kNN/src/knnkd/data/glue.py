# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Union

import datasets
from torch.utils.data.dataset import Dataset
from transformers import (
    InputExample,
    InputFeatures,
    PreTrainedTokenizerBase,
    DataProcessor,
    glue_processors as hf_glue_processors,
    glue_output_modes as hf_glue_output_modes,
    glue_tasks_num_labels as hf_glue_tasks_num_labels,
)
from transformers.data.processors.glue import Sst2Processor

from .utils import InputFeaturesV2, InputExampleV2, AugmentedInputExample, read_jsonl

PathType = Union[str, Path]


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
        naive_augment: bool = False,
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

        self.naive_augment = naive_augment
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

        if self.naive_augment:
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
    "sst-5": 5,
    "ag_news": 4,
    "trec": 6,
    "cr": 2,
    "imp": 2,
}


glue_output_modes = {
    **hf_glue_output_modes,
    "sst-5": "classification",
    "ag_news": "classification",
    "trec": "classification",
    "cr": "classification",
    "imp": "classification",
}


class Sst5Processor(Sst2Processor):
    def get_labels(self):
        return ["0", "1", "2", "3", "4"]


class TrecProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "train")

    def get_labels(self) -> List[str]:
        """See base class."""
        return ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]

    def _create_examples(self, lines, set_type: str):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[-1]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class CrProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "train")

    def get_labels(self) -> List[str]:
        """See base class."""
        return ["negative", "positive"]

    def _create_examples(self, lines, set_type: str):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[0]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ImpProcessor(CrProcessor):
    def get_labels(self) -> List[str]:
        return ["0", "1"]


class AGNewsProcessor(TrecProcessor):
    def get_labels(self) -> List[str]:
        """See base class."""
        return ["World", "Sports", "Business", "Sci/Tech"]


glue_processors = {
    **hf_glue_processors,
    "sst-5": Sst5Processor,
    "trec": TrecProcessor,
    "ag_news": AGNewsProcessor,
    "cr": CrProcessor,
    "imp": ImpProcessor,
}


glue_submission_names = {
    "cola": "CoLA",
    "mnli": "MNLI-m",
    "mnli-mm": "MNLI-mm",
    "mrpc": "MRPC",
    "sst-2": "SST-2",
    "sst-5": "SST-5",
    "sts-b": "STS-B",
    "qqp": "QQP",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI",
    "trec": "TREC",
    "ag_news": "AGNews",
    "cr": "CR",
    "imp": "IMP",
}


glue_submission_labels = {
    "cola": "integer",
    "mnli": "string",
    "mnli-mm": "string",
    "mrpc": "integer",
    "sst-2": "integer",
    "sst-5": "integer",
    "sts-b": "float",
    "qqp": "integer",
    "qnli": "string",
    "rte": "string",
    "wnli": "integer",
    "trec": "string",
    "ag_news": "string",
    "cr": "string",
    "imp": "integer",
}


def load_metric(task_name: str) -> datasets.Metric:
    if task_name in ("sst-5", "ag_news", "trec", "cr", "imp"):
        task_name = "sst2"
    elif task_name == "mnli-mm":
        task_name = "mnli_mismatched"
    else:
        task_name = task_name.replace("-", "")

    return datasets.load_metric(
        "glue",
        task_name,
        experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    )


class GLUEv2Processor(DataProcessor):
    def __init__(self, task_name: Optional[str], num_augments: int = 0):
        self.task_name = task_name
        if task_name:
            self.hf_processor = glue_processors[self.task_name]()
        else:
            self.hf_processor = None
        self.num_augments = num_augments

    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_train_examples(self, data_dir: PathType):
        data_dir = Path(data_dir)
        train_json_file = data_dir / "train.jsonl"
        if train_json_file.exists():
            return self._create_examples(read_jsonl(str(train_json_file)))
        else:
            return self.hf_processor.get_train_examples(str(data_dir))

    def get_dev_examples(self, data_dir: PathType):
        data_dir = Path(data_dir)
        dev_json_file = data_dir / "dev.jsonl"
        if dev_json_file.exists():
            return self._create_examples(read_jsonl(str(dev_json_file)))
        else:
            return self.hf_processor.get_dev_examples(str(data_dir))

    def get_test_examples(self, data_dir: PathType):
        return self.hf_processor.get_test_examples(str(data_dir))

    def get_labelled_test_examples(self, data_dir: PathType):
        data_dir = Path(data_dir)
        return self.hf_processor._create_examples(
            self.hf_processor._read_tsv(os.path.join(data_dir, "test.tsv")), "dev"
        )

    def get_labels(self) -> List[str]:
        return self.hf_processor.get_labels()

    def _create_examples(self, lines: Iterable[Mapping[str, Any]]) -> List[AugmentedInputExample]:
        """Creates examples for the training, dev and test sets."""
        examples = []
        for line in lines:
            text_a = line["text_a"]
            text_b = line.get("text_b", None)
            guid = line["guid"]
            label = str(line.get("label", -100))

            augmented_examples = []
            if self.num_augments != 0 and "augmented_samples" in line:
                for k, nn_line in enumerate(line["augmented_samples"]):
                    if self.num_augments >= 0 and k >= self.num_augments:
                        break

                    src = nn_line.get("src", None)
                    src_label = nn_line.get("src_label", None)
                    nn_guid = f"{guid}|aux-{k + 1}"
                    augmented_examples.append(
                        InputExampleV2(
                            guid=nn_guid,
                            text_a=nn_line["text_a"],
                            text_b=nn_line.get("text_b", None),
                            src=src,
                            src_label=src_label,
                        )
                    )

            examples.append(AugmentedInputExample(guid, text_a, text_b, label, augmented_examples))

        return examples
