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

import os
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Union

from transformers import (
    InputExample,
    DataProcessor,
    glue_processors as hf_glue_processors,
)
from transformers.data.processors.glue import RteProcessor, QqpProcessor

from ..data.utils import InputExampleV2, AugmentedInputExample, read_jsonl

PathType = Union[str, Path]


class IMDbReviewProcessor(DataProcessor):
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
        return ["Negative", "Positive"]

    def _create_examples(self, lines, set_type: str):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[1]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class PawsQqpProcessor(QqpProcessor):
    def _create_examples(self, lines, set_type: str):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[3]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


glue_processors = {
    **hf_glue_processors,
    "mnli2": RteProcessor,
    "mnli2-mm": RteProcessor,
    "imdb": IMDbReviewProcessor,
    "paws_qqp": PawsQqpProcessor,
}


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
