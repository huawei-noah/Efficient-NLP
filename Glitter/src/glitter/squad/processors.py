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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List, Sequence, Union

from transformers import DataProcessor

SquadFeaturesType = Union[Dict[str, Any], List[Dict[str, Any]]]


@dataclass
class AugmentedQuestion:
    id: str
    question: str
    answer: Optional[str] = None
    answer_start: Optional[int] = None
    src: Optional[str] = None

    @property
    def answers(self) -> List[str]:
        if self.answer is not None:
            return [self.answer]
        else:
            return []

    @property
    def answer_starts(self) -> List[int]:
        if self.answer_start is not None:
            return [self.answer_start]
        else:
            return []


@dataclass
class SquadExample:
    id: str
    title: str
    context: str
    question: Optional[str]
    answers: List[str]
    answer_starts: List[int]
    is_impossible: bool = False
    augmented_examples: Sequence[AugmentedQuestion] = ()

    @property
    def answers_json(self):
        return [{"text": ans, "answer_start": st} for ans, st in zip(self.answers, self.answer_starts)]

    @property
    def json_dict(self, v2: bool = False):
        dump = dict(id=self.id, title=self.title, context=self.context, question=self.question)
        if self.answers and self.answer_starts:
            dump["answers"] = self.answers_json
        if v2:
            dump["is_impossible"] = self.is_impossible
        return dump

    def clone(self) -> "SquadExample":
        return SquadExample(
            self.id,
            self.title,
            self.context,
            self.question,
            self.answers,
            self.answer_starts,
            self.is_impossible,
        )


def read_jsonl(data_file: str) -> Iterable[SquadExample]:
    with open(data_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            ex = json.loads(line.strip())

            id = ex["id"]
            answers = []
            answer_starts = []
            if "answers" in ex:
                if isinstance(ex["answers"], (list, tuple)):
                    answers = [ans["text"] for ans in ex["answers"]]
                    answer_starts = [ans["answer_start"] for ans in ex["answers"]]
                else:
                    answers = ex["answers"]["text"]
                    answer_starts = ex["answers"]["answer_start"]

            augmented_samples = []
            for i, aug_ex in enumerate(ex.get("augmented_samples", [])):
                aug_answer = aug_ex.get("answer", None)
                aug_answer_start = aug_ex.get("answer_start", None)

                if aug_answer is not None and isinstance(aug_answer, (list, tuple)):
                    aug_answer = aug_answer[0]

                if aug_answer_start is not None and isinstance(aug_answer_start, (list, tuple)):
                    aug_answer_start = aug_answer_start[0]

                if aug_answer is None and aug_answer_start is None and "answers" in aug_ex:
                    if aug_ex["answers"]:
                        aug_answer = aug_ex["answers"][0]["text"]
                        aug_answer_start = aug_ex["answers"][0]["answer_start"]

                augmented_samples.append(
                    AugmentedQuestion(
                        f"{id}|{aug_ex.get('id', 'aug-' + str(i + 1))}",
                        aug_ex["question"],
                        aug_answer,
                        aug_answer_start,
                        aug_ex.get("src", None),
                    )
                )

            yield SquadExample(
                id,
                ex["title"],
                ex["context"],
                ex.get("question", None),
                answers,
                answer_starts,
                ex.get("is_impossible", False),
                augmented_samples,
            )


class SquadProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_train_examples(self, data_dir: str):
        data_dir = Path(data_dir)
        train_json_file = data_dir / "train.jsonl"
        return list(read_jsonl(str(train_json_file)))

    def get_dev_examples(self, data_dir: str):
        data_dir = Path(data_dir)
        dev_json_file = data_dir / "dev.jsonl"
        return list(read_jsonl(str(dev_json_file)))

    def get_test_examples(self, data_dir: str):
        data_dir = Path(data_dir)
        test_json_file = data_dir / "test.jsonl"
        return list(read_jsonl(str(test_json_file)))

    def get_labelled_test_examples(self, data_dir: str):
        data_dir = Path(data_dir)
        test_file_path = data_dir / "test.jsonl"
        return list(read_jsonl(str(test_file_path)))

    def get_labels(self) -> List[str]:
        raise ValueError("labels in SQuAD are indices")
