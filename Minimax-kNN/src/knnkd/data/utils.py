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

import json
from dataclasses import dataclass
from typing import Any, List, Optional, Mapping, Sequence

from transformers import InputFeatures, InputExample


def read_jsonl(input_file: str) -> List[Mapping[str, Any]]:
    """Reads a Jsonl file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f]


@dataclass(frozen=True)
class InputFeaturesV2(InputFeatures):
    example_index: Optional[int] = None
    augmented_indices: Optional[List[int]] = None
    augmented_features: Sequence[InputFeatures] = ()
    augmented_rank: Optional[int] = None


@dataclass
class InputExampleV2(InputExample):
    src: Optional[str] = None
    src_label: Optional[str] = None


@dataclass
class AugmentedInputExample(InputExample):
    augmented_examples: Sequence[InputExampleV2] = ()


@dataclass(frozen=True)
class DataSubset:
    name: str
    subset: Optional[str] = None
    split: str = "train"

    @property
    def key(self) -> str:
        if self.subset is None:
            return self.name
        else:
            return f"{self.name}/{self.subset}"

    @property
    def out_prefix(self) -> str:
        return self.subset or self.name

    @property
    def args(self) -> tuple:
        if self.subset is None:
            return (self.name,)
        else:
            return self.name, self.subset

    @classmethod
    def of(cls, key: str, split: str):
        sets = key.split("/")
        assert 1 <= len(sets) <= 2, "invalid key"
        return cls(*sets, split=split)
