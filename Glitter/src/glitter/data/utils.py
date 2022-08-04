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
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Mapping, Sequence, Union

from transformers import InputFeatures, InputExample


def read_jsonl(input_file: str) -> List[Mapping[str, Any]]:
    """Reads a Jsonl file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f]


def copy_dev_file(data_dir: str, output_dir: Union[str, Path]):
    for ext in ("tsv", "jsonl"):
        dev_path = os.path.join(data_dir, f"dev.{ext}")
        if os.path.isfile(dev_path):
            shutil.copy(dev_path, output_dir)
            break


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

