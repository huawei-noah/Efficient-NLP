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
import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Type

import datasets
from tqdm import tqdm

datasets.logging.set_verbosity_warning()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s][%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("glue")


@dataclass(frozen=True)
class DataSubset:
    name: str
    subset: Optional[str] = None

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
    def of(cls, key: str):
        sets = key.split("/")
        assert 1 <= len(sets) <= 2, "invalid key"
        return cls(*sets)


@dataclass
class TextClassMetaData:
    col_a: str
    col_b: Optional[str] = None
    convert_label: bool = False
    label_col: str = "label"
    label_type: Type = int

    @property
    def has_two_cols(self):
        return self.col_b is not None

    def to_hf_dict(
        self,
        item: Mapping[str, Any],
        idx: int = 0,
        convert_fn=None,
    ) -> Mapping[str, Any]:
        item_dict = dict(
            guid=str(item.get("idx", idx) + 1),
            text_a=item[self.col_a],
        )

        if self.has_two_cols:
            item_dict["text_b"] = item[self.col_b]

        label = item[self.label_col]
        if self.convert_label and convert_fn is not None and label >= 0:
            label = convert_fn(label)

        item_dict["label"] = self.label_type(label)

        return item_dict


supported_tasks = {
    "glue/cola": TextClassMetaData("sentence", label_type=str),
    "glue/sst2": TextClassMetaData("sentence"),
    "glue/mrpc": TextClassMetaData("sentence1", "sentence2"),
    "glue/qqp": TextClassMetaData("question1", "question2"),
    "glue/stsb": TextClassMetaData("sentence1", "sentence2", label_type=float),
    "glue/mnli": TextClassMetaData("premise", "hypothesis", convert_label=True, label_type=str),
    "glue/qnli": TextClassMetaData("question", "sentence", convert_label=True, label_type=str),
    "glue/rte": TextClassMetaData("sentence1", "sentence2", convert_label=True, label_type=str),
    "hans": TextClassMetaData("premise", "hypothesis", convert_label=True, label_type=str),
    "scitail/tsv_format": TextClassMetaData("premise", "hypothesis"),
}

split_names = {
    "validation": "dev",
    "validation_matched": "dev_matched",
    "validation_mismatched": "dev_mismatched",
}


def _prune(ditem, with_labels: bool = True):
    return {
        k: v
        for k, v in ditem.items()
        if k
        in (
            "guid",
            "text_a",
            "text_b",
        )
        or (with_labels and k == "label")
    }


def export_jsonl(dataset, with_labels: bool, out_path: Path):
    with out_path.open("w", encoding="utf-8") as writer:
        for i in range(len(dataset)):
            writer.write(json.dumps(_prune(dataset[i], with_labels)) + "\n")


def convert(split: str, subset, md, args):
    dataset = datasets.load_dataset(
        *subset.args,
        split=split,
        cache_dir=args.cache_dir,
        download_config=datasets.DownloadConfig(
            proxies=dict(http=os.getenv("HTTP_PROXY", None), https=os.getenv("HTTPS_PROXY", None))
        ),
    )
    dataset = dataset.map(
        lambda ex, idx: md.to_hf_dict(ex, idx, convert_fn=dataset.features[md.label_col].int2str), with_indices=True
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_jsonl(dataset, not split.startswith("test"), output_dir / f"{split_names.get(split, split)}.jsonl")


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        required=True,
        help="Name of dataset to augment. The supported names are: " + ", ".join(supported_tasks.keys()),
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Path of the directory where the outputs will be saved.",
    )

    args = parser.parse_args()

    logger.info(f"loading dataset: '{args.dataset}'")

    subset = DataSubset.of(args.dataset)
    md = supported_tasks[subset.key]

    splits = ("train", "validation", "test")

    if subset.subset == "mnli":
        splits = ("train", "validation_matched", "validation_mismatched", "test_matched", "test_mismatched")

    for split in tqdm(splits, desc="Split"):
        convert(split, subset, md, args)

    logger.info(f"done!")


if __name__ == "__main__":
    main()
