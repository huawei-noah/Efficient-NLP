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

import csv
import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping

import datasets

datasets.logging.set_verbosity_warning()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s][%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("hf")


def _prune(ditem):
    exp = {}
    for k, v in ditem.items():
        if k in ("guid", "text_a", "text_b"):
            exp[k] = v
        elif k == "label":
            if isinstance(v, str) or v >= 0:
                exp[k] = v
    return exp


def _export_jsonl(dataset, out_path: Path):
    with out_path.open("w", encoding="utf-8") as writer:
        for i in range(len(dataset)):
            writer.write(json.dumps(_prune(dataset[i])) + "\n")


def _export_tsv(dataset, has_two_cols: bool, out_path: Path):
    header = ("guid", "label", "text_a", "text_b") if has_two_cols else ("guid", "label", "text_a")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for i in range(len(dataset)):
            writer.writerow([dataset[i].get(col, None) for col in header])


LABEL_COLUMNS = {
    "trec": "label-coarse",
}

TEXT_COLUMNS = {
    "trec": "text",
    "glue/sst2": "sentence",
}

LABEL_TYPES = {
    "trec": "int2str",
}


def to_hf_dict(
    dataset_name: str,
    item: Mapping[str, Any],
    idx: int = 0,
    convert_fn=None,
) -> Mapping[str, Any]:
    item_dict = dict(
        guid=str(item.get("idx", idx) + 1),
        text_a=item[TEXT_COLUMNS[dataset_name]],
    )

    label_col = LABEL_COLUMNS.get(dataset_name, "label")
    if convert_fn is not None:
        label = convert_fn(item[label_col])
    else:
        label = item[label_col]
    item_dict["label"] = label

    return item_dict


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "dataset",
        type=str,
        help="Name of dataset based on Huggingface's datasets package.",
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
    parser.add_argument(
        "--output_format",
        type=str,
        default="jsonl",
        choices=("jsonl", "tsv"),
        help="If provided, dataset would be split based on this size",
    )
    parser.add_argument(
        "--dev_split", type=int, default=0, help="If provided, train dataset would be split based on this size"
    )

    args = parser.parse_args()

    logger.info(f"loading dataset: '{args.dataset}'")

    dataset = datasets.load_dataset(
        *args.dataset.split("/"),
        cache_dir=args.cache_dir,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    SPLIT_NAMES = {
        "validation": "dev"
    }

    for split in ("train", "validation", "test"):
        if split not in dataset:
            continue
            
        if split == "validation" and args.dev_split > 0:
            logger.warning("Validation set ignored because it is taken from training set")
            continue
            
        dataset_fold = dataset[split]
        label_col = dataset_fold.features[LABEL_COLUMNS.get(args.dataset, "label")]
        dataset_fold = dataset_fold.map(
            lambda ex, idx: to_hf_dict(args.dataset, ex, idx, convert_fn=getattr(label_col, LABEL_TYPES.get(args.dataset, ""), None)),
            with_indices=True,
        )

        if split == "train" and args.dev_split > 0:
            split_dataset = dataset_fold.train_test_split(test_size=args.dev_split)
            dataset_fold = split_dataset["train"]
            new_dataset = split_dataset["test"]
            _export_tsv(new_dataset, False, output_dir / "dev.tsv")

        if args.output_format == "tsv":
            _export_tsv(dataset_fold, False, output_dir / f"{SPLIT_NAMES.get(split, split)}.tsv")
        else:
            _export_jsonl(dataset_fold, output_dir / f"{SPLIT_NAMES.get(split, split)}.jsonl")

    logger.info("Done!")


if __name__ == "__main__":
    main()
