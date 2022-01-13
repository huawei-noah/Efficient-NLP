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
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("CR")


def read_reviews(data_path: str) -> Iterable[str]:
    with open(data_path, "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                yield line


def split_reviews(
    pos_reviews: List[str], neg_reviews: List[str], train_ratio: float, dev_ratio: float
) -> Tuple[Tuple[List[str], List[str]], ...]:
    pos_train, pos_dev_test = train_test_split(pos_reviews, train_size=train_ratio, shuffle=True)
    neg_train, neg_dev_test = train_test_split(neg_reviews, train_size=train_ratio, shuffle=True)

    pos_dev, pos_test = train_test_split(pos_dev_test, train_size=dev_ratio, shuffle=True)
    neg_dev, neg_test = train_test_split(neg_dev_test, train_size=dev_ratio, shuffle=True)

    return (pos_train, neg_train), (pos_dev, neg_dev), (pos_test, neg_test)


def combine(pos_data: List[str], neg_data: List[str]) -> List[Tuple[str, str]]:
    dataset = [(review, "positive") for review in pos_data]
    dataset.extend([(review, "negative") for review in neg_data])
    np.random.shuffle(dataset)

    return dataset


def export_jsonl(dataset: List[Tuple[str, str]], out_path: Path):
    with out_path.open("w") as writer:
        for i, (text, label) in enumerate(dataset):
            data = {
                "guid": str(i + 1),
                "text_a": text,
                "label": label,
            }
            writer.write(json.dumps(data) + "\n")


def export_tsv(dataset: List[Tuple[str, str]], out_path: Path):
    with out_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(("sentence", "label"))
        for text, label in dataset:
            writer.writerow((text, label))


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--pos_file",
        default=None,
        type=str,
        required=True,
        help="Path of the positive review file.",
    )
    parser.add_argument(
        "--neg_file",
        default=None,
        type=str,
        required=True,
        help="Path of the negative review file.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Path of the directory where the outputs will be saved.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.66, help="Split ratio between train and dev/test")
    parser.add_argument("--dev_ratio", type=float, default=0.5, help="Split ratio between dev and test")

    args = parser.parse_args()

    pos_reviews = list(set(read_reviews(args.pos_file)))
    neg_reviews = list(set(read_reviews(args.neg_file)))
    logger.info(f"#pos {len(pos_reviews)} | #neg {len(neg_reviews)} -> {len(pos_reviews) + len(neg_reviews)}")

    if args.output_dir is None:
        args.output_dir = "."

    output_dir = Path(args.output_dir) / "CR"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data = split_reviews(pos_reviews, neg_reviews, args.train_ratio, args.dev_ratio)
    splits = ("train", "dev", "test")

    for mode, data in zip(splits, split_data):
        pos_data, neg_data = data
        total_size = len(pos_data) + len(neg_data)
        logger.info(
            f"{mode}: #pos {len(pos_data)} ({100 * len(pos_data) / total_size:.1f}%) | "
            f"#neg {len(neg_data)} ({100 * len(neg_data) / total_size:.1f}%) -> {total_size}"
        )
        dataset = combine(pos_data, neg_data)
        if mode == "train":
            export_jsonl(dataset, output_dir / f"{mode}.jsonl")

        export_tsv(dataset, output_dir / f"{mode}.tsv")

    logger.info("Done!")


if __name__ == "__main__":
    main()
