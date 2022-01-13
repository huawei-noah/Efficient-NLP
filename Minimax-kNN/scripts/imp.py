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


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("Impremium")


def read_data(data_path: str) -> Iterable[Tuple[int, str, int]]:
    with open(data_path, "r") as f:
        reader = csv.reader(f, doublequote=True)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) < 5:
                label, content = row[0], row[2]
            else:
                label, content = row[1], row[3]
            yield i + 1, content.strip().strip('"'), label


def export_jsonl(dataset: List[Tuple[int, str, int]], out_path: Path):
    with out_path.open("w") as writer:
        for id, text, label in dataset:
            data = {
                "guid": str(id),
                "text_a": text,
                "label": label,
            }
            writer.write(json.dumps(data) + "\n")


def export_tsv(dataset: List[Tuple[int, str, int]], out_path: Path):
    with out_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(("sentence", "label"))
        for _, text, label in dataset:
            writer.writerow((text, label))


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "data_file",
        default=None,
        type=str,
        help="Path of the positive review file.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        required=True,
        choices=("train", "test", "dev"),
        type=str,
        help="Split mode",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Path of the directory where the outputs will be saved.",
    )

    args = parser.parse_args()

    data = list(read_data(args.data_file))
    logger.info(f"#data {len(data)}")

    if args.output_dir is None:
        args.output_dir = "."

    output_dir = Path(args.output_dir) / "IMP"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        export_jsonl(data, output_dir / f"{args.mode}.jsonl")

    export_tsv(data, output_dir / f"{args.mode}.tsv")

    logger.info("Done!")


if __name__ == "__main__":
    main()
