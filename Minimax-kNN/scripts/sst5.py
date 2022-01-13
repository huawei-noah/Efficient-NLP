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
import ssl
import urllib.request
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Tuple

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("sst5")


SPLIT_URLS = {
    "dev": "https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.dev",
    "train": "https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.train",
    "test": "https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.test",
}


def _download(split: str, data_dir: str) -> str:
    logger.info(f"Downloading {split}...")
    output_file = os.path.join(data_dir, f"{split}.txt")
    urllib.request.urlretrieve(
        SPLIT_URLS[split],
        output_file,
    )
    return output_file


def _normalize(text: str) -> str:
    return text.replace("-lrb-", "(").replace("-rrb-", ")")


def read_original_file(data_path: Path) -> Iterable[Tuple[str, int]]:
    with data_path.open("r") as reader:
        for line in reader:
            space_index = line.index(" ")
            label = int(line[:space_index])
            text = line[space_index + 1 :].strip()
            yield _normalize(text), label


def export_jsonl(data_path: Path, out_path: Path):
    with out_path.open("w") as writer:
        for i, (text, label) in enumerate(read_original_file(data_path)):
            data = {
                "guid": str(i + 1),
                "text_a": text,
                "label": label,
            }
            writer.write(json.dumps(data) + "\n")


def export_tsv(data_path: Path, out_path: Path):
    with out_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(("sentence", "label"))
        for text, label in read_original_file(data_path):
            writer.writerow((text, label))


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Path of the directory where the outputs will be saved.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = Path("./SST-5")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_dir = output_dir / "original"
    original_dir.mkdir(exist_ok=True)

    for split in SPLIT_URLS.keys():
        original_file = Path(_download(split, str(original_dir)))
        if split == "train":
            export_jsonl(original_file, output_dir / f"{split}.jsonl")
        export_tsv(original_file, output_dir / f"{split}.tsv")

    logger.info("Done!")


if __name__ == "__main__":
    main()
