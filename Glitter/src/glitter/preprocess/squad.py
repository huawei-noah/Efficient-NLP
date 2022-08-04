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
from pathlib import Path
from typing import Iterable, Tuple

import datasets

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("squad")


def _normalize(text: str) -> str:
    return text.replace("-lrb-", "(").replace("-rrb-", ")")


def read_original_file(data_path: Path) -> Iterable[Tuple[str, int]]:
    with data_path.open("r") as reader:
        for line in reader:
            space_index = line.index(" ")
            label = int(line[:space_index])
            text = line[space_index + 1 :].strip()
            yield _normalize(text), label


def export_jsonl(data, out_path: Path):
    with out_path.open("w") as writer:

        for i, ex in enumerate(data):
            data = {
                "id": ex["id"],
                "title": ex["title"],
                "question": ex["question"],
                "context": ex["context"],
                "answers": ex["answers"],
                "is_impossible": len(ex["answers"]["text"]) == 0,
            }
            writer.write(json.dumps(data) + "\n")


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path of the directory where the outputs will be saved.",
    )
    parser.add_argument(
        "--v2",
        default=False,
        action="store_true",
        help="Whether to convert SQuAD-v2 or not",
    )
    parser.add_argument(
        "--shifts",
        default=False,
        action="store_true",
        help="Whether to convert SQuADshifts or not",
    )

    args = parser.parse_args()

    squad_dataset = datasets.load_dataset(
        "squad_v2" if args.v2 else "squad",
        download_config=datasets.DownloadConfig(
            proxies=dict(http=os.getenv("HTTP_PROXY", None), https=os.getenv("HTTPS_PROXY", None))
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    export_jsonl(squad_dataset["train"], output_dir / "train.jsonl")
    export_jsonl(squad_dataset["validation"], output_dir / "dev.jsonl")

    if args.shifts:
        logger.info("Exporting SQuAD-shifts...")
        for subset in ("new_wiki", "nyt", "reddit", "amazon"):
            squad_dataset = datasets.load_dataset(
                "squadshifts",
                subset,
                download_config=datasets.DownloadConfig(
                    proxies=dict(http=os.getenv("HTTP_PROXY", None), https=os.getenv("HTTPS_PROXY", None))
                ),
            )

            output_dir = Path(args.output_dir) / f"squadshifts-{subset}"
            output_dir.mkdir(exist_ok=True, parents=True)
            export_jsonl(squad_dataset["test"], output_dir / "test.jsonl")

    logger.info("Done!")


if __name__ == "__main__":
    main()
