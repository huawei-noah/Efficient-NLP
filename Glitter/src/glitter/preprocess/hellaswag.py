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

import datasets

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("HellaSwag")


def export_jsonl(data, out_path: Path):
    with out_path.open("w") as writer:

        for i, ex in enumerate(data):
            data = {
                "guid": ex["ind"],
                "ctx_a": ex["ctx_a"],
                "ctx_b": ex["ctx_b"],
                "context": ex["ctx"],
                "activity_label": ex["activity_label"],
                "split_type": ex["split_type"],
                "endings": ex["endings"],
            }
            if ex["label"]:
                data["label"] = int(ex["label"])
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

    args = parser.parse_args()

    dataset = datasets.load_dataset(
        "hellaswag",
        download_config=datasets.DownloadConfig(
            proxies=dict(http=os.getenv("HTTP_PROXY", None), https=os.getenv("HTTPS_PROXY", None))
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    export_jsonl(dataset["train"], output_dir / "train.jsonl")
    export_jsonl(dataset["validation"], output_dir / "dev.jsonl")
    export_jsonl(dataset["test"], output_dir / "test.jsonl")

    logger.info("Done!")


if __name__ == "__main__":
    main()
