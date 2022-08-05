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

"""
Requires to install:
 - nlpaug == 1.1.3
 - nltk == 3.5.0
"""
import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import spacy
from nlpaug.util.file.download import DownloadUtil
from sacremoses import MosesDetokenizer
from tqdm import tqdm
from transformers import InputExample

from glitter import GLUEv2Processor, copy_dev_file

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("HeuristicDA")

nlp = spacy.load("en_core_web_sm")
md = MosesDetokenizer(lang="en")


def _tokenize(text: str, as_string=False) -> Union[str, List[str]]:
    tokens = [tok.text for tok in nlp(text)]
    if as_string:
        return " ".join(tokens)
    else:
        return tokens


def _detokenize(tokens: Union[str, List[str]]) -> str:
    if isinstance(tokens, str):
        tokens = tokens.split()

    return md.detokenize(tokens)


def _to_ex_json(ex: InputExample) -> Dict[str, Any]:
    return _to_json(ex.text_a, ex.text_b, ex.guid, ex.label)


def _to_json(
    text_a: str, text_b: Optional[str], guid: str, label: Optional[str] = None, src: Optional[str] = None
) -> Dict[str, Any]:
    json_dict = dict(guid=guid, text_a=text_a)

    if text_b is not None:
        json_dict["text_b"] = text_b

    if label is not None:
        json_dict["label"] = label

    if src is not None:
        json_dict["src"] = src

    return json_dict


def generate(augmenter, example: InputExample, num_augments: int, field: str, src: str, tokenize: bool) -> List[Dict[str, Any]]:
    sentence = getattr(example, field)
    aug_sentences = augmenter.augment(sentence, n=num_augments)

    augmented_samples = []
    for j, aug_sentence in enumerate(aug_sentences):
        if tokenize:
            aug_sentence = _tokenize(aug_sentence, as_string=True)

        if field == "text_a":
            text_a = aug_sentence
            text_b = example.text_b
            if aug_sentence == example.text_a:
                continue
        else:
            text_a = example.text_a
            text_b = aug_sentence
            if aug_sentence == example.text_b:
                continue

        augmented_samples.append(_to_json(text_a, text_b, f"aug-{j + 1}", src=src))
    return augmented_samples


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--num_augments",
        default=8,
        type=int,
        help="Number of augmentations per samples",
    )
    parser.add_argument(
        "--syn_p",
        default=0.05,
        type=float,
        help="Percentage of words to be replaced with synonyms",
    )
    parser.add_argument(
        "--del_p",
        default=0.1,
        type=float,
        help="Percentage of words to be deleted",
    )
    parser.add_argument(
        "--syn_src",
        default="wordnet",
        choices=("wordnet", "glove"),
        type=str,
        help="Task name",
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        help="Task name",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Path of the directory where the outputs will be saved.",
    )
    parser.add_argument(
        "--cache_dir",
        default=".aug_cache",
        type=str,
        help="Path of the directory where the outputs will be saved.",
    )
    parser.add_argument(
        "--field",
        default="text_a",
        choices=("text_a", "text_b", "both"),
        type=str,
        help="Pivot field for augmentation",
    )
    parser.add_argument(
        "--tokenize",
        default=False,
        action="store_true",
        help="Whether to tokenize augmented example",
    )

    args = parser.parse_args()

    assert args.syn_p > 0

    if args.syn_src == "glove":
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        glove_path = os.path.join(args.cache_dir, "glove.6B.300d.txt")
        if not os.path.exists(glove_path):
            DownloadUtil.download_glove(model_name="glove.6B", dest_dir=args.cache_dir)
        syn_augmenter = naw.WordEmbsAug(
            model_type="glove",
            model_path=glove_path,
            action="substitute",
            aug_p=args.syn_p,
            tokenizer=_tokenize,
            reverse_tokenizer=_detokenize,
        )
    else:
        syn_augmenter = naw.SynonymAug(
            aug_src="wordnet",
            aug_p=args.syn_p,
            tokenizer=_tokenize,
            reverse_tokenizer=_detokenize,
        )

    src = f"syn={args.syn_src}"

    if args.del_p > 0:
        src += f"+del"
        augmenter = naf.Sequential(
            [
                syn_augmenter,
                naw.RandomWordAug(
                    aug_p=args.del_p,
                    tokenizer=_tokenize,
                    reverse_tokenizer=_detokenize,
                ),
            ]
        )
    else:
        augmenter = syn_augmenter

    processor = GLUEv2Processor(args.task)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "train.jsonl"

    examples = processor.get_train_examples(args.data_dir)

    with out_path.open("w", encoding="utf-8") as writer:
        for i, ex in enumerate(tqdm(examples)):
            out_json = _to_ex_json(ex)

            if args.field == "both":
                out_json["augmented_samples"] = generate(augmenter, ex, args.num_augments // 2, "text_a", src, args.tokenize)
                out_json["augmented_samples"].extend(generate(augmenter, ex, args.num_augments // 2, "text_b", src, args.tokenize))
            else:
                out_json["augmented_samples"] = generate(augmenter, ex, args.num_augments, args.field, src, args.tokenize)

            writer.write(json.dumps(out_json) + "\n")

    copy_dev_file(args.data_dir, output_dir)


if __name__ == "__main__":
    main()
