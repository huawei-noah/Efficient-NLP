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
 - fairseq (https://github.com/facebookresearch/fairseq/):
    To work with latest version (as of 01-Jul-2022), we had to upgrade pytorch to 1.8.2 (lts) with CUDA 10.1

For more information, see https://github.com/facebookresearch/fairseq/tree/main/examples/backtranslation
"""
import json
import logging
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import spacy
import torch
from tqdm import tqdm
from transformers import InputExample

from glitter import GLUEv2Processor, copy_dev_file

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("backtranslation")

nlp = spacy.load("en_core_web_sm")


def _to_ex_json(ex: InputExample, guid: int) -> Dict[str, Any]:
    return _to_json(ex.text_a, ex.text_b, ex.guid or str(guid), ex.label)


def _tokenize(text: str) -> str:
    return " ".join([tok.text for tok in nlp(text)])


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


def generate(input_ids: List[List[int]], en2x, x2en, args, num_augments: Optional[int] = None) -> List[List[str]]:
    num_augments = num_augments or args.num_augments

    gen_kwargs = {}

    if args.do_sample:
        gen_kwargs["sampling"] = True
        if args.topp > 0:
            gen_kwargs["sampling_topp"] = args.topp
        else:
            gen_kwargs["sampling_topk"] = args.topk

    augmented_samples = []

    if args.max_seq_length > 0:
        input_ids = [t[:args.max_seq_length] for t in input_ids]

    translated_outputs = en2x.generate(input_ids, beam=num_augments, **gen_kwargs)
    backt_batch = [x2en.encode(en2x.decode(x["tokens"])) for best_outputs in translated_outputs for x in best_outputs]
    backt_outputs = x2en.generate(backt_batch, beam=num_augments, **gen_kwargs)

    i = 0
    for j in range(len(translated_outputs)):
        samples = set()
        for k in range(len(translated_outputs[j])):
            for y in backt_outputs[i]:
                augmented_text = x2en.decode(y["tokens"])
                if augmented_text != input_ids[j]:
                    samples.add(augmented_text)
            i += 1
        augmented_samples.append(list(samples)[:num_augments])

    return augmented_samples


def prepare_outputs(
    ex: InputExample, augmented_texts: List[str], field: str, lang: str, tokenize: bool
) -> List[Dict[str, Any]]:
    suffix = field[field.index("_") + 1 :]

    augmented_samples = []
    for j, aug_text in enumerate(augmented_texts):
        if tokenize:
            aug_text = _tokenize(aug_text)

        if field == "text_a":
            text_a = aug_text
            text_b = ex.text_b
        else:
            text_a = ex.text_a
            text_b = aug_text

        augmented_samples.append(_to_json(text_a, text_b, f"aug-{j + 1}", src=f"en-{lang}/{suffix}"))
    return augmented_samples


def prepare_batch(
    examples: List[InputExample],
    batch_generated_texts: List[List[str]],
    field: str,
    lang: str,
    start_index: int = 0,
    tokenize: bool = False,
) -> Iterable[Dict[str, Any]]:
    for i, (ex, augmented_samples) in enumerate(zip(examples, batch_generated_texts)):
        out_json = _to_ex_json(ex, start_index + i)
        out_json["augmented_samples"] = prepare_outputs(ex, augmented_samples, field, lang, tokenize)
        yield out_json


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--lang",
        default="de",
        choices=("de", "ru"),
        type=str,
        help="Other language to use for back-translation",
    )
    parser.add_argument(
        "--num_augments",
        default=8,
        type=int,
        help="Number of augmentations per samples",
    )
    parser.add_argument(
        "--do_sample",
        default=False,
        action="store_true",
        help="Whether to do sampling during translation",
    )
    parser.add_argument(
        "--topk",
        default=10,
        type=int,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--topp",
        default=0.0,
        type=float,
        help="Nucleus sampling",
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
        "--field",
        default="text_a",
        choices=("text_a", "text_b", "both"),
        type=str,
        help="Pivot field for augmentation",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--tokenize",
        default=False,
        action="store_true",
        help="Whether to tokenize augmented example",
    )
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="Max sequence length",
    )

    args = parser.parse_args()

    processor = GLUEv2Processor(args.task)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    en2x = torch.hub.load(
        "pytorch/fairseq",
        f"transformer.wmt19.en-{args.lang}",
        checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt",
        tokenizer="moses",
        bpe="fastbpe",
    )
    en2x.cuda()

    x2en = torch.hub.load(
        "pytorch/fairseq",
        f"transformer.wmt19.{args.lang}-en",
        checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt",
        tokenizer="moses",
        bpe="fastbpe",
    )
    x2en.cuda()

    out_path = output_dir / "train.jsonl"

    with out_path.open("w", encoding="utf-8") as writer:
        examples = processor.get_train_examples(args.data_dir)
        num_batches = math.ceil(len(examples) / args.batch_size)

        for i in tqdm(range(num_batches), total=num_batches):
            start_index = i * args.batch_size
            batch = examples[start_index : min(len(examples), (i + 1) * args.batch_size)]

            if args.field == "both":
                src_batch_a = [en2x.encode(getattr(ex, "text_a")) for ex in batch]
                generated_texts_a = generate(src_batch_a, en2x, x2en, args, args.num_augments // 2)
                augmented_a = prepare_batch(batch, generated_texts_a, "text_a", args.lang, start_index, args.tokenize)

                src_batch_b = [en2x.encode(getattr(ex, "text_b")) for ex in batch]
                generated_texts_b = generate(src_batch_b, en2x, x2en, args, args.num_augments // 2)
                augmented_b = prepare_batch(batch, generated_texts_b, "text_b", args.lang, start_index, args.tokenize)

                augmented_examples = []
                for a, b in zip(augmented_a, augmented_b):
                    combined = dict(a)
                    combined["augmented_samples"].extend(b["augmented_samples"])
                    augmented_examples.append(combined)
            else:
                src_batch = [en2x.encode(getattr(ex, args.field)) for ex in batch]
                generated_texts = generate(src_batch, en2x, x2en, args=args)
                augmented_examples = list(
                    prepare_batch(batch, generated_texts, args.field, args.lang, start_index, args.tokenize)
                )

            for ex_json in augmented_examples:
                writer.write(json.dumps(ex_json) + "\n")

    copy_dev_file(args.data_dir, output_dir)


if __name__ == "__main__":
    main()
