# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Changes:
# 2022.06.30 - generating new data based on span-based masking along with code for saving the output in a file
#               Huawei Technologies Co., Ltd. <foss@huawei.com>

import json
import math
import logging
import os
import random
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from sacremoses import MosesDetokenizer
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    InputExample,
    PreTrainedTokenizerBase,
)
from transformers.data.data_collator import DataCollatorWithPadding

from glitter import GLUEv2Processor, copy_dev_file

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("mlm")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

TokenList = List[str]
TokenIdList = List[int]
detok = MosesDetokenizer()
nlp = spacy.load("en_core_web_sm")
nlp_lite = spacy.load("en_core_web_sm", disable=("tagger", "ner", "parser", "lemmatizer"))
CUSTOM_TOK_RULES = {
    "'m": ("'", "m"),
    "'s": ("'", "s"),
    "'d": ("'", "d"),
    "'ll": ("'", "ll"),
    "'re": ("'", "re"),
    "'ve": ("'", "ve"),
    "''": ("'", "'"),
}

MULTITOK_RULES = {
    ("ai", "n't"): ("ain", "'", "t"),
    ("could", "n't"): ("couldn", "'", "t"),
    ("would", "n't"): ("wouldn", "'", "t"),
    ("should", "n't"): ("shouldn", "'", "t"),
    ("does", "n't"): ("doesn", "'", "t"),
}

REGEX_TOK_RULES = (
    (r"^([a-zA-Z].*)(,)(')(.+)$", lambda m: (m.group(1), m.group(2), m.group(3), m.group(4))),
    (r"^([a-zA-Z].*)([.,])$", lambda m: (m.group(1), m.group(2))),
    (r"^([a-zA-Z].*)(')(.+)$", lambda m: (m.group(1), m.group(2), m.group(3))),
    (r"^(\.{2,})$", lambda m: list(m.group(1))),
)


def _apply_rules(_nlp, text: str) -> List[Tuple[str, Any]]:
    tokens = list(_nlp(text))

    custom_tokens = []
    skip = False
    for i, t in enumerate(tokens):
        if skip:
            skip = False
            continue

        if i < len(tokens) - 1 and (t.text, tokens[i + 1].text) in MULTITOK_RULES:
            for tt in MULTITOK_RULES[(t.text, tokens[i + 1].text)]:
                custom_tokens.append((tt, t))
                skip = True
        elif t.text in CUSTOM_TOK_RULES:
            for tt in CUSTOM_TOK_RULES[t.text]:
                custom_tokens.append((tt, t))
        else:
            for rge, token_rule in REGEX_TOK_RULES:
                m = re.match(rge, t.text)
                if m is not None:
                    for tt in token_rule(m):
                        custom_tokens.append((tt, t))
                    break
            else:
                custom_tokens.append((t.text, t))

    return custom_tokens


def _tokenize(text: str, as_string: bool = True) -> Union[str, List[str]]:
    tokens = _apply_rules(nlp_lite, text)

    tokens = [txt for txt, _ in tokens]

    if as_string:
        return " ".join(tokens)
    else:
        return tokens


def _ner(text: str) -> List[bool]:
    tokens = [t for _, t in _apply_rules(nlp, text)]
    ents = [t.ent_iob_ == "O" for t in tokens]

    return ents


def _to_ex_json(ex: InputExample, guid: int) -> Dict[str, Any]:
    return _to_json(ex.text_a, ex.text_b, str(guid), ex.label)


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


def _is_start_word(tokenizer: PreTrainedTokenizerBase, token_id: Union[int, str]) -> bool:
    if isinstance(token_id, str):
        token = token_id
    else:
        token = tokenizer._convert_id_to_token(token_id)
    return not token.startswith("##") or token.startswith("Ġ")


class ParagraphInfo(object):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def get_word_piece_map(self, sentence: TokenIdList) -> List[bool]:
        return [_is_start_word(self.tokenizer, tok) for tok in sentence]

    def get_word_at_k(self, sentence, left, right, k, word_piece_map=None):
        num_words = 0
        while num_words < k and right < len(sentence):
            # complete current word
            left = right
            right = self.get_word_end(sentence, right, word_piece_map)
            num_words += 1
        return left, right

    def get_word_start(self, sentence, anchor, word_piece_map=None):
        word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
        left = anchor
        while left > 0 and word_piece_map[left] == False:
            left -= 1
        return left

    # word end is next word start
    def get_word_end(self, sentence, anchor, word_piece_map=None):
        word_piece_map = word_piece_map if word_piece_map is not None else self.get_word_piece_map(sentence)
        right = anchor + 1
        while right < len(sentence) and word_piece_map[right] == False:
            right += 1
        return right


class MaskingScheme:
    """
    Copied from SpanBERT: https://github.com/facebookresearch/SpanBERT/blob/188d1c32a7840f1049d9fb6795dcb925eb034620/pretraining/fairseq/data/masking.py#L43
    """

    def __init__(self, **kwargs):
        self.mask_ratio = kwargs.pop("mask_ratio", None)

    def mask(self, tokens, **kwargs):
        raise NotImplementedError()


class PairWithSpanMaskingScheme(MaskingScheme):
    """
    Copied from SpanBERT: https://github.com/facebookresearch/SpanBERT/blob/188d1c32a7840f1049d9fb6795dcb925eb034620/pretraining/fairseq/data/masking.py#L72
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_ratio: float = 0.15,
        span_lower: int = 1,
        span_upper: int = 1,
        geometric_p: float = 0.2,
        random_token_prob: float = 0.1,
        leave_unmasked_prob: float = 0.1,
    ):
        super().__init__(mask_ratio=mask_ratio)
        self.lower = span_lower
        self.upper = span_upper

        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id
        self.paragraph_info = ParagraphInfo(tokenizer)

        self.lens = list(range(self.lower, self.upper + 1))
        self.p = geometric_p
        self.len_distrib = (
            [self.p * (1 - self.p) ** (i - self.lower) for i in range(self.lower, self.upper + 1)]
            if self.p >= 0
            else None
        )
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]

        self.random_token_prob = random_token_prob
        self.leave_unmasked_prob = leave_unmasked_prob

        if self.random_token_prob > 0:
            weights = np.ones(len(tokenizer.get_vocab()))
            weights[tokenizer.all_special_ids] = 0
            for k, v in tokenizer.get_vocab().items():
                if "[unused" in k:
                    weights[v] = 0
            self.weights = weights / weights.sum()
        else:
            self.weights = None

        print(self.len_distrib, self.lens)

    def mask(self, sentence, include_indices: Optional[List[bool]] = None):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.mask_ratio)

        mask = set()
        word_piece_map = self.paragraph_info.get_word_piece_map(sentence)

        spans = []
        retries = 0
        while len(mask) < mask_num and retries < 5:
            span_len = np.random.choice(self.lens, p=self.len_distrib)
            anchor = np.random.choice(sent_length)
            if anchor in mask:
                retries += 1
                continue

            if sentence[anchor] in self.tokenizer.all_special_ids:
                retries += 1
                continue

            # find word start, end
            left1, right1 = (
                self.paragraph_info.get_word_start(sentence, anchor, word_piece_map),
                self.paragraph_info.get_word_end(sentence, anchor, word_piece_map),
            )

            if any(sentence[t] in self.tokenizer.all_special_ids for t in range(left1, right1 + 1)):
                retries += 1
                continue

            if include_indices is not None:
                if any(not include_indices[t] for t in range(left1, right1 + 1)):
                    retries += 1
                    continue

            retries = 0

            spans.append([left1, left1])
            for i in range(left1, right1):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i
            num_words = 1
            right2 = right1
            while num_words < span_len and right2 < len(sentence) and len(mask) < mask_num:
                # complete current word
                left2 = right2
                right2 = self.paragraph_info.get_word_end(sentence, right2, word_piece_map)
                num_words += 1
                for i in range(left2, right2):
                    if len(mask) >= mask_num:
                        break
                    mask.add(i)
                    spans[-1][-1] = i
        sentence, target, pair_targets = span_masking(
            sentence,
            spans,
            self.pad,
            self.mask_id,
            mask,
            self.random_token_prob,
            self.leave_unmasked_prob,
            self.weights,
        )

        return sentence, target, pair_targets


def span_masking(
    sentence,
    spans,
    pad,
    mask_id,
    mask,
    random_token_prob: float = 0.0,
    leave_token_prob: float = 0.0,
    vocab_weights=None,
):
    """
    Copied from SpanBERT
    Args:
        sentence:
        spans:
        tokens:
        pad:
        mask_id:
        pad_len:
        mask:

    Returns:

    """
    sentence = np.copy(sentence)
    sent_length = len(sentence)
    target = np.full(sent_length, pad)
    pair_targets = []
    spans = merge_intervals(spans)
    assert len(mask) == sum([e - s + 1 for s, e in spans])
    # print(list(enumerate(sentence)))
    for start, end in spans:
        lower_limit = 0
        upper_limit = sent_length - 1
        if start > lower_limit and end < upper_limit:
            pair_targets += [[start, end + 1]]
            # pair_targets[-1] += [sentence[i] for i in range(start, end + 1)]
        for i in range(start, end + 1):
            assert i in mask
            target[i] = sentence[i]
            if random_token_prob > 0 or leave_token_prob > 0:
                rand = np.random.random()
                if rand < random_token_prob:
                    sentence[i] = np.random.choice(
                        len(vocab_weights),
                        p=vocab_weights,
                    )
                elif rand < random_token_prob + leave_token_prob:
                    sentence[i] = target[i]
                else:
                    sentence[i] = mask_id
            else:
                sentence[i] = mask_id

    # pair_targets = pad_to_len(pair_targets, pad, pad_len + 2)
    # if pair_targets is None:
    return sentence, target, pair_targets


def merge_intervals(intervals):
    """
    Copied from SpanBERT
    Args:
        intervals:

    Returns:

    """
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


class _Dataset(Dataset):
    def __init__(
        self,
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizerBase,
        mask_scheme: MaskingScheme,
        field: str,
        tokenized: bool,
        max_length: int,
        num_augments: int,
        keep_entities: bool = True,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.mask_scheme = mask_scheme

        self.field = field
        self.tokenized = tokenized
        self.max_length = max_length

        self.num_augments = num_augments
        self.keep_entities = keep_entities

        self._data = list(self._prepare())

    def __len__(self):
        return len(self._data)

    def _prepare(self):
        for i, ex in tqdm(enumerate(self.examples), total=len(self.examples), desc="Prepare examples"):
            text = getattr(ex, self.field)

            if not self.tokenized:
                text = _tokenize(text)

            encoded_example = self.tokenizer(
                text,
                max_length=self.max_length if self.max_length > 0 else None,
                padding="longest",
                truncation=True,
            )

            if self.keep_entities:
                tokenized_text = self.tokenizer.decode(
                    encoded_example["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                entities_mask = _ner(tokenized_text)
                special_tokens = self.tokenizer.all_special_ids
                t = -1
                entities_subword_mask = [True] * len(encoded_example["input_ids"])
                for s, tok in enumerate(encoded_example["input_ids"]):
                    if tok not in special_tokens:
                        if _is_start_word(self.tokenizer, tok):
                            t += 1
                        if t >= len(entities_mask):
                            print(f"entities_mask = {len(entities_mask)}")
                            print(
                                f"{len(_tokenize(tokenized_text, as_string=False))} ---- {_tokenize(tokenized_text, as_string=False)}"
                            )
                            print(
                                f"{len(self.tokenizer.tokenize(tokenized_text))} ---- {self.tokenizer.tokenize(tokenized_text)}"
                            )
                        entities_subword_mask[s] = entities_mask[t]
            else:
                entities_subword_mask = None

            for j in range(self.num_augments):
                encoded_aug = dict(encoded_example)
                masked_input, _, pair_targets = self.mask_scheme.mask(
                    encoded_aug.pop("input_ids"), include_indices=entities_subword_mask
                )

                yield {
                    "pair_targets": pair_targets,
                    "index": i,
                    "aug_index": j,
                    "input_ids": masked_input,
                    **encoded_aug,
                }

    def __getitem__(self, index) -> Dict[str, TokenIdList]:
        return self._data[index]


@dataclass
class _Collator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        pair_targets = [f.pop("pair_targets", []) for f in features]
        indices = [f.pop("index", -1) for f in features]
        aug_indices = [f.pop("aug_index", -1) for f in features]
        batch = super().__call__(features)
        batch["pair_targets"] = pair_targets
        batch["indices"] = indices
        batch["aug_indices"] = aug_indices

        return batch


def prepare_json(example: InputExample, field: str, augmented_text: str, guid: int, src: str) -> Dict[str, Any]:
    if field == "text_a":
        return _to_json(augmented_text, example.text_b, f"aug-{guid}", src=src)
    else:
        return _to_json(example.text_a, augmented_text, f"aug-{guid}", src=src)


def generate(model, tokenizer, examples, field, num_augments, args, start_idx=0):
    mask_scheme = PairWithSpanMaskingScheme(
        tokenizer,
        mask_ratio=args.mask_ratio,
        span_lower=args.span_lower,
        span_upper=args.span_upper,
        geometric_p=args.geometric_p,
    )

    dataset = _Dataset(
        examples,
        tokenizer,
        mask_scheme,
        field,
        args.tokenize,
        args.max_seq_length,
        args.num_augments,
        args.keep_entities,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=_Collator(tokenizer, args.padding, args.max_seq_length, args.pad_to_multiple_of),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_name = getattr(model, "module", model).base_model_prefix

    num_augs_skipped = 0

    last_ex_idx = 0
    augmented_samples = []
    for batch_idx, batch in tqdm(
        enumerate(dataloader), desc=f"Enumerating over batches n={num_augments}", total=len(dataloader)
    ):
        pair_targets = batch.pop("pair_targets", None)
        ex_indices = batch.pop("indices", None)
        aug_indices = batch.pop("aug_indices", None)
        assert pair_targets is not None, f"{batch_idx}"

        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)

        for b, pairs in enumerate(pair_targets):
            ex_idx = ex_indices[b]
            ex = examples[ex_idx]

            if last_ex_idx != ex_idx:
                out_json = _to_ex_json(examples[last_ex_idx], last_ex_idx + 1)
                out_json["augmented_samples"] = augmented_samples
                yield out_json
                augmented_samples = []

            mlm_replaces = []
            for lb, ub in pairs:
                logits = output.logits[b, range(lb, ub), :]
                mlm_tokens = pick(logits, args.do_sample, args.temperature, args.topk, args.topp)

                for tok_idx in range(lb, ub):
                    sampled_tokens = mlm_tokens[tok_idx - lb]
                    if isinstance(sampled_tokens, int):
                        sampled_tokens = [sampled_tokens]

                    for i, mlm_token in enumerate(sampled_tokens):
                        if len(mlm_replaces) <= i:
                            mlm_replaces.append({tok_idx: mlm_token})
                        else:
                            mlm_replaces[i][tok_idx] = mlm_token

            example_ids = batch["input_ids"][b]

            for j, repl in enumerate(mlm_replaces):
                augmented_ids = example_ids.detach().cpu().tolist()

                for idx, tok in repl.items():
                    augmented_ids[idx] = tok

                mlm_text = tokenizer.decode(
                    augmented_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                if mlm_text.lower() == getattr(ex, args.field).lower():
                    num_augs_skipped += 1
                    continue

                augmented_samples.append(
                    prepare_json(
                        ex,
                        field,
                        mlm_text,
                        start_idx + aug_indices[b] + j + 1,
                        f"MLM-{model_name}-lo={args.span_lower}-up={args.span_upper}-mask={args.mask_ratio}",
                    )
                )

            last_ex_idx = ex_idx

    if augmented_samples:
        out_json = _to_ex_json(examples[last_ex_idx], last_ex_idx + 1)
        out_json["augmented_samples"] = augmented_samples
        yield out_json

    if num_augs_skipped > 0:
        logger.warning(f"Num of augs skipped: {num_augs_skipped}")


def pick(logits, do_sample: bool, temperature: float, topk: int, topp: float, num_samples: int = 1) -> TokenIdList:
    if do_sample:
        if temperature != 1.0:
            scores = logits / temperature
        else:
            scores = logits

        if 0 < topk:
            top_k = min(topk, scores.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
            scores[indices_to_remove] = -float("Inf")

        if 0 < topp <= 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > topp

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores[indices_to_remove] = -float("Inf")

        probs = F.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=num_samples).squeeze(1)
    else:
        next_tokens = torch.topk(logits, dim=-1, k=num_samples)[1]

    return next_tokens.detach().cpu().tolist()


def merge(augmented_data1, augmented_data2):
    for ex1, ex2 in zip(augmented_data1, augmented_data2):
        ex1["augmented_samples"].extend(ex2["augmented_samples"])
        yield ex1


def save(augmented_data: List[Dict[str, Any]], output_dir: Path):
    out_path = output_dir / "train.jsonl"

    with out_path.open("w", encoding="utf-8") as writer:
        for json_out in augmented_data:
            writer.write(json.dumps(json_out) + "\n")


def main():
    parser = ArgumentParser()

    parser.add_argument("--span_lower", default=1, type=int, help="lower bound on the number of words in a span")
    parser.add_argument("--span_upper", default=1, type=int, help="upper bound on the number of words in a span")
    parser.add_argument("--mask_ratio", default=0.15, type=float, help="proportion of words to be masked")
    parser.add_argument(
        "--random_token_prob",
        default=0.1,
        type=float,
        help="Probability of a selected token being replaced randomly from the vocabulary.",
    )
    parser.add_argument(
        "--leave_unmasked_prob", default=0.1, type=float, help="Probability of a selected token being left unchanged."
    )
    parser.add_argument(
        "--geometric_p",
        default=0.2,
        type=float,
        help="p for the geometric distribution used in span masking. -1 is uniform",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
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
        "--temperature",
        default=1.0,
        type=float,
        help="temperature to control randomness",
    )
    parser.add_argument(
        "--topk",
        default=0,
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
        "--max_seq_length",
        default=0,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--padding", default="longest", type=str, choices=("longest", "max_length"))
    parser.add_argument(
        "--pad_to_multiple_of",
        default=None,
        type=int,
        help="See https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/",
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
        choices=("text_a", "text_b"),
        type=str,
        help="Pivot field for augmentation",
    )
    parser.add_argument(
        "--tokenize",
        default=False,
        action="store_true",
        help="Whether to tokenize augmented example",
    )
    parser.add_argument(
        "--keep_entities",
        default=False,
        action="store_true",
        help="Whether to keep entities",
    )
    parser.add_argument("--num_workers", default=12, type=int, help="kwarg passed to DataLoader")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    processor = GLUEv2Processor(args.task)
    examples = processor.get_train_examples(args.data_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.config_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
        return_dict=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=True,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    args.device = torch.cuda.current_device()

    model = DataParallel(model)
    model.to(args.device)
    model.eval()

    if args.field == "both":
        augmented_data1 = generate(model, tokenizer, examples, "text_a", args.num_augments // 2, args)
        augmented_data2 = generate(
            model, tokenizer, examples, "text_b", args.num_augments // 2, args, args.num_augments // 2
        )
        augmented_data = list(merge(augmented_data1, augmented_data2))
    else:
        augmented_data = list(generate(model, tokenizer, examples, args.field, args.num_augments, args))

    save(augmented_data, output_dir)
    copy_dev_file(args.data_dir, output_dir)


if __name__ == "__main__":
    main()
