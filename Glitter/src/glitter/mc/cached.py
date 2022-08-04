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

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.tokenization_utils_base import PaddingStrategy

from .dataset import MultiChoiceFeatures
from glitter.mc.processors import AugmentedMultiChoice, MultiChoiceExample, mc_processors
from ..hf_utils import get_last_layer_hidden_states

logger = logging.getLogger(__name__)


@dataclass
class SavedModelOutput:
    logits: Union[np.memmap, torch.Tensor]
    hidden_states: Union[np.memmap, torch.Tensor]


class _Dataset(Dataset):
    def __init__(
        self,
        examples: Sequence[MultiChoiceExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def _prepare(self, ex: MultiChoiceExample, aug_ex: Optional[AugmentedMultiChoice] = None):
        first_sentences = [(ex if aug_ex is None else aug_ex).question] * ex.num_choices
        second_sentences = ex.choices

        tokenized_example = self.tokenizer(
            first_sentences,
            second_sentences,
            max_length=self.max_length if self.max_length > 0 else None,
            padding="longest",
            truncation=True,
        )

        return tokenized_example

    def __getitem__(self, index) -> MultiChoiceFeatures:
        ex = self.examples[index]
        tokenized_ex = self._prepare(ex)
        augmented_features = [self._prepare(ex, aug_ex) for aug_ex in ex.augmented_examples]

        return MultiChoiceFeatures(
            tokenized_ex,
            index,
            augmented_features=augmented_features,
        )


@dataclass
class _Collator:
    tokenizer: PreTrainedTokenizer
    num_augments: int = 0
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, batch: List[MultiChoiceFeatures]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        num_choices = len(batch[0].features["input_ids"])

        flattened_features = []

        for item in batch:
            flattened_features.extend([{k: v[i] for k, v in item.features.items()} for i in range(num_choices)])
            for augf in item.augmented_features:
                flattened_features.extend([{k: v[i] for k, v in augf.items()} for i in range(num_choices)])

        padded_batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        padded_batch = {k: v.view(-1, num_choices, v.shape[-1]) for k, v in padded_batch.items()}

        max_length = padded_batch["input_ids"].shape[-1]

        new_batch = {}
        for k in padded_batch.keys():
            new_batch[k] = padded_batch[k].new_zeros((len(batch), self.num_augments + 1, num_choices, max_length))

        idx = 0
        for i, f in enumerate(batch):
            num_augs = len(f.augmented_features) + 1

            for k in new_batch.keys():
                new_batch[k][i, :num_augs, :, :] = padded_batch[k][idx : idx + num_augs, ...]
            idx += num_augs

        return new_batch


def cached_model_outputs(
    model_name_or_path: str,
    task: str,
    data_dir: str,
    batch_size: int,
    max_seq_length: int,
    return_tensor: bool = True,
    **kwargs,
) -> SavedModelOutput:
    cache_dir = kwargs.pop("cache_dir", None)
    config_name = kwargs.pop("config_name", None)
    tokenizer_name = kwargs.pop("tokenizer_name", None)
    dataset_name = kwargs.pop("dataset_name", None)
    device = kwargs.pop("device", torch.cuda.current_device())
    num_workers = kwargs.pop("num_workers", 8)
    output_dir = kwargs.pop("output_dir", os.path.dirname(os.path.abspath(model_name_or_path)))

    config = AutoConfig.from_pretrained(
        config_name or model_name_or_path,
        cache_dir=cache_dir,
        return_dict=True,
        output_hidden_states=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name_or_path,
        cache_dir=cache_dir,
    )

    processor = mc_processors[task]()

    examples = processor.get_train_examples(data_dir)
    num_augments = max(len(ex.augmented_examples) for ex in examples)
    num_choices = max(ex.num_choices for ex in examples)
    dataset = _Dataset(
        examples,
        tokenizer,
        max_seq_length,
    )

    num_labels = processor.num_labels

    output_dir = Path(output_dir) / ".output_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_prefix = f"_{dataset_name}" if dataset_name else ""
    prefix = f"{config.model_type}_{task}{dataset_prefix}_seq{max_seq_length}_train"

    logits_file = output_dir / f"{prefix}_logits.npy"
    hidden_states_file = output_dir / f"{prefix}_hidden_states.npy"

    if not logits_file.exists() or not hidden_states_file.exists():
        collator = _Collator(tokenizer, num_augments)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers, pin_memory=True
        )

        model = AutoModelForMultipleChoice.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir,
        )

        model = DataParallel(model)
        model.to(device)

        logits = np.memmap(
            str(logits_file),
            dtype=np.float32,
            mode="w+",
            shape=(len(dataset), num_augments + 1, num_labels),
        )

        hidden_states = np.memmap(
            str(hidden_states_file),
            dtype=np.float32,
            mode="w+",
            shape=(len(dataset), num_augments + 1, num_choices, config.hidden_size),
        )

        idx = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc="Caching model output"):
            # bsz = batch["input_ids"].shape[0]
            bsz, _, nch, _ = batch["input_ids"].shape

            batch = {k: v.view(-1, nch, v.shape[-1]).to(device) for k, v in batch.items()}

            model.eval()
            with torch.no_grad():
                output = model(**batch)
            logits[idx : idx + bsz, :, :] = output.logits.reshape(bsz, -1, output.logits.shape[-1]).cpu().numpy()
            last_hidden_state = get_last_layer_hidden_states(config, output)
            hidden_states[idx : idx + bsz, ...] = (
                last_hidden_state.reshape(bsz, -1, nch, last_hidden_state.shape[-1]).cpu().numpy()
            )

            idx += bsz

    logits = np.memmap(
        str(logits_file), mode="r", dtype=np.float32, shape=(len(dataset), num_augments + 1, num_labels)
    )

    hidden_states = np.memmap(
        str(hidden_states_file), mode="r", dtype=np.float32, shape=(len(dataset), num_augments + 1, config.hidden_size)
    )

    if return_tensor:
        logits = torch.from_numpy(np.array(logits))
        hidden_states = torch.from_numpy(np.array(hidden_states))

    return SavedModelOutput(logits, hidden_states)
