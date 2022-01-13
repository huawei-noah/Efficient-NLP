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

import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    InputExample,
    InputFeatures,
    PreTrainedTokenizer,
)
from transformers.tokenization_utils_base import PaddingStrategy

from .glue import GLUEv2Processor, glue_tasks_num_labels
from .utils import AugmentedInputExample, InputFeaturesV2
from ..hf_utils import get_last_layer_hidden_states

logger = logging.getLogger(__name__)


@dataclass
class SavedModelOutput:
    logits: Union[np.memmap, torch.Tensor]
    hidden_states: Union[np.memmap, torch.Tensor]


class _Dataset(Dataset):
    def __init__(
        self,
        examples: Union[List[InputExample], List[AugmentedInputExample]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        max_augment_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_augment_length = max_augment_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> InputFeaturesV2:
        ex = self.examples[index]
        encoded_example = self.tokenizer(
            ex.text_a,
            ex.text_b,
            max_length=self.max_length if self.max_length > 0 else None,
            padding="longest",
            truncation=True,
        )

        augmented_features = []
        if hasattr(ex, "augmented_examples"):
            for aug_ex in ex.augmented_examples:
                encoded_aug = self.tokenizer(
                    aug_ex.text_a,
                    aug_ex.text_b,
                    max_length=self.max_augment_length if self.max_augment_length > 0 else None,
                    padding="longest",
                    truncation=True,
                )
                augmented_features.append(InputFeatures(**encoded_aug))

        return InputFeaturesV2(
            **encoded_example,
            label=None,
            augmented_features=augmented_features,
        )


def _asdict(f: Union[InputFeaturesV2, InputFeatures]):
    return {
        k: v
        for k, v in asdict(f).items()
        if k not in ("augmented_features", "example_index", "augmented_indices", "augmented_rank") and v is not None
    }


@dataclass
class _Collator:
    tokenizer: PreTrainedTokenizer
    num_augments: int = 0
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[InputFeaturesV2]) -> Dict[str, torch.Tensor]:
        flattened_features = []

        for f in features:
            flattened_features.append(_asdict(f))
            flattened_features.extend([_asdict(augf) for augf in f.augmented_features])

        flattened_batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_length = flattened_batch["input_ids"].shape[-1]

        batch = {}
        for k in flattened_batch.keys():
            batch[k] = flattened_batch[k].new_zeros((len(features), self.num_augments + 1, max_length))

        idx = 0
        for i, f in enumerate(features):
            num_augs = len(f.augmented_features) + 1

            for k in flattened_batch.keys():
                batch[k][i, :num_augs, :] = flattened_batch[k][idx : idx + num_augs, :]
            idx += num_augs

        return batch


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

    max_augment_length = kwargs.pop("max_augment_length", max_seq_length)

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

    processor = GLUEv2Processor(
        task,
        num_augments=-1,
    )

    examples = processor.get_train_examples(data_dir)
    num_augments = max(len(ex.augmented_examples) for ex in examples)
    dataset = _Dataset(
        examples,
        tokenizer,
        max_seq_length,
        max_augment_length,
    )

    num_labels = glue_tasks_num_labels[task]

    output_dir = Path(output_dir) / ".output_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_prefix = f"_{dataset_name}" if dataset_name else ""
    aug_prefix = f"_augseq{max_augment_length}" if max_augment_length != max_seq_length else ""
    prefix = f"{config.model_type}_{task}{dataset_prefix}_seq{max_seq_length}{aug_prefix}_train"

    logits_file = output_dir / f"{prefix}_logits.npy"
    hidden_states_file = output_dir / f"{prefix}_hidden_states.npy"

    if not logits_file.exists() or not hidden_states_file.exists():
        collator = _Collator(tokenizer, num_augments)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers, pin_memory=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
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
            shape=(len(dataset), num_augments + 1, config.hidden_size),
        )

        idx = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc="Caching model output"):
            bsz = batch["input_ids"].shape[0]
            batch = {k: v.view(-1, v.shape[-1]).to(device) for k, v in batch.items()}

            model.eval()
            with torch.no_grad():
                output = model(**batch)
            logits[idx : idx + bsz, :, :] = output.logits.reshape(bsz, -1, output.logits.shape[-1]).cpu().numpy()
            last_hidden_state = get_last_layer_hidden_states(config, output)
            hidden_states[idx : idx + bsz, :, :] = (
                last_hidden_state.reshape(bsz, -1, last_hidden_state.shape[-1]).cpu().numpy()
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
