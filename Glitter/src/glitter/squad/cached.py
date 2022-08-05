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
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.tokenization_utils_base import PaddingStrategy

from glitter.squad.processors import SquadExample, SquadFeaturesType
from .dataset import convert_to_features
from ..hf_utils import get_last_layer_hidden_states

logger = logging.getLogger(__name__)


@dataclass
class SavedModelOutput:
    start_logits: Union[np.array, torch.Tensor]
    end_logits: Union[np.array, torch.Tensor]
    hidden_states: Union[np.array, torch.Tensor]
    index_map: Sequence[Union[int, Sequence[int]]]
    reverse_index_map: Sequence[int]


class _Dataset(Dataset):
    def __init__(
        self,
        features: List[SquadFeaturesType],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        doc_stride: int,
    ):
        self.features = list(self._flatten(features))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride

    def _flatten(self, features):
        for f in features:
            if isinstance(f, (list, tuple)):
                yield from f
            else:
                yield f

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index) -> Dict[str, Any]:
        f = dict(self.features[index])
        f.pop("offset_mapping", None)
        f["augment_index"] = f.get("augment_index", -1) + 1
        return f


@dataclass
class _Collator:
    tokenizer: PreTrainedTokenizer
    num_augments: int = 0
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # flattened_features = []
        # for f in features:
        #     flattened_features.append(_asdict(f))
        # flattened_features.extend([_asdict(augf) for augf in f.augmented_features])
        padded_batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return padded_batch


def _map_to_indices(features) -> Tuple[Sequence[Union[int, Sequence[int]]], Sequence[int]]:
    index_map = []
    reverse_map = []
    idx = 0
    for i, f in enumerate(features):
        if isinstance(f, (list, tuple)):
            index_map.append([idx + s for s in range(len(f))])
            reverse_map.extend([i] * len(f))
            idx += len(f)
        else:
            index_map.append(idx)
            reverse_map.append(i)
            idx += 1

    return index_map, reverse_map


def cached_model_outputs(
    examples: List[SquadExample],
    model_name_or_path: str,
    batch_size: int,
    max_seq_length: int = 384,
    doc_stride: int = 128,
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
    overwrite_cache = kwargs.pop("overwrite_cache", False)

    config = AutoConfig.from_pretrained(
        config_name or model_name_or_path,
        cache_dir=cache_dir,
        return_dict=True,
        output_hidden_states=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
    )

    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    output_dir = Path(output_dir) / ".output_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_prefix = f"_{dataset_name}" if dataset_name else ""
    prefix = f"{config.model_type}{dataset_prefix}_seq{max_seq_length}_stride{doc_stride}_train"

    logits_file = output_dir / f"{prefix}_logits.npz"
    hidden_states_file = output_dir / f"{prefix}_hidden_states.npz"
    indices_file = output_dir / f"{prefix}_indices.pkl"

    if overwrite_cache or (not logits_file.exists() or not hidden_states_file.exists() or not indices_file.exists()):
        features = list(
            convert_to_features(
                examples, tokenizer, fold="train", max_length=max_seq_length, doc_stride=doc_stride, num_augments=-1
            )
        )

        index_map, reverse_index_map = _map_to_indices(features)

        dataset = _Dataset(
            features,
            tokenizer,
            max_seq_length,
            doc_stride,
        )

        collator = _Collator(tokenizer, max_length=max_seq_length)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers, pin_memory=True
        )

        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir,
        )

        model = DataParallel(model)
        model.to(device)

        all_start_logits = np.full((len(dataset), max_seq_length), -100.0, dtype=np.float32)
        all_end_logits = np.full((len(dataset), max_seq_length), -100.0, dtype=np.float32)
        all_hidden_states = np.zeros((len(dataset), config.hidden_size), dtype=np.float32)

        idx = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc="Caching model output"):
            bsz = batch["input_ids"].shape[0]
            batch = {k: v.to(device) for k, v in batch.items()}

            batch.pop("example_index", None)
            batch.pop("augment_index", None)

            model.eval()
            with torch.no_grad():
                output = model(**batch)

            batch_start_logits = output.start_logits.cpu().numpy()
            batch_end_logits = output.end_logits.cpu().numpy()
            last_hidden_states = get_last_layer_hidden_states(config, output)
            batch_hidden_states = last_hidden_states.cpu().numpy()

            max_batch_len = batch_start_logits.shape[-1]

            all_start_logits[idx : idx + bsz, :max_batch_len] = batch_start_logits
            all_end_logits[idx : idx + bsz, :max_batch_len] = batch_end_logits
            all_hidden_states[idx : idx + bsz, :] = batch_hidden_states

            idx += bsz

        np.savez(
            str(logits_file),
            start_logits=all_start_logits,
            end_logits=all_end_logits,
        )

        np.savez(
            str(hidden_states_file),
            hidden_states=all_hidden_states,
        )

        with indices_file.open("wb") as f:
            pickle.dump({"index": index_map, "reverse_index": reverse_index_map}, f)
    else:
        cached_logits = np.load(str(logits_file))
        all_start_logits = cached_logits["start_logits"]
        all_end_logits = cached_logits["end_logits"]

        cached_hidden_states = np.load(str(hidden_states_file))
        all_hidden_states = cached_hidden_states["hidden_states"]

        with indices_file.open("rb") as f:
            p = pickle.load(f)
        index_map = p["index"]
        reverse_index_map = p["reverse_index"]

    if return_tensor:
        all_start_logits = torch.from_numpy(all_start_logits)
        all_end_logits = torch.from_numpy(all_end_logits)
        all_hidden_states = torch.from_numpy(all_hidden_states)

    return SavedModelOutput(all_start_logits, all_end_logits, all_hidden_states, index_map, reverse_index_map)
