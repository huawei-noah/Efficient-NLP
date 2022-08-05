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

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_max
from transformers import PreTrainedModel

from .hf_utils import get_last_layer_hidden_states


@dataclass
class GlitterOutput:
    selected_neighbors: Optional[torch.LongTensor]
    selected_distances: Optional[torch.FloatTensor]
    selected_ranks: Optional[torch.LongTensor] = None
    max_filtered_ranks: Optional[torch.LongTensor] = None
    min_filtered_ranks: Optional[torch.LongTensor] = None
    weights: Optional[torch.FloatTensor] = None

    @property
    def is_empty(self):
        return self.selected_neighbors is None or self.selected_neighbors.nelement() == 0

    @property
    def has_selected_ranks(self):
        return self.selected_ranks is not None and self.selected_ranks.nelement() > 0

    @property
    def has_selected_distances(self):
        return self.selected_distances is not None and self.selected_distances.nelement() > 0

    @property
    def has_weights(self):
        return self.weights is not None


@dataclass
class SequenceClassifierGlitterOutput(GlitterOutput):
    teacher_logits: Optional[torch.Tensor] = None


@dataclass
class QuestionAnsweringGlitterOutput(GlitterOutput):
    teacher_start_logits: Optional[torch.Tensor] = None
    teacher_end_logits: Optional[torch.Tensor] = None


class KDHead(nn.Module):
    def __init__(
        self,
        teacher_logits: torch.Tensor,
        temperature: float,
    ):
        super().__init__()
        self.register_buffer("teacher_logits", teacher_logits)
        self.temperature = temperature

    def forward(
        self,
        student_logits,
        teacher_logits=None,
        example_indices=None,
        augmented_indices=None,
    ):
        if teacher_logits is None:
            assert example_indices is not None
            if augmented_indices is None:
                augmented_indices = torch.zeros_like(example_indices)
            teacher_logits = self.teacher_logits[example_indices, augmented_indices]

        assert teacher_logits.size() == student_logits.size()

        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_ce = (
            loss_fct(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
            )
            * (self.temperature ** 2)
        )

        return loss_ce


class GlitterHead(nn.Module):
    def __init__(
        self,
        teacher_logits: torch.Tensor,
        student: PreTrainedModel,
        temperature: float,
    ):
        super().__init__()
        self.register_buffer("teacher_logits", teacher_logits[:, 1:])
        self.student = student
        self.temperature = temperature

    def _detach(self, model: PreTrainedModel) -> PreTrainedModel:
        model_class = type(model)
        detached_model = model_class(model.config)
        detached_model.load_state_dict(model.state_dict())

        for p in detached_model.parameters():
            p.requires_grad = False

        return detached_model

    def forward(
        self,
        augment_rank: int,
        nn_mask,
        example_indices,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        augmented_indices=None,
        nn_ranks=None,
    ):
        assert example_indices is not None

        inputs = dict(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        with torch.no_grad():
            stu_output = self.student(**inputs)

        stu_logits = stu_output.logits
        tea_logits = self.teacher_logits[example_indices[nn_mask], augmented_indices, :]

        loss_fct = nn.KLDivLoss(reduction="none")
        distances = loss_fct(
            F.log_softmax(stu_logits / self.temperature, dim=-1),
            F.softmax(tea_logits / self.temperature, dim=-1),
        ).sum(-1)

        selected_ranks = None
        if distances.nelement() > 0:
            _dists, _indices = None, None

            for j in range(augment_rank):
                _dists, _indices = scatter_max(distances, nn_mask)
                if j > 0:
                    _indices = _indices[(_dists != 0) & ~_dists.isinf()]
                    _dists = _dists[(_dists != 0) & ~_dists.isinf()]

                distances[_indices] = float('-inf')

            selected_indices = _indices
            if nn_ranks is not None:
                selected_ranks = nn_ranks[selected_indices]
        else:
            selected_indices = None

        return GlitterOutput(
            tea_logits[selected_indices],
            selected_indices,
            None,
            selected_ranks,
            None,
            None,
        )
