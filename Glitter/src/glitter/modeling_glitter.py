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

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_max


@dataclass
class GlitterOutput:
    selected_neighbors: Optional[torch.LongTensor]
    selected_ranks: Optional[torch.LongTensor] = None

    @property
    def is_empty(self):
        return self.selected_neighbors is None or self.selected_neighbors.nelement() == 0

    @property
    def has_selected_ranks(self):
        return self.selected_ranks is not None and self.selected_ranks.nelement() > 0


@dataclass
class SequenceClassifierGlitterOutput(GlitterOutput):
    teacher_logits: Optional[torch.Tensor] = None


@dataclass
class QuestionAnsweringGlitterOutput(GlitterOutput):
    teacher_start_logits: Optional[torch.Tensor] = None
    teacher_end_logits: Optional[torch.Tensor] = None


class GlitterForSequenceClassification(nn.Module):
    def __init__(
        self,
        teacher_logits: torch.Tensor,
        temperature: float,
        num_augments: int,
    ):
        super().__init__()
        self.register_buffer("teacher_logits", teacher_logits[:, 1:])
        self.temperature = temperature
        self.num_augments = num_augments

    def forward(
        self,
        stu_logits,
        augment_rank: int,
        nn_mask,
        example_indices,
        augmented_indices=None,
        nn_ranks=None,
    ) -> SequenceClassifierGlitterOutput:
        assert example_indices is not None
        tea_logits = self.teacher_logits[example_indices[nn_mask], augmented_indices, :]

        if tea_logits.shape[-1] == 1 and stu_logits.shape[-1] == 1:
            #  We are doing regression
            loss_fct = nn.MSELoss(reduction="none")
            distances = loss_fct(tea_logits.view(-1), stu_logits.view(-1))
        else:
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
                distances[_indices] = float("-inf")
            selected_indices = _indices
            if nn_ranks is not None:
                selected_ranks = nn_ranks[selected_indices]
        else:
            selected_indices = None

        return SequenceClassifierGlitterOutput(
            selected_indices,
            selected_ranks,
            teacher_logits=tea_logits[selected_indices],
        )


class GlitterForQuestionAnswering(nn.Module):
    def __init__(
        self,
        teacher_start_logits: torch.Tensor,
        teacher_end_logits: torch.Tensor,
        temperature: float,
        num_augments: int,
    ):
        super().__init__()
        self.register_buffer("teacher_start_logits", teacher_start_logits)
        self.register_buffer("teacher_end_logits", teacher_end_logits)
        self.temperature = temperature
        self.num_augments = num_augments

    def forward(
        self,
        stu_start_logits,
        stu_end_logits,
        augment_rank: int,
        nn_mask,
        indices,
        nn_ranks=None,
    ) -> QuestionAnsweringGlitterOutput:
        assert indices is not None

        tea_start_logits = self.teacher_start_logits[
            indices[nn_mask], : stu_start_logits.shape[-1]
        ]
        tea_end_logits = self.teacher_end_logits[
            indices[nn_mask], : stu_end_logits.shape[-1]
        ]

        loss_fct = nn.KLDivLoss(reduction="none")
        start_distances = loss_fct(
            F.log_softmax(stu_start_logits / self.temperature, dim=-1),
            F.softmax(tea_start_logits / self.temperature, dim=-1),
        ).sum(-1)
        end_distances = loss_fct(
            F.log_softmax(stu_end_logits / self.temperature, dim=-1),
            F.softmax(tea_end_logits / self.temperature, dim=-1),
        ).sum(-1)

        distances = (start_distances + end_distances) / 2

        selected_ranks = None

        if distances.nelement() > 0:
            _dists, _indices = None, None

            for j in range(augment_rank):
                _dists, _indices = scatter_max(distances, nn_mask)
                if j > 0:
                    _indices = _indices[(_dists != 0) & ~_dists.isinf()]
                    _dists = _dists[(_dists != 0) & ~_dists.isinf()]
                distances[_indices] = float("-inf")
            selected_indices = _indices
            if nn_ranks is not None:
                selected_ranks = nn_ranks[selected_indices]
        else:
            selected_indices = None

        return QuestionAnsweringGlitterOutput(
            selected_indices,
            selected_ranks,
            teacher_start_logits=tea_start_logits[selected_indices],
            teacher_end_logits=tea_end_logits[selected_indices],
        )


class GlitterForMultipleChoice(GlitterForSequenceClassification):
    def __init__(
        self,
        teacher_logits: torch.Tensor,
        temperature: float,
        num_augments: int,
    ):
        super().__init__(teacher_logits, temperature, num_augments)


class RandomGlitterForSequenceClassification(nn.Module):
    def __init__(
        self,
        teacher_logits: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("teacher_logits", teacher_logits[:, 1:])

    def forward(
        self,
        stu_logits,
        augment_rank: int,
        nn_mask,
        example_indices,
        augmented_indices=None,
        nn_ranks=None,
    ):
        assert example_indices is not None

        tea_logits = self.teacher_logits[example_indices[nn_mask], augmented_indices, :]
        _dists, _indices = None, None
        unq_mask = torch.unique(nn_mask)
        self_mask = nn_mask.unsqueeze(0) == unq_mask.unsqueeze(1)
        selected_indices = torch.multinomial(self_mask.float(), num_samples=1).squeeze(1)

        if nn_ranks is not None:
            selected_ranks = nn_ranks[selected_indices]
        else:
            selected_indices = None

        return SequenceClassifierGlitterOutput(
            selected_indices,
            selected_ranks,
            teacher_logits=tea_logits[selected_indices],
        )