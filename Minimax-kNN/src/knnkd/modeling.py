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
class KnnModelOutput:
    teacher_logits: torch.Tensor
    selected_neighbors: Optional[torch.LongTensor]
    selected_distances: Optional[torch.FloatTensor]
    selected_ranks: Optional[torch.LongTensor]
    max_filtered_ranks: Optional[torch.LongTensor]
    min_filtered_ranks: Optional[torch.LongTensor]

    @property
    def is_empty(self):
        return self.selected_neighbors is None or self.selected_neighbors.nelement() == 0

    @property
    def has_selected_ranks(self):
        return self.selected_ranks is not None and self.selected_ranks.nelement() > 0

    @property
    def has_selected_distances(self):
        return self.selected_distances is not None and self.selected_distances.nelement() > 0


class KDHead(nn.Module):
    def __init__(
        self,
        teacher: PreTrainedModel,
        temperature: float,
    ):
        super().__init__()
        self.teacher = teacher
        self.temperature = temperature

    def forward(
        self,
        student_logits,
        teacher_logits=None,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        output_hidden_states: bool = None,
    ):
        teacher_hidden_states = None
        if teacher_logits is None:
            with torch.no_grad():
                teacher_output = self.teacher(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                )

            teacher_logits = teacher_output.logits
            teacher_hidden_states = get_last_layer_hidden_states(self.teacher.config, teacher_output)

        assert teacher_logits.size() == student_logits.size()

        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_ce = (
            loss_fct(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
            )
            * (self.temperature ** 2)
        )

        return loss_ce, teacher_hidden_states


class KDHeadFast(nn.Module):
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


class ReferenceL2Head(nn.Module):
    def __init__(self, ref_dim: int, teacher_dim: int, student_dim: int, loss_type: str = "MSE"):
        super().__init__()
        self.teacher_proj = nn.Linear(ref_dim, teacher_dim)
        self.teacher_dim = teacher_dim

        self.student_proj = nn.Linear(ref_dim, student_dim)
        self.student_dim = student_dim

        if loss_type in ("MAE", "L1"):
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        self.pdist = nn.PairwiseDistance()

    def forward(self, ref_hidden_states, teacher_hidden_states, student_hidden_states):
        d1 = self.pdist(self.teacher_proj(ref_hidden_states), teacher_hidden_states) / np.sqrt(self.teacher_dim)
        d2 = self.pdist(self.student_proj(ref_hidden_states), student_hidden_states) / np.sqrt(self.student_dim)

        return self.loss_fn(d1, d2)


class MinimaxKnnHead(nn.Module):
    def __init__(
        self,
        teacher: PreTrainedModel,
        student: PreTrainedModel,
        temperature: float,
        min_distance: float = 0.0,
        max_distance: float = 0.0,
        maxim_func: str = "ce",
    ):
        super().__init__()
        self.teacher = teacher
        self.student = self._detach(student)

        if self.teacher.config.hidden_size != self.student.config.hidden_size:
            self.teacher_proj = nn.Linear(self.teacher.config.hidden_size, self.student.config.hidden_size)
        else:
            self.teacher_proj = lambda t: t

        self.temperature = temperature
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.maxim_func = maxim_func

    def _detach(self, model: PreTrainedModel) -> PreTrainedModel:
        model_class = type(model)
        detached_model = model_class(model.config)
        detached_model.load_state_dict(model.state_dict())
        return detached_model

    def forward(
        self,
        augment_rank: int,
        nn_mask,
        tea_orig_hidden_states,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        nn_ranks=None,
    ):
        inputs = dict(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        l2_enabled = self.maxim_func == "l2"

        with torch.no_grad():
            stu_output = self.student(**inputs, output_hidden_states=l2_enabled)
            tea_output = self.teacher(**inputs, output_hidden_states=True)

        stu_logits = stu_output.logits
        tea_logits = tea_output.logits
        tea_hidden_states = get_last_layer_hidden_states(self.teacher.config, tea_output)
        if l2_enabled:
            stu_hidden_states = get_last_layer_hidden_states(self.student.config, stu_output)

        cos = nn.CosineSimilarity(dim=-1)
        tea_orig_hidden_states = tea_orig_hidden_states[nn_mask]
        nn_distances = torch.acos(cos(tea_orig_hidden_states, tea_hidden_states)) / np.pi

        min_filtered_ranks, max_filtered_ranks = None, None
        if self.max_distance > 0 or self.min_distance > 0:
            if nn_ranks is not None:
                if self.max_distance > 0:
                    max_filtered_ranks = nn_ranks[nn_distances > self.max_distance]
                if self.min_distance > 0:
                    min_filtered_ranks = nn_ranks[nn_distances < self.min_distance]

            if self.max_distance > 0:
                nn_mask = nn_mask[nn_distances <= self.max_distance]
                stu_logits = stu_logits[nn_distances <= self.max_distance, :]
                tea_logits = tea_logits[nn_distances <= self.max_distance, :]
                if l2_enabled:
                    stu_hidden_states = stu_hidden_states[nn_distances <= self.max_distance, :]
                    tea_hidden_states = tea_hidden_states[nn_distances <= self.max_distance, :]
                nn_distances = nn_distances[nn_distances <= self.max_distance]

            if self.min_distance > 0:
                nn_mask = nn_mask[nn_distances >= self.min_distance]
                stu_logits = stu_logits[nn_distances >= self.min_distance, :]
                tea_logits = tea_logits[nn_distances >= self.min_distance, :]
                if l2_enabled:
                    stu_hidden_states = stu_hidden_states[nn_distances >= self.min_distance, :]
                    tea_hidden_states = tea_hidden_states[nn_distances >= self.min_distance, :]
                nn_distances = nn_distances[nn_distances >= self.min_distance]

        if l2_enabled:
            pdist = nn.PairwiseDistance()
            distances = pdist(self.teacher_proj(tea_hidden_states), stu_hidden_states) / np.sqrt(
                self.student.config.hidden_size
            )
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
                if j > 0 or self.max_distance > 0 or self.min_distance > 0:
                    _indices = _indices[(_dists != 0) & ~_dists.isinf()]
                    _dists = _dists[(_dists != 0) & ~_dists.isinf()]

                distances[_indices] = float('-inf')

            selected_indices = _indices
            if nn_ranks is not None:
                selected_ranks = nn_ranks[selected_indices]

            selected_dists = nn_distances[selected_indices]
        else:
            selected_indices = None
            selected_dists = None

        # if nn_rank is not None:
        #     rank_counts = torch.zeros((num_augments,), dtype=torch.long).scatter_add_(
        #         nn_rank[selected_indices], torch.ones_like(selected_indices)
        #     )
        # else:
        #     rank_counts = None

        return KnnModelOutput(
            tea_logits[selected_indices],
            selected_indices,
            selected_dists,
            selected_ranks,
            max_filtered_ranks,
            min_filtered_ranks,
        )


class MinimaxKnnHeadFast(nn.Module):
    def __init__(
        self,
        teacher_logits: torch.Tensor,
        student: PreTrainedModel,
        temperature: float,
    ):
        super().__init__()
        self.register_buffer("teacher_logits", teacher_logits[:, 1:])
        self.student = student
        # self.student = self._detach(student)
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

        # if nn_rank is not None:
        #     rank_counts = torch.zeros((num_augments,), dtype=torch.long).scatter_add_(
        #         nn_rank[selected_indices], torch.ones_like(selected_indices)
        #     )
        # else:
        #     rank_counts = None

        return KnnModelOutput(
            tea_logits[selected_indices],
            selected_indices,
            None,
            selected_ranks,
            None,
            None,
        )
