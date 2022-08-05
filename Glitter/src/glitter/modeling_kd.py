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

import torch
import torch.nn.functional as F
from torch import nn


class KDForSequenceClassification(nn.Module):
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
        weights=None,
    ):
        if teacher_logits is None:
            assert example_indices is not None
            if augmented_indices is None:
                augmented_indices = torch.zeros_like(example_indices)
            teacher_logits = self.teacher_logits[example_indices, augmented_indices]

        assert teacher_logits.size() == student_logits.size()

        if weights is not None:
            loss_fct = nn.KLDivLoss(reduction="none")
            loss_ce = (
                loss_fct(
                    F.log_softmax(student_logits / self.temperature, dim=-1),
                    F.softmax(teacher_logits / self.temperature, dim=-1),
                ).sum(-1)
                * (self.temperature ** 2)
            )
            weighted_loss = loss_ce * weights
            loss_ce = torch.sum(weighted_loss)
        elif teacher_logits.shape[-1] == 1 and student_logits.shape[-1] == 1:
            #  We are doing regression
            loss_fct = nn.MSELoss()
            loss_ce = loss_fct(teacher_logits.view(-1), student_logits.view(-1))
        else:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            loss_ce = (
                loss_fct(
                    F.log_softmax(student_logits / self.temperature, dim=-1),
                    F.softmax(teacher_logits / self.temperature, dim=-1),
                )
                * (self.temperature ** 2)
            )
        return loss_ce


class KDForQuestionAnswering(nn.Module):
    def __init__(
        self,
        teacher_start_logits: torch.Tensor,
        teacher_end_logits: torch.Tensor,
        temperature: float,
    ):
        super().__init__()
        self.register_buffer("teacher_start_logits", teacher_start_logits)
        self.register_buffer("teacher_end_logits", teacher_end_logits)
        self.temperature = temperature

    def _loss(self, loss_fn, tea_logits, stu_logits):
        return (
                loss_fn(
                    F.log_softmax(stu_logits / self.temperature, dim=-1),
                    F.softmax(tea_logits / self.temperature, dim=-1),
                ).sum(-1)
                * (self.temperature ** 2)
        )

    def forward(
        self,
        student_start_logits,
        student_end_logits,
        indices=None,
        teacher_start_logits=None,
        teacher_end_logits=None,
        weights=None,
    ):
        if teacher_start_logits is None or teacher_end_logits is None:
            assert indices is not None

        if teacher_start_logits is None:
            teacher_start_logits = self.teacher_start_logits[indices, :student_start_logits.shape[-1]]

        if teacher_end_logits is None:
            teacher_end_logits = self.teacher_end_logits[indices, :student_end_logits.shape[-1]]

        if weights is not None:
            loss_fct = nn.KLDivLoss(reduction="none")
            start_loss = self._loss(loss_fct, teacher_start_logits, student_start_logits)
            end_loss = self._loss(loss_fct, teacher_end_logits, student_end_logits)
            loss_ce = torch.sum((start_loss + end_loss) * weights / 2)
        else:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            start_loss = self._loss(loss_fct, teacher_start_logits, student_start_logits)
            end_loss = self._loss(loss_fct, teacher_end_logits, student_end_logits)
            loss_ce = (start_loss + end_loss) / 2

        return loss_ce


class KDForMultipleChoice(nn.Module):
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
            weights=None,
    ):
        if teacher_logits is None:
            assert example_indices is not None
            if augmented_indices is None:
                augmented_indices = torch.zeros_like(example_indices)
            teacher_logits = self.teacher_logits[example_indices, augmented_indices, :student_logits.shape[-1]]

        if weights is not None:
            loss_fct = nn.KLDivLoss(reduction="none")
            loss_ce = (
                    loss_fct(
                        F.log_softmax(student_logits / self.temperature, dim=-1),
                        F.softmax(teacher_logits / self.temperature, dim=-1),
                    ).sum(-1)
                    * (self.temperature ** 2)
            )
            weighted_loss = loss_ce * weights
            loss_ce = torch.sum(weighted_loss)
        else:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            loss_ce = (
                    loss_fct(
                        F.log_softmax(student_logits / self.temperature, dim=-1),
                        F.softmax(teacher_logits / self.temperature, dim=-1),
                    )
                    * (self.temperature ** 2)
            )

        return loss_ce
