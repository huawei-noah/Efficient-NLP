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
from dataclasses import dataclass
from torch_scatter import scatter_max
from typing import Optional

from .modeling_glitter import GlitterOutput


@dataclass
class ConsistencyGlitterOutput(GlitterOutput):
    orig_logits: Optional[torch.Tensor] = None


class GlitterForConsistency(nn.Module):
    def __init__(
        self,
        temperature: float,
        num_augments: int,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_augments = num_augments

    def forward(
        self,
        orig_logits: torch.Tensor,
        cand_logits: torch.Tensor,
        augment_rank: int,
        cand_mask,
        cand_ranks=None,
    ) -> ConsistencyGlitterOutput:

        aligned_ref_logits = orig_logits[cand_mask]

        if aligned_ref_logits.shape[-1] == 1 and cand_logits.shape[-1] == 1:
            #  We are doing regression
            loss_fct = nn.MSELoss(reduction="none")
            distances = loss_fct(aligned_ref_logits.view(-1), cand_logits.view(-1))
        else:
            loss_fct = nn.KLDivLoss(reduction="none")
            distances = loss_fct(
                F.log_softmax(cand_logits / self.temperature, dim=-1),
                F.softmax(aligned_ref_logits, dim=-1),
            ).sum(-1)

        selected_ranks = None

        if distances.nelement() > 0:
            _dists, _indices = None, None

            for j in range(augment_rank):
                _dists, _indices = scatter_max(distances, cand_mask)
                if j > 0:
                    _indices = _indices[(_dists != 0) & ~_dists.isinf()]
                    _dists = _dists[(_dists != 0) & ~_dists.isinf()]
                distances[_indices] = float("-inf")
            selected_indices = _indices
            if cand_ranks is not None:
                selected_ranks = cand_ranks[selected_indices]
        else:
            selected_indices = None

        return ConsistencyGlitterOutput(
            selected_indices,
            selected_ranks,
            aligned_ref_logits[selected_indices],
        )


class ConsistencyHead(nn.Module):
    """
    Impl. of "Unsupervised Data Augmentation for Consistency Training" (https://arxiv.org/abs/1904.12848)
    by Xie et al., NeurIPS 2020

    Official repo: https://github.com/google-research/uda

    PyTorch repo: https://github.com/SanghunYun/UDA_pytorch
    """

    def __init__(
        self,
        softmax_temp: float = 1.0,
        confidence_threshold: float = -1.0,
    ):
        super().__init__()
        self.softmax_temp = softmax_temp
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        ori_logits: torch.Tensor,
        aug_logits: torch.Tensor,
    ):
        ori_prob = F.softmax(ori_logits, dim=-1)

        # confidence-based masking
        if self.confidence_threshold != -1:
            unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > self.confidence_threshold
            unsup_loss_mask = unsup_loss_mask.type(torch.float32)
        else:
            unsup_loss_mask = torch.ones(ori_prob.shape[0], device=ori_prob.device, dtype=torch.float32)

        if ori_logits.shape[-1] == 1:
            unsup_fn = nn.MSELoss(reduction="none")
            unsup_loss = unsup_fn(ori_logits.view(-1), aug_logits.view(-1))
        else:
            unsup_fct = nn.KLDivLoss(reduction="none")
            aug_log_prob = F.log_softmax(aug_logits / self.softmax_temp, dim=-1)

            unsup_loss = torch.sum(unsup_fct(aug_log_prob, ori_prob), dim=-1)

        unsup_loss = (
            torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.clamp(torch.sum(unsup_loss_mask, dim=-1), min=1.0)
        )

        return unsup_loss
