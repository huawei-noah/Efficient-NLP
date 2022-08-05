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

import numpy as np
import torch
from datasets import load_metric
from torchmetrics import Metric


class SquadMetric(Metric):
    def __init__(self, postprocess_func, version_2_with_negative: bool):
        super().__init__(compute_on_step=False)
        self.metric = load_metric("squad_v2" if version_2_with_negative else "squad")
        self.postprocess_func = postprocess_func
        self.add_state("start_logits", [])
        self.add_state("end_logits", [])
        self.add_state("example_ids", [])

    def update(self, example_ids: torch.Tensor, start_logits: torch.Tensor, end_logits: torch.Tensor):
        self.example_ids += example_ids
        self.start_logits += start_logits
        self.end_logits += end_logits

    @classmethod
    def _accumulate(cls, logits):
        max_length = max(l.shape[-1] for l in logits)
        return np.vstack(
            [np.pad(l.cpu().numpy(), (0, max_length - l.shape[-1]), constant_values=-100.0) for l in logits]
        )

    def compute(self):
        example_ids = [i.item() for i in self.example_ids]

        start_logits = self._accumulate(self.start_logits)
        end_logits = self._accumulate(self.end_logits)
        predictions = (start_logits, end_logits)
        predictions, references = self.postprocess_func(predictions=predictions, example_indices=example_ids)

        return self.metric.compute(predictions=predictions, references=references)
