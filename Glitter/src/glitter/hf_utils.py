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
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput, MultipleChoiceModelOutput
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead


def _reinit_classifier_head(config: PretrainedConfig, model: PreTrainedModel):
    for module in model.classifier.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@dataclass
class HFSequenceClassificationMetadata:
    head_type: Union[Callable[[PretrainedConfig], nn.Module], nn.Module]
    init_fn: Callable[[PretrainedConfig, PreTrainedModel], None] = _reinit_classifier_head
    attr: str = "classifier"

    def reinitialize(self, config: PretrainedConfig, model: PreTrainedModel):
        setattr(model, self.attr, self.head_type(config))
        self.init_fn(config, model)


def get_model_inputs(
    model_type: str, input_ids, token_type_ids=None, attention_mask=None, labels=None
) -> Dict[str, Optional[torch.Tensor]]:
    inputs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    if labels is not None:
        inputs["labels"] = labels

    if model_type != "distilbert":
        inputs["token_type_ids"] = token_type_ids if model_type in ["bert", "xlnet", "albert"] else None

    return inputs


HF_SEQUENCE_CLASSIFICATION = {
    "electra": HFSequenceClassificationMetadata(ElectraClassificationHead),
    "roberta": HFSequenceClassificationMetadata(RobertaClassificationHead),
    "bert": HFSequenceClassificationMetadata(lambda config: nn.Linear(config.hidden_size, config.num_labels)),
    "albert": HFSequenceClassificationMetadata(lambda config: nn.Linear(config.hidden_size, config.num_labels)),
}


def get_last_layer_hidden_states(
    config: PretrainedConfig, model_output: Union[SequenceClassifierOutput, MultipleChoiceModelOutput]
) -> Optional[torch.Tensor]:
    if model_output.hidden_states is None:
        return None

    if config.model_type == "xlnet":
        cls_index = -1
    else:
        cls_index = 0

    return model_output.hidden_states[-1][:, cls_index, :]
