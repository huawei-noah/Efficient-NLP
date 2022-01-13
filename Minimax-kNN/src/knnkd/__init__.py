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

from knnkd.version import VERSION as __version__

from .cls import run_glue
from .data.cached import cached_model_outputs, SavedModelOutput
from .hf_utils import HF_SEQUENCE_CLASSIFICATION, get_model_inputs as hf_get_model_inputs, get_last_layer_hidden_states
from .log_utils import set_global_logging_info, set_global_logging_warning, set_global_logging_error
from .modeling import KnnModelOutput, KDHead, KDHeadFast, MinimaxKnnHead, MinimaxKnnHeadFast, ReferenceL2Head
