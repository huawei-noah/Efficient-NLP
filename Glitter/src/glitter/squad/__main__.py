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

import argparse
import os

from glitter.lightning_base import add_generic_args
from glitter.squad import run as run_squad

os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    parser = run_squad.GlitterSQuADTransformer.add_model_specific_args(parser)
    args = parser.parse_args()

    run_squad.main(args)
