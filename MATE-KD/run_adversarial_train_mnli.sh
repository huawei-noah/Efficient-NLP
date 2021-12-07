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


# BERT version
export CUDA_VISIBLE_DEVICES=4
export GLUE_DIR=glue_data
export TASK=MNLI
python train_adversarial_student_bert.py --model_name_or_path  bert_models/distilbert-base \
    --task_name $TASK --data_dir $GLUE_DIR/$TASK --output_dir ckpts/students_distilbert/MNLI  \
    --do_train  --evaluate_during_training --overwrite_output_dir --learning_rate 2e-5   \
    --teacher_path ckpts/teachers_bert_base/MNLI --num_train_epochs 30 --per_gpu_train_batch_size 32 --logging_steps 500 --save_steps 500  \
    # Redirect to file
    #> logs/mnli_disbert.log 2>&1 &

