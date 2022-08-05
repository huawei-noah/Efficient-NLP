# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Changes:
# 2022.06.30 - A dictionary that maps tasks to their corresponding metric function added
#               Huawei Technologies Co., Ltd. <foss@huawei.com>
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr


def simple_accuracy(preds, labels, metric_name: str = "acc"):
    """
    Modified from 🤗 Transformers: https://github.com/huggingface/transformers/blob/v3.5.1/src/transformers/data/metrics/__init__.py
    Args:
        preds:
        labels:

    Returns:

    """
    return {metric_name: (preds == labels).mean()}


def acc_and_f1(preds, labels):
    """
    Verbatim from 🤗 Transformers: https://github.com/huggingface/transformers/blob/v3.5.1/src/transformers/data/metrics/__init__.py
    Args:
        preds:
        labels:

    Returns:

    """
    acc = next(iter(simple_accuracy(preds, labels).values()))
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    """
    Verbatim from 🤗 Transformers: https://github.com/huggingface/transformers/blob/v3.5.1/src/transformers/data/metrics/__init__.py
    Args:
        preds:
        labels:

    Returns:

    """
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


glue_metrics = {
    "cola": (lambda preds, labels: {"mcc": matthews_corrcoef(labels, preds)}),
    "mrpc": acc_and_f1,
    "qqp": acc_and_f1,
    "paws_qqp": acc_and_f1,
    "sts-b": pearson_and_spearman,
    "mnli": (lambda preds, labels: simple_accuracy(preds, labels, "mnli/acc")),
    "mnli-mm": (lambda preds, labels: simple_accuracy(preds, labels, "mnli-mm/acc")),
    "mnli2": simple_accuracy,
    "mnli2-mm": simple_accuracy,
    "sst-2": simple_accuracy,
    "rte": simple_accuracy,
    "wnli": simple_accuracy,
    "qnli": simple_accuracy,
    "hans": simple_accuracy,
    "sst-5": simple_accuracy,
    "sstx": simple_accuracy,
    "trec": simple_accuracy,
    "ag_news": simple_accuracy,
    "cr": simple_accuracy,
    "imp": simple_accuracy,
    "imdb": simple_accuracy,
    "imdb-5": simple_accuracy,
    "tf-imdb": simple_accuracy,
    "tf-imdb-5": simple_accuracy,
    "boolq": simple_accuracy,
}

