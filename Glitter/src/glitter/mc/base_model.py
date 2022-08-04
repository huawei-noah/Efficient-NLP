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

import logging
import os
from argparse import Namespace
from collections import OrderedDict
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
import transformers
from pytorch_lightning.utilities import rank_zero_warn
from torchmetrics import Accuracy, ConfusionMatrix
from transformers import AutoTokenizer

from glitter.mc.processors import mc_processors
from .dataset import MultiChoiceBatch
from ..lightning_base import BaseTransformer
from ..log_utils import set_global_logging_error

transformers.logging.set_verbosity_error()
set_global_logging_error(["tensorflow", "tensorboard", "urllib3.connectionpool"])
logger = logging.getLogger(__name__)


def get_model_inputs(
    model_type: str, input_ids, token_type_ids=None, attention_mask=None, labels=None
) -> Dict[str, Optional[torch.Tensor]]:
    inputs = dict(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    if model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
        del inputs["token_type_ids"]

    return inputs


class MultipleChoiceTransformer(BaseTransformer):
    mode = "multiple-choice"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self._set_defaults(hparams)
        tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name if hparams.tokenizer_name else hparams.model_name_or_path,
            cache_dir=hparams.cache_dir,
            use_fast=True,
        )
        super().__init__(hparams, None, self.mode, tokenizer=tokenizer, return_dict=True)

    def _set_defaults(self, hparams):
        self.data_processor = mc_processors[hparams.task]()

    def forward(self, **inputs):
        return self.model(**inputs)

    def setup(self, mode):
        super().setup(mode)
        self.valid_dataset = self.get_dataset("dev")
        # self.valid_features_per_example = collections.defaultdict(list)
        # for i, feature in enumerate(self.valid_dataset.features):
        #     self.valid_features_per_example[feature["example_index"]].append(i)

    def configure_metrics(self, stage: str) -> Optional[Any]:
        self.accuracy = Accuracy(num_classes=self.data_processor.num_labels)
        self.confusion_matrix = ConfusionMatrix(num_classes=self.data_processor.num_labels)

    def validation_step(self, batch, batch_idx):
        batch = MultiChoiceBatch.sub_batches(batch)[0]

        inputs = get_model_inputs(
            self.config.model_type,
            batch.input_ids,
            batch.token_type_ids,
            batch.attention_mask,
            batch.labels,
        )
        outputs = self(**inputs)

        preds = torch.argmax(outputs.logits, dim=1)
        if batch.labels is not None:
            self.accuracy.update(preds, batch.labels)
            self.confusion_matrix.update(preds, batch.labels)

            tmp_eval_loss = outputs.loss
            return {
                "val_loss": tmp_eval_loss.detach().cpu(),
                "pred": preds,
                "target": batch.labels.detach().cpu().numpy(),
            }
        else:
            return {"pred": preds}

    def _eval_end(self, outputs) -> tuple:
        preds = torch.cat([x["pred"] for x in outputs], dim=0)
        # predictions = {k: v for out in outputs for k, v in out["predictions"].items()}

        val_loss_mean = None
        if all("val_loss" in x for x in outputs):
            val_loss_mean = np.stack([x["val_loss"] for x in outputs]).mean()

        accuracy = self.accuracy.compute().item()
        self.confusion_matrix.compute()

        metric = OrderedDict({"accuracy": accuracy})

        logger.info(f"Evaluation metric: {metric}")

        return val_loss_mean, preds, metric

    def on_validation_epoch_start(self) -> None:
        self.accuracy.reset()
        self.confusion_matrix.reset()

    def on_test_epoch_start(self) -> None:
        self.accuracy.reset()
        self.confusion_matrix.reset()

    def validation_epoch_end(self, outputs: list):
        val_loss, _, metrics = self._eval_end(outputs)
        self.log("valid/loss", torch.FloatTensor([val_loss])[0], prog_bar=True, logger=True)

        val_metric = next(iter(metrics.values()))
        self.log("val_metric", torch.FloatTensor([val_metric])[0], logger=True)

        for met_name, met_val in metrics.items():
            self.log(f"valid/{met_name}", met_val, prog_bar=False, logger=True)
            self.log(met_name, torch.FloatTensor([met_val])[0], prog_bar=True, logger=False)

    def test_epoch_end(self, outputs):
        test_loss, preds, metrics = self._eval_end(outputs)

        for met_name, met_val in metrics.items():
            self.log(f"valid/{met_name}", met_val, prog_bar=False, logger=True)
            self.log(met_name, torch.FloatTensor([met_val])[0], prog_bar=True, logger=False)

        if test_loss is not None:
            self.log("test/loss", test_loss)

        if self.trainer.use_ddp and torch.cuda.device_count() > 1:
            out_preds = [torch.zeros_like(preds) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(out_preds, preds)
            if dist.get_rank() == 0:
                all_preds = torch.empty(
                    preds.shape[0] * dist.get_world_size(),
                    device=preds.device,
                    dtype=preds.dtype,
                )
                for current_rank in range(dist.get_world_size()):
                    all_preds[current_rank :: dist.get_world_size()] = out_preds[current_rank]
                preds = all_preds

        self._save_predictions(preds.detach().cpu().numpy())
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"test_loss": test_loss}

    def _save_predictions(self, preds):
        task_name = self.hparams.task

        output_suffix = ""
        if self.hparams.do_test and getattr(self.hparams, "test_mode", "") == "test2":
            output_suffix = "-test"
        elif self.hparams.do_eval and getattr(self.hparams, "test_mode", "") == "test":
            output_suffix = "-dev"

        output_test_results_file = os.path.join(
            self.hparams.output_dir,
            f"{task_name}{output_suffix}.tsv",
        )

        try:
            with open(output_test_results_file, "w") as writer:
                writer.write("index\tprediction\n")
                for i, pred in enumerate(preds):
                    writer.write("{}\t{}\n".format(i, pred))
        except PermissionError:
            rank_zero_warn(f"Cannot save predictions due to premission error at `{output_test_results_file}`")

    @staticmethod
    def add_model_specific_args(parser):
        BaseTransformer.add_model_specific_args(parser)
        parser.add_argument(
            "--max_seq_length",
            default=0,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--dataset_name",
            default=None,
            type=str,
            help="Dataset name to be used in the name of the output directory.",
        )
        parser.add_argument(
            "--task",
            default="",
            type=str,
            required=True,
            choices=tuple(mc_processors.keys()),
            help="The GLUE task to run",
        )

        return parser
