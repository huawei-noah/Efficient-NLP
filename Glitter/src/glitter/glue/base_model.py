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
from dataclasses import dataclass
from typing import Optional, Dict, Iterable, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from pytorch_lightning.utilities import rank_zero_warn
from transformers import AutoConfig

from .dataset import (
    glue_output_modes,
    glue_tasks_num_labels,
    glue_submission_names,
    glue_submission_labels,
    glue_compute_metrics,
)
from ..hf_utils import (
    HF_SEQUENCE_CLASSIFICATION,
)
from ..lightning_base import BaseTransformer
from ..log_utils import set_global_logging_error

transformers.logging.set_verbosity_error()
set_global_logging_error(["tensorflow", "tensorboard", "urllib3.connectionpool"])
logger = logging.getLogger(__name__)


@dataclass
class SimpleBatch:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


@dataclass
class GLUEBatch:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    token_type_ids: Optional[torch.Tensor]
    labels: Optional[torch.Tensor] = None
    augmented_input_ids: Optional[torch.Tensor] = None
    augmented_attention_mask: Optional[torch.Tensor] = None
    augmented_token_types: Optional[torch.Tensor] = None
    example_indices: Optional[torch.Tensor] = None
    augmented_indices: Optional[torch.Tensor] = None
    augmented_mask: Optional[torch.Tensor] = None
    augmented_ranks: Optional[torch.Tensor] = None

    def select_subset(self, attr: Union[str, Optional[torch.Tensor]], selected_indices) -> Optional[torch.Tensor]:
        if attr is not None and isinstance(attr, str):
            val = getattr(self, attr, None)
        else:
            val = attr

        return val[selected_indices] if val is not None else val

    @classmethod
    def get_minibatches(cls, batch: Iterable[Tuple[Optional[torch.Tensor], ...]]) -> Dict[int, "GLUEBatch"]:
        return {i: GLUEBatch(*minibatch) for i, minibatch in enumerate(batch) if minibatch}


class GLUETransformer(BaseTransformer):
    mode = "sequence-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self._set_defaults(hparams)

        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]

        if hparams.config_name:
            num_labels_old = AutoConfig.from_pretrained(hparams.config_name).num_labels
        elif os.path.exists(hparams.model_name_or_path):
            num_labels_old = AutoConfig.from_pretrained(hparams.model_name_or_path).num_labels
        else:
            num_labels_old = num_labels

        super().__init__(hparams, num_labels_old, self.mode, return_dict=True)

        if num_labels != num_labels_old:
            self.config.num_labels = num_labels
            self.model.num_labels = num_labels
            logger.info(
                f"Classifier heads in model are reset because number of labels is different "
                f"in the pre-trained model from the task: {num_labels_old} != {num_labels}"
            )
            HF_SEQUENCE_CLASSIFICATION[self.config.model_type].reinitialize(self.config, self.model)

        self._reinitialize()

    def _set_defaults(self, hparams):
        pass

    def _reinitialize(self):
        """
        Taken from: https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
        """
        if self.hparams.reinit_pooler:
            if self.config.model_type in ["bert", "roberta"]:
                encoder_temp = getattr(self.model, self.config.model_type)
                if encoder_temp.pooler is not None:
                    encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                    encoder_temp.pooler.dense.bias.data.zero_()
                    for p in encoder_temp.pooler.parameters():
                        p.requires_grad = True
            elif self.config.model_type in ["xlnet", "bart", "electra"]:
                raise ValueError(f"{self.config.model_type} does not have a pooler at the end")
            else:
                raise NotImplementedError

        if self.hparams.reinit_layers > 0:
            if self.config.model_type in ["bert", "roberta", "electra"]:
                assert self.hparams.reinit_pooler or self.config.model_type == "electra"

                encoder_temp = getattr(self.model, self.config.model_type)
                for layer in encoder_temp.encoder.layer[-self.hparams.reinit_layers :]:
                    for module in layer.modules():
                        if isinstance(module, (nn.Linear, nn.Embedding)):
                            # Slightly different from the TF version which uses truncated_normal for initialization
                            # cf https://github.com/pytorch/pytorch/pull/5617
                            module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                        elif isinstance(module, nn.LayerNorm):
                            module.bias.data.zero_()
                            module.weight.data.fill_(1.0)
                        if isinstance(module, nn.Linear) and module.bias is not None:
                            module.bias.data.zero_()
            elif self.config.model_type == "xlnet":
                from transformers.models.xlnet.modeling_xlnet import XLNetLayerNorm, XLNetRelativeAttention

                for layer in self.model.transformer.layer[-self.hparams.reinit_layers :]:
                    for module in layer.modules():
                        if isinstance(module, (nn.Linear, nn.Embedding)):
                            # Slightly different from the TF version which uses truncated_normal for initialization
                            # cf https://github.com/pytorch/pytorch/pull/5617
                            module.weight.data.normal_(mean=0.0, std=self.model.transformer.config.initializer_range)
                            if isinstance(module, nn.Linear) and module.bias is not None:
                                module.bias.data.zero_()
                        elif isinstance(module, XLNetLayerNorm):
                            module.bias.data.zero_()
                            module.weight.data.fill_(1.0)
                        elif isinstance(module, XLNetRelativeAttention):
                            for param in [
                                module.q,
                                module.k,
                                module.v,
                                module.o,
                                module.r,
                                module.r_r_bias,
                                module.r_s_bias,
                                module.r_w_bias,
                                module.seg_embed,
                            ]:
                                param.data.normal_(mean=0.0, std=self.model.transformer.config.initializer_range)
            elif self.config.model_type == "bart":
                for layer in self.model.model.decoder.layers[-self.hparams.reinit_layers :]:
                    for module in layer.modules():
                        self.model.model._init_weights(module)

            else:
                raise NotImplementedError

    def forward(self, **inputs):
        return self.model(**inputs)

    def _eval_end(self, outputs) -> tuple:
        preds = torch.cat([x["pred"] for x in outputs], dim=0)

        val_loss_mean = None
        metrics = {}
        out_label_ids = None
        if all("val_loss" in x for x in outputs):
            val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()

        if self.hparams.glue_output_mode == "classification":
            preds = torch.argmax(preds, dim=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = preds.squeeze()

        if all("target" in x for x in outputs):
            out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
            metrics = glue_compute_metrics(self.hparams.task, preds.detach().cpu().numpy(), out_label_ids)

        return val_loss_mean, metrics, preds, out_label_ids

    def validation_epoch_end(self, outputs: list):
        val_loss, metrics, *_ = self._eval_end(outputs)
        self.log("valid/loss", torch.FloatTensor([val_loss])[0], prog_bar=True, logger=True)

        val_metric = next(iter(metrics.values()))
        self.log("val_metric", torch.FloatTensor([val_metric])[0], logger=True)

        for met_name, met_val in metrics.items():
            self.log(f"valid/{met_name}", met_val, prog_bar=False, logger=True)
            self.log(met_name, torch.FloatTensor([met_val])[0], prog_bar=True, logger=False)

    def test_epoch_end(self, outputs):
        test_loss, test_metrics, preds, out_labels = self._eval_end(outputs)

        if test_metrics:
            val_metric = next(iter(test_metrics.values()))
            self.log("test/metric", val_metric, logger=True)

        for met_name, met_val in test_metrics.items():
            self.log(f"test/{met_name}", met_val, logger=True)

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

    def _save_predictions(self, preds):
        task_name = self.hparams.task
        label_type = glue_submission_labels[task_name]

        output_suffix = ""
        if self.hparams.do_test and getattr(self.hparams, "test_mode", "") == "test2":
            output_suffix = "-test"
        elif self.hparams.do_eval and getattr(self.hparams, "test_mode", "") == "test":
            output_suffix = "-dev"

        output_test_results_file = os.path.join(
            self.hparams.output_dir,
            f"{glue_submission_names[task_name]}{output_suffix}.tsv",
        )

        try:
            with open(output_test_results_file, "w") as writer:
                writer.write("index\tprediction\n")
                for i, pred in enumerate(preds):
                    if label_type == "string":
                        writer.write("{}\t{}\n".format(i, self.labels[pred]))
                    elif label_type == "float":
                        if task_name == "sts-b":
                            pred = np.clip(pred, 0.0, 5.0)
                        writer.write("{}\t{:.3f}\n".format(i, float(pred)))
                    else:
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
            help="The GLUE task to run",
        )

        parser.add_argument(
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets",
        )

        parser.add_argument(
            "--reinit_layers",
            type=int,
            default=0,
            help="re-initialize the last N Transformer blocks. reinit_pooler must be turned on.",
        )
        parser.add_argument(
            "--reinit_pooler",
            action="store_true",
            help="reinitialize the pooler",
        )

        return parser
