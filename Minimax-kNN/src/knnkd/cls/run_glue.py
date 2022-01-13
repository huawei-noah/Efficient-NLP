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

import glob
import logging
import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Iterable, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader
from transformers import AutoConfig

from ..hf_utils import (
    HF_SEQUENCE_CLASSIFICATION,
    get_last_layer_hidden_states,
    get_model_inputs as hf_get_model_inputs,
)
from ..log_utils import set_global_logging_error
from ..modeling import (
    KDHead,
    KDHeadFast,
    MinimaxKnnHead,
    MinimaxKnnHeadFast,
)
from ..data import (
    AugmentedInputExample,
    GLUEv2Dataset,
    GLUEv2Processor,
    FeaturesCollatorWithPadding,
    cached_model_outputs,
    glue_output_modes,
    glue_submission_names,
    glue_submission_labels,
    glue_tasks_num_labels,
    load_metric,
)
from ..lightning_base import BaseTransformer, generic_train


transformers.logging.set_verbosity_error()
set_global_logging_error(["tensorflow", "tensorboard", "urllib3.connectionpool"])
logger = logging.getLogger(__name__)


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
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]

        if hparams.config_name:
            num_labels_old = AutoConfig.from_pretrained(hparams.config_name).num_labels
        elif os.path.exists(hparams.model_name_or_path):
            num_labels_old = AutoConfig.from_pretrained(hparams.model_name_or_path).num_labels
        else:
            num_labels_old = num_labels

        super().__init__(hparams, num_labels_old, self.mode, return_dict=True)

        self.metric = load_metric(self.hparams.task)

        self.teacher_saved_output = None
        self.teacher = None
        if self.hparams.teacher_name_or_path is not None:
            if self.hparams.online_teacher:
                teacher_config = AutoConfig.from_pretrained(
                    self.hparams.teacher_name_or_path,
                    **({"num_labels": num_labels} if num_labels is not None else {}),
                    cache_dir=self.hparams.cache_dir,
                    return_dict=True,
                )
                self.teacher = self.model_type.from_pretrained(
                    self.hparams.teacher_name_or_path,
                    from_tf=bool(".ckpt" in self.hparams.teacher_name_or_path),
                    config=teacher_config,
                    cache_dir=self.hparams.cache_dir,
                )
            else:
                self.teacher_saved_output = cached_model_outputs(
                    self.hparams.teacher_name_or_path,
                    self.hparams.task,
                    self.hparams.data_dir,
                    self.hparams.cache_batch_size or self.hparams.train_batch_size,
                    self.hparams.max_seq_length,
                    cache_dir=self.hparams.cache_dir,
                    dataset_name=self.hparams.dataset_name,
                    max_augment_length=self.hparams.max_aug_length,
                    num_workers=self.hparams.num_workers,
                )

        if num_labels != num_labels_old:
            self.config.num_labels = num_labels
            self.model.num_labels = num_labels
            logger.info(
                f"Classifier heads in model are reset because number of labels is different "
                f"in the pre-trained model from the task: {num_labels_old} != {num_labels}"
            )
            HF_SEQUENCE_CLASSIFICATION[self.config.model_type].reinitialize(self.config, self.model)

        self._reinitialize()

        if self._is_distillation_on():
            if self._is_teacher_cached():
                self.kd_head = KDHeadFast(self.teacher_saved_output.logits, self.hparams.temperature)
            else:
                self.kd_head = KDHead(self.teacher, self.hparams.temperature)
        else:
            self.kd_head = None

        if self._is_minimax_on():
            if self._is_teacher_cached():
                self.minimax_head = MinimaxKnnHeadFast(
                    self.teacher_saved_output.logits,
                    self.model,
                    self.hparams.temperature,
                )
            else:
                self.minimax_head = MinimaxKnnHead(
                    self.teacher,
                    self.model,
                    self.hparams.temperature,
                    self.hparams.min_distance,
                    self.hparams.max_distance,
                    self.hparams.maxim_func,
                )
        else:
            self.minimax_head = None

    def _is_teacher_cached(self) -> bool:
        return self.teacher_saved_output is not None

    def _is_distillation_on(self) -> bool:
        return self.teacher is not None or self.teacher_saved_output is not None

    def _is_augmentation_on(self) -> bool:
        return self._is_distillation_on() and self.hparams.num_augments > 0

    def _is_minimax_on(self) -> bool:
        return self._is_augmentation_on() and not self.hparams.naive_augment

    def _is_naive_augment_on(self) -> bool:
        return self._is_augmentation_on() and self.hparams.naive_augment

    def forward(self, **inputs):
        return self.model(**inputs)

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
                from transformers.modeling_xlnet import XLNetLayerNorm, XLNetRelativeAttention

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

    def get_trainable_parameters(self):
        optimizer_grouped_parameters = super().get_trainable_parameters()
        # if self.ref_logits_head is not None:
        #     optimizer_grouped_parameters.append(
        #         {
        #             "params": [p for n, p in self.ref_logits_head.named_parameters()],
        #             "weight_decay": self.hparams.weight_decay,
        #         }
        #     )
        return optimizer_grouped_parameters

    def _add_distance_threshold_log(
        self, threshold: int, filtered_ranks: Optional[torch.Tensor], prefix: str
    ) -> Optional[int]:
        if threshold > 0 and filtered_ranks is not None:
            if filtered_ranks.nelement() > 0:
                self.logger.experiment.add_histogram(
                    f"minimax/{prefix}_filtered_ranks",
                    filtered_ranks,
                    self.global_step,
                )

            self.log(
                f"minimax/num_{prefix}_filtered_neighbors",
                filtered_ranks.nelement(),
                logger=True,
            )
            return filtered_ranks.nelement()

        return None

    def training_step(self, batch, batch_idx):
        minibatches = GLUEBatch.get_minibatches(batch)

        # skip a batch
        if 0 not in minibatches and (self._is_minimax_on() and self.current_epoch < self.hparams.augment_start_epoch):
            return None

        if self.teacher is not None:
            self.teacher.eval()

        loss = 0.0
        for aug_rank, minibatch in minibatches.items():
            inputs = hf_get_model_inputs(
                self.config.model_type,
                minibatch.input_ids,
                minibatch.token_type_ids,
                minibatch.attention_mask,
                minibatch.labels,
            )

            if aug_rank == 0:
                output = self(**inputs)
                cls_loss = output.loss

                # for auxiliary samples that do not have labels
                if cls_loss is None:
                    cls_loss = 0.0
                else:
                    self.log("train/cls_loss", cls_loss, logger=True)

                if self._is_distillation_on():
                    if self._is_teacher_cached():
                        kd_loss = self.kd_head(
                            output.logits,
                            example_indices=minibatch.example_indices,
                            augmented_indices=minibatch.augmented_indices,
                        )
                    else:
                        teacher_inputs = dict(inputs)
                        teacher_inputs.pop("labels", None)
                        if "token_type_ids" not in inputs:
                            teacher_inputs["token_type_ids"] = (
                                None if self.teacher.config.model_type == "xlm" else minibatch.token_type_ids
                            )

                        kd_loss, _ = self.kd_head(
                            output.logits,
                            **teacher_inputs,
                        )
                    self.log("train/kd_loss", kd_loss, logger=True)

                    loss += self.hparams.alpha_ce * kd_loss + self.hparams.alpha_true * cls_loss
                else:
                    loss = cls_loss
                    break
            else:
                if self.current_epoch < self.hparams.augment_start_epoch:
                    continue

                assert self._is_minimax_on() and minibatch.augmented_input_ids is not None

                augmented_inputs = hf_get_model_inputs(
                    self.config.model_type,
                    minibatch.augmented_input_ids,
                    minibatch.augmented_token_types,
                    minibatch.augmented_attention_mask,
                )

                if self._is_teacher_cached():
                    minimax_output = self.minimax_head(
                        aug_rank,
                        minibatch.augmented_mask,
                        minibatch.example_indices,
                        nn_ranks=minibatch.augmented_ranks,
                        augmented_indices=minibatch.augmented_indices,
                        **augmented_inputs,
                    )
                else:
                    teacher_inputs = dict(inputs)
                    teacher_inputs.pop("labels", None)
                    if "token_type_ids" not in inputs:
                        teacher_inputs["token_type_ids"] = (
                            None if self.teacher.config.model_type == "xlm" else minibatch.token_type_ids
                        )

                    with torch.no_grad():
                        teacher_output = self.teacher(
                            **teacher_inputs,
                            output_hidden_states=True,
                        )

                    minimax_output = self.minimax_head(
                        aug_rank,
                        nn_mask=minibatch.augmented_mask,
                        tea_orig_hidden_states=get_last_layer_hidden_states(self.teacher.config, teacher_output),
                        nn_ranks=minibatch.augmented_ranks,
                        **augmented_inputs,
                    )

                max_filtered_ranks = self._add_distance_threshold_log(
                    self.hparams.max_distance, minimax_output.max_filtered_ranks, "max"
                )
                min_filtered_ranks = self._add_distance_threshold_log(
                    self.hparams.min_distance, minimax_output.min_filtered_ranks, "min"
                )

                if max_filtered_ranks is not None or min_filtered_ranks is not None:
                    num_filtered_ranks = (max_filtered_ranks or 0) + (min_filtered_ranks or 0)
                    self.log(
                        "minimax/num_filtered_neighbors",
                        num_filtered_ranks,
                        logger=True,
                    )
                    self.log(
                        "minimax/filtered_neighbors_percent",
                        100.0 * num_filtered_ranks / batch.augmented_ranks.nelement(),
                        logger=True,
                    )

                if not minimax_output.is_empty:
                    if minimax_output.has_selected_ranks:
                        self.logger.experiment.add_histogram(
                            "minimax/selected_ranks",
                            minimax_output.selected_ranks,
                            self.global_step,
                        )

                    if minimax_output.has_selected_distances:
                        self.logger.experiment.add_histogram(
                            "minimax/selected_distances",
                            minimax_output.selected_distances,
                            self.global_step,
                        )

                        self.log(
                            "minimax/avg_selected_distances",
                            torch.mean(minimax_output.selected_distances),
                            logger=True,
                        )

                    aug_output = self(
                        input_ids=minibatch.select_subset("augmented_input_ids", minimax_output.selected_neighbors),
                        attention_mask=minibatch.select_subset(
                            "augmented_attention_mask",
                            minimax_output.selected_neighbors,
                        ),
                        token_type_ids=minibatch.select_subset(
                            augmented_inputs.get("token_type_ids", None),
                            minimax_output.selected_neighbors,
                        ),
                        labels=None,
                    )

                    if self._is_teacher_cached():
                        loss_aug = self.kd_head(aug_output.logits, minimax_output.teacher_logits)
                    else:
                        loss_aug, _ = self.kd_head(aug_output.logits, minimax_output.teacher_logits)
                    self.log("train/aug_loss", loss_aug, logger=True)
                    loss += self.hparams.alpha_aug * loss_aug

        if loss is not None:
            self.log("train/loss", loss, prog_bar=False, logger=True)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log("train/lr", lr_scheduler.get_last_lr()[-1], prog_bar=True, logger=True)

        return loss

    # def on_train_epoch_start(self):
    #     if self._is_minimax_on() and self.current_epoch > 0:
    #         if (
    #             self.current_epoch > self.hparams.augment_start_epoch
    #             and self.hparams.minimax_update_interval > 0
    #             and (self.current_epoch - self.hparams.augment_start_epoch) % self.hparams.minimax_update_interval == 0
    #         ) or (self.current_epoch == self.hparams.augment_start_epoch):
    #             rank_zero_info(f"Ep {self.current_epoch}, updating student model for minimax ranking...")
    #             self.minimax_head.student.load_state_dict(self.model.state_dict())

    def total_steps(self) -> int:
        if self._is_augmentation_on():
            num_devices = max(1, self.get_number_of_gpus())
            effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
            effective_dataset_size = len(self.train_loader.dataset) / effective_batch_size
            effective_augmented_dataset_size = len(self.augmented_train_loader.dataset) / effective_batch_size
            return effective_dataset_size * self.hparams.augment_start_epoch + effective_augmented_dataset_size * (
                self.hparams.max_epochs - self.hparams.augment_start_epoch
            )
        else:
            return super().total_steps()

    def setup(self, mode):
        super().setup(mode)
        if mode == "fit" and self._is_augmentation_on():
            args = self.hparams

            processor = GLUEv2Processor(args.task, num_augments=-1)

            self._filter_augments(processor.get_train_examples(args.data_dir))
            self.augmented_train_loader = self.get_dataloader("augmented_train", args.train_batch_size, shuffle=True)

    def train_dataloader(self):
        if self._is_augmentation_on() and self.current_epoch >= self.hparams.augment_start_epoch:
            return self.augmented_train_loader

        return super().train_dataloader()

    def get_dataloader(self, mode: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        "Load datasets. Called after prepare data."

        # We test on dev set to compare to benchmarks without having to submit to GLUE server
        mode = "dev" if mode == "test" else mode
        if not mode.endswith("train"):
            self.hparams.test_mode = mode

        args = self.hparams

        processor = GLUEv2Processor(
            args.task,
            num_augments=-1,
        )
        self.labels = processor.get_labels()

        if mode == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif mode == "infer":
            examples = processor.get_test_examples(args.data_dir)
        elif mode == "test2":
            examples = processor.get_labelled_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)

        dataset = GLUEv2Dataset(
            examples,
            self.labels,
            args.glue_output_mode,
            args.max_seq_length,
            self.tokenizer,
            args.max_aug_length,
            args.num_augments if mode == "augmented_train" else 0,
            torch.load(self._get_cached_allowed_indices_file())
            if mode == "augmented_train" and self._is_teacher_cached()
            else None,
            naive_augment=mode == "augmented_train" and self._is_naive_augment_on(),
            padding=self.hparams.padding,
        )

        dl_kwargs = {}
        # if is_ddp_enabled(self.hparams.distributed_backend, self.hparams.gpus):
        # dl_kwargs["sampler"] = DistributedSampler(dataset, shuffle=shuffle, drop_last=mode != "train")
        # dl_kwargs["drop_last"] = mode != "train"

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=FeaturesCollatorWithPadding(
                self.tokenizer,
                pad_to_multiple_of=self.hparams.pad_to_multiple_of,
                group_features=mode == "augmented_train" and self._is_minimax_on(),
                num_augment_ranks=(self.hparams.num_augments + 1) if mode == "augmented_train" else 0,
            ),
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            **dl_kwargs,
        )

    def validation_step(self, batch, batch_idx):
        batch = GLUEBatch.get_minibatches(batch)[0]
        inputs = hf_get_model_inputs(
            self.config.model_type, batch.input_ids, batch.token_type_ids, batch.attention_mask, batch.labels
        )
        outputs = self(**inputs)

        preds = outputs.logits  # .detach().cpu().numpy()

        if inputs.get("labels", None) is not None:
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            tmp_eval_loss = outputs.loss
            return {
                "val_loss": tmp_eval_loss.detach().cpu(),
                "pred": preds,
                "target": out_label_ids,
            }
        else:
            return {"pred": preds}

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
            metrics = self.metric.compute(predictions=preds.detach().cpu().numpy(), references=out_label_ids)

        return val_loss_mean, metrics, preds, out_label_ids

    def validation_epoch_end(self, outputs: list):
        val_loss, metrics, *_ = self._eval_end(outputs)
        self.log("valid/loss", val_loss, prog_bar=True, logger=True)

        val_metric = next(iter(metrics.values()))
        self.log("val_metric", val_metric, logger=True)

        for met_name, met_val in metrics.items():
            self.log(f"valid/{met_name}", met_val, prog_bar=False, logger=True)
            self.log(met_name, met_val, prog_bar=True, logger=False)

    def test_epoch_end(self, outputs):
        test_loss, test_metrics, preds, out_labels = self._eval_end(outputs)

        if test_metrics:
            val_metric = next(iter(test_metrics.values()))
            self.log("test/metric", val_metric, logger=True)

        for met_name, met_val in test_metrics.items():
            self.log(f"test/{met_name}", met_val, logger=True)

        if test_loss is not None:
            self.log("test/loss", test_loss)

        if self.use_ddp and torch.cuda.device_count() > 1:
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
        # return {"test_loss": test_loss}

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

        with open(output_test_results_file, "w") as writer:
            writer.write("index\tprediction\n")
            for i, pred in enumerate(preds):
                if label_type == "string":
                    writer.write("{}\t{}\n".format(i, self.labels[pred]))
                elif label_type == "float":
                    writer.write("{}\t{.3f}\n".format(i, float(pred)))
                else:
                    writer.write("{}\t{}\n".format(i, pred))

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def _get_cached_allowed_indices_file(self):
        return os.path.join(self.hparams.output_dir, f".allowed_indices.bin")

    def _filter_augments(
        self,
        examples: List[AugmentedInputExample],
    ):
        if not self._is_teacher_cached():
            return

        cache_file = self._get_cached_allowed_indices_file()

        if os.path.exists(cache_file):
            return

        min_distance = self.hparams.min_distance
        max_distance = self.hparams.max_distance

        hidden_states = self.teacher_saved_output.hidden_states
        num_augmented_examples = 0
        num_retained_augmented_examples = 0
        allowed_indices = []

        cos = nn.CosineSimilarity()

        rank_zero_info("*** Filtering based on distance constraints")

        for i, ex in enumerate(examples):
            num_augmented_examples += len(ex.augmented_examples)
            n_augs = self.hparams.num_aug_candidates if self._is_minimax_on() else self.hparams.num_augments

            if self.hparams.preserve_order:
                # if self.hparams.start_rank < len(ex.augmented_examples):
                retained_indices = np.arange(
                    min(n_augs, len(ex.augmented_examples)),
                )
            else:
                distances = (
                    torch.acos(
                        torch.clamp(
                            cos(
                                self.teacher_saved_output.hidden_states[i, 0].reshape(1, -1),
                                hidden_states[i, 1 : len(ex.augmented_examples) + 1],
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                    / np.pi
                )

                if min_distance > 0 and max_distance > 0:
                    retained_indices = torch.nonzero(
                        (distances >= min_distance) & (distances <= max_distance), as_tuple=True
                    )[0]
                elif min_distance > 0:
                    retained_indices = torch.nonzero(distances >= min_distance, as_tuple=True)[0]
                elif max_distance > 0:
                    retained_indices = torch.nonzero((distances <= max_distance) & (distances > 0), as_tuple=True)[0]
                else:
                    retained_indices = torch.nonzero(distances > 0, as_tuple=True)[0]

                if n_augs < retained_indices.shape[0]:
                    retained_distances = distances[retained_indices]
                    retained_indices = retained_indices[torch.topk(retained_distances, n_augs, largest=False)[1]]

            num_retained_augmented_examples += len(retained_indices)
            allowed_indices.append(retained_indices.tolist())

        rank_zero_info(f">>> original examples: {len(examples)}")
        rank_zero_info(
            f">>> after augmentation: {num_augmented_examples} ({num_augmented_examples / len(examples):.2f})"
        )
        rank_zero_info(
            f">>> after distance constraints: {num_retained_augmented_examples} "
            f"(w.r.t. aug {num_retained_augmented_examples / num_augmented_examples:.2f} - "
            f"w.r.t. orig {num_retained_augmented_examples / len(examples):.2f})"
        )

        torch.save(allowed_indices, cache_file)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
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

        parser.add_argument(
            "--num_augments",
            default=0,
            type=int,
            help="Number of augmentations per train samples used during training",
        )
        parser.add_argument(
            "--num_aug_candidates",
            default=4,
            type=int,
            help="Number of candidates per train samples to select augmentation samples from",
        )
        # parser.add_argument(
        #     "--start_rank",
        #     default=0,
        #     type=int,
        #     help="Starting rank (from zero) for nearest neighbors "
        #     "(e.g., for second nearest neighbor, `start_rank` and `num_augments` should be set to 1)",
        # )
        parser.add_argument(
            "--augment_start_epoch",
            default=0,
            type=int,
            help="First epoch (starting from zero) in which augmented data incorporated into training. "
            "Earlier epochs are warm-up and run standard KD.",
        )
        parser.add_argument(
            "--minimax_update_interval",
            default=1,
            type=int,
            help="Number of epochs that minimax ranking is frozen and gets updated after. 0=never updates",
        )
        parser.add_argument(
            "--max_distance",
            default=0.0,
            type=float,
            help="Maximum distance threshold to skip augmentation for too distant neighbors. "
            "The threshold's order of magnitude depends on the distance function.",
        )
        parser.add_argument(
            "--min_distance",
            default=0.0,
            type=float,
            help="Minimum distance threshold to skip augmentation for too close neighbors. "
            "The threshold's order of magnitude depends on the distance function.",
        )
        parser.add_argument(
            "--max_aug_length",
            default=0,
            type=int,
            help="Maximum length of augmented text sequence after tokenization. "
            "(0 = means follow `--max_seq_length`). Only when `num_augments` > 0",
        )
        parser.add_argument(
            "--naive_augment",
            action="store_true",
            default=False,
            help="Ignore minimax and naively include the whole augmented data in training. Only when `num_augments` > 0",
        )
        parser.add_argument(
            "--preserve_order",
            action="store_true",
            default=False,
            help="Preserves the order of augmented examples by skipping computing the distance in the teacher's space",
        )
        parser.add_argument(
            "--maxim_func",
            type=str,
            choices=("ce", "l2"),
            default="ce",
            help="Maximization function for ranking augmented examples w.r.t. teacher output and student output",
        )

        # Distillation parameters (optional)
        parser.add_argument(
            "--teacher_name_or_path",
            default=None,
            type=str,
            help="Path to the already fine-tuned teacher model. Only for distillation.",
        )
        parser.add_argument(
            "--alpha_ce",
            default=0.5,
            type=float,
            help="Distillation loss linear weight. Only for distillation.",
        )
        parser.add_argument(
            "--alpha_true",
            default=0.5,
            type=float,
            help="Classification loss linear weight. Only for distillation.",
        )
        parser.add_argument(
            "--temperature",
            default=2.0,
            type=float,
            help="Distillation temperature. Only for distillation.",
        )
        parser.add_argument(
            "--alpha_aug",
            default=None,
            type=float,
            help="Linear weight corresponding to cross-entropy loss of augmented samples between teacher and student. "
            "Only for distillation when `num_augments` > 0",
        )
        parser.add_argument(
            "--cache_batch_size",
            default=None,
            type=int,
            help="Batch size for caching teacher output (default: `--train_batch_size`)",
        )
        parser.add_argument(
            "--online_teacher",
            action="store_true",
            default=False,
            help="Whether to bypass offline caching of teacher output and acquire teacher outputs during training",
        )

        return parser


def _get_default_output_dir(args: Namespace) -> str:
    arg_summary = f"{args.task}_{Path(args.model_name_or_path).name}"

    if args.dataset_name:
        arg_summary += f"_{args.dataset_name}"

    if args.gpus < 0:
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = args.gpus

    arg_summary += f"_ep{args.max_epochs}" f"_bsz{args.train_batch_size}x{n_gpus}"

    if args.warmup_ratio > 0:
        arg_summary += f"_wmr{args.warmup_ratio}"
    else:
        arg_summary += f"_wm{args.warmup_steps}"

    arg_summary += (
        f"_lr{args.learning_rate}"
        f"_acc{args.accumulate_grad_batches}"
        f"_seq{args.max_seq_length}"
        f"_val{args.val_check_interval}"
        f"_pat{args.patience}"
        f"_sd{args.seed}"
        f"_wdec{args.weight_decay}"
    )

    if args.padding is not None:
        arg_summary += f"_pa{args.padding[:3]}"
    if args.pad_to_multiple_of is not None:
        arg_summary += f"_pam{args.pad_to_multiple_of}"

    if args.fp16:
        arg_summary += "_amp"

    if args.reinit_pooler:
        arg_summary += "_rep"

    if args.reinit_layers > 0:
        arg_summary += f"_rel{args.reinit_layers}"

    if args.deterministic:
        arg_summary += "_det"

    if args.num_augments > 0:
        arg_summary += f"_naug{args.num_augments}"
        if not args.naive_augment:
            arg_summary += f"of{args.num_aug_candidates}"
        # elif args.start_rank > 0:
        #     arg_summary += f"st{args.start_rank}"

        if args.maxim_func != "ce":
            arg_summary += f"_{args.maxim_func}"

        if args.min_distance > 0:
            arg_summary += f"_mind{args.min_distance}"

        if args.max_distance > 0:
            arg_summary += f"_maxd{args.max_distance}"

    if args.teacher_name_or_path is not None:
        arg_summary += f"_tau{args.temperature}"
        arg_summary += f"_ce{args.alpha_ce}"
        arg_summary += f"_tru{args.alpha_true}"

        if args.num_augments > 0:
            arg_summary += f"_aug{args.alpha_aug}"

            if args.augment_start_epoch > 0:
                arg_summary += f"_augst{args.augment_start_epoch}"

            if args.naive_augment:
                arg_summary += f"_naive"
                if args.preserve_order:
                    arg_summary += f"_order"

        arg_summary += "_distilled"

    return arg_summary


def _sanity_check(args):
    if args.do_train:
        assert 0.0 <= args.warmup_ratio <= 1.0, "`--warmup_ratio` must be in [0, 1]"

        if args.teacher_name_or_path is not None:
            if not args.naive_augment and args.num_augments > 0:
                assert args.minimax_update_interval >= 0, "`--minimax_update_interval` must not be negative"

                if args.min_distance > 0 and args.max_distance > 0:
                    assert args.min_distance <= args.max_distance, "`--min_distance` <= `--max_distance` must be true"

        assert not (
            args.do_infer or args.do_test or args.do_eval
        ), "`--do_infer`, `--do_test` and `--do_eval` cannot be done if training is enabled"


def _load_default_args(args):
    if args.teacher_name_or_path is not None and args.num_augments > 0 and args.alpha_aug is None:
        args.alpha_aug = args.alpha_ce

    if args.naive_augment and args.preserve_order:
        if args.min_distance > 0 or args.max_distance > 0:
            args.min_distance = 0
            args.max_distance = 0
            logger.warning("Distance thresholds are overwritten to zero because `--preserver_order` is activated")

    if not args.do_train and (args.do_infer or args.do_eval or args.do_test):
        hparams_path = Path(args.model_name_or_path).parent / "hparams.yaml"
        if hparams_path.exists():
            logger.info(
                "`hparams.yaml` found from which parameter values (max_seq_length, pad_to_multiple_of, and padding) will be loaded"
            )

            with hparams_path.open("r") as hparams_file:
                train_hparams = yaml.safe_load(hparams_file)

            if args.max_seq_length == 0:
                args.max_seq_length = train_hparams.get("max_seq_length", 0)

            if args.pad_to_multiple_of is None:
                args.pad_to_multiple_of = train_hparams.get("pad_to_multiple_of", None)

            if args.padding is None and "padding" in train_hparams:
                args.padding = train_hparams["padding"]

    if args.accelerator is None:
        if args.gpus == 0:
            args.accelerator = "ddp_cpu"
        elif args.gpus == 1:
            args.accelerator = "dp"
        elif args.gpus == -1:
            args.accelerator = "ddp" if torch.cuda.device_count() > 1 else "dp"
        else:
            args.accelerator = "ddp"


def main(args: Namespace):
    _sanity_check(args)
    _load_default_args(args)

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        if os.path.exists(args.model_name_or_path):
            args.output_dir = args.model_name_or_path
        else:
            args.output_dir = "./results"
            os.makedirs(args.output_dir, exist_ok=True)
        # args.output_dir = os.path.join(
        #     "./results",
        #     f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        # )

    if args.do_train:
        args.output_dir = os.path.join(args.output_dir, _get_default_output_dir(args))

    model = GLUETransformer(args)

    extra_callbacks = []
    if args.do_train and args.patience > 0:
        extra_callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_metric",
                min_delta=args.min_delta,
                patience=args.patience,
                verbose=False,
                mode="max",
            )
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir,
        prefix="checkpoint",
        monitor="val_metric",
        mode="max",
        save_top_k=1,
    )

    augmentation_on = args.teacher_name_or_path is not None and args.num_augments > 0
    trainer_kwargs = dict(
        reload_dataloaders_every_epoch=augmentation_on,
        profiler="simple",
    )
    # if is_ddp_enabled(args.distributed_backend, args.gpus):
    #     trainer_kwargs["replace_sampler_ddp"] = False

    logger = (
        pl_loggers.TensorBoardLogger(
            save_dir=args.output_dir,
            name="train_logs",
            default_hp_metric=False,
        )
        if args.do_train
        else False
    )

    trainer = generic_train(
        model,
        args,
        logger,
        extra_callbacks,
        checkpoint_callback,
        weights_summary="top",
        **trainer_kwargs,
    )

    # Optionally, predict on dev set and write to output_dir
    if args.do_eval or args.do_infer or args.do_test:
        checkpoints = list(
            sorted(
                glob.glob(
                    os.path.join(args.output_dir, "checkpointepoch=*.ckpt"),
                    recursive=True,
                )
            )
        )
        if checkpoints:
            model = model.load_from_checkpoint(checkpoints[-1])

        if args.do_eval:
            trainer.test(model)

        if args.do_infer:
            trainer.test(model, model.get_dataloader("infer", args.eval_batch_size))

        if args.do_test:
            trainer.test(model, model.get_dataloader("test2", args.eval_batch_size))
