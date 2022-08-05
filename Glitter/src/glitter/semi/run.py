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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import transformers
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_info
from transformers import PreTrainedModel

from .dataset import SemiGLUEDataset, DynamicAugBatchTensor, BatchTensor, SemiFeaturesCollator
from ..modeling_consistency import GlitterForConsistency, ConsistencyHead
from ..glue.processors import (
    AugmentedInputExample,
    GLUEv2Processor,
)
from ..hf_utils import (
    get_model_inputs as hf_get_model_inputs,
)
from ..lightning_base import generic_train
from ..glue import GLUETransformer, SimpleBatch
from ..log_utils import set_global_logging_error

transformers.logging.set_verbosity_error()
set_global_logging_error(["tensorflow", "tensorboard", "urllib3.connectionpool"])
logger = logging.getLogger(__name__)


@dataclass
class GlitterBatch:
    batch_indices: torch.Tensor
    augmented_indices: torch.Tensor
    augmented_mask: torch.Tensor
    augmented_ranks: torch.Tensor
    augmented_input_ids: torch.Tensor
    augmented_attention_mask: Optional[torch.Tensor] = None
    augmented_token_types: Optional[torch.Tensor] = None

    def select_subset(self, attr: Union[str, Optional[torch.Tensor]], selected_indices) -> Optional[torch.Tensor]:
        if attr is not None and isinstance(attr, str):
            val = getattr(self, attr, None)
        else:
            val = attr

        return val[selected_indices] if val is not None else val

    @classmethod
    def sub_batches(cls, batch: Iterable[DynamicAugBatchTensor]) -> Dict[int, "GlitterBatch"]:
        return {i: GlitterBatch(*minibatch) for i, minibatch in enumerate(batch) if minibatch}


@dataclass
class SemiGLUEBatch:
    sup_batch: Optional[BatchTensor]
    unsup_orig_batch: Optional[BatchTensor]
    unsup_aug_batch: Optional[BatchTensor] = None
    glitter_batch: Optional[List[DynamicAugBatchTensor]] = None

    def as_obj(self, attr) -> Optional[SimpleBatch]:
        b = getattr(self, attr, None)
        return None if b is None else SimpleBatch(**b)

    @property
    def sup(self):
        return self.as_obj("sup_batch")

    @property
    def unsup_orig(self):
        return self.as_obj("unsup_orig_batch")

    @property
    def unsup_aug(self):
        return self.as_obj("unsup_aug_batch")


class SemiGLUETransformer(GLUETransformer):
    def __init__(self, hparams):
        super().__init__(hparams)

        if self._is_glitter_on():
            self.glitter_head = GlitterForConsistency(self.hparams.uda_softmax_temp, self.hparams.num_augments)
        else:
            self.glitter_head = None

        self.uda_head = ConsistencyHead(
            self.hparams.uda_softmax_temp,
            self.hparams.uda_confidence_threshold,
        )
        if self.hparams.update_interval > 0:
            self.fixed_copy = self._detach()
        else:
            self.fixed_copy = None

    def _detach(self) -> PreTrainedModel:
        model_class = type(self.model)
        detached_model = model_class(self.model.config)
        detached_model.load_state_dict(self.model.state_dict())

        for p in detached_model.parameters():
            p.requires_grad = False

        return detached_model

    def _is_glitter_on(self) -> bool:
        return not self.hparams.vanilla_augment

    def on_train_epoch_start(self):
        if self.current_epoch > self.hparams.augment_start_epoch > 0:
            if (
                self.hparams.update_interval > 0
                and (self.current_epoch - self.hparams.augment_start_epoch) % self.hparams.update_interval == 0
            ) or (self.current_epoch == self.hparams.augment_start_epoch):
                rank_zero_info(f"****** Epoch {self.current_epoch}, updating the fixed model...")
                self.fixed_copy.load_state_dict(self.model.state_dict())

    def training_step(self, batch, batch_idx):
        batch = SemiGLUEBatch(*batch)
        assert batch.sup_batch is not None or batch.unsup_orig_batch is not None

        if batch.sup_batch is not None:
            sup_loss = self.sup_training_step(batch.sup)
        else:
            sup_loss = torch.tensor(0.0, device=self.device)

        if batch.unsup_orig_batch is not None:
            unsup_orig = batch.unsup_orig
            orig_inputs = hf_get_model_inputs(
                self.config.model_type,
                unsup_orig.input_ids,
                unsup_orig.token_type_ids,
                unsup_orig.attention_mask,
            )

            with torch.no_grad():
                if self.fixed_copy is not None:
                    orig_logits = self.fixed_copy(**orig_inputs).logits
                else:
                    orig_logits = self(**orig_inputs).logits

            if self.hparams.vanilla_augment:
                assert batch.unsup_aug_batch is not None
                unsup_aug = batch.unsup_aug
                aug_inputs = hf_get_model_inputs(
                    self.config.model_type,
                    unsup_aug.input_ids,
                    unsup_aug.token_type_ids,
                    unsup_aug.attention_mask,
                    unsup_aug.labels,
                )
                aug_logits = self(**aug_inputs).logits
                uda_loss = self.uda_head(orig_logits, aug_logits)
            else:
                assert batch.glitter_batch is not None
                uda_loss = self.glitter_training_step(batch.glitter_batch, orig_logits)
        else:
            uda_loss = torch.tensor(0.0, device=self.device)
        self.log("train/uda_loss", uda_loss, prog_bar=True, logger=True)

        loss = sup_loss + self.hparams.alpha_uda * uda_loss
        self.log("train/loss", loss, prog_bar=False, logger=True)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log("train/lr", lr_scheduler.get_last_lr()[-1], prog_bar=True, logger=True)

        return loss

    def glitter_training_step(self, glitter_batch: List[DynamicAugBatchTensor], orig_logits: torch.Tensor):
        sub_batches = GlitterBatch.sub_batches(glitter_batch)

        aug_loss = 0.0

        for aug_rank, sub_batch in sub_batches.items():
            augmented_inputs = hf_get_model_inputs(
                self.config.model_type,
                sub_batch.augmented_input_ids,
                sub_batch.augmented_token_types,
                sub_batch.augmented_attention_mask,
            )

            aug_output = self(**augmented_inputs)

            glitter_output = self.glitter_head(
                orig_logits[sub_batch.batch_indices],
                aug_output.logits,
                aug_rank,
                sub_batch.augmented_mask,
                sub_batch.augmented_ranks,
            )

            if glitter_output.has_selected_ranks:
                self.logger.experiment.add_histogram(
                    "glitter/selected_ranks",
                    glitter_output.selected_ranks,
                    self.global_step,
                )

            aug_loss += self.uda_head(glitter_output.orig_logits, aug_output.logits[glitter_output.selected_neighbors])

        return aug_loss

    def sup_training_step(self, sup_batch: SimpleBatch):
        inputs = hf_get_model_inputs(
            self.config.model_type,
            sup_batch.input_ids,
            sup_batch.token_type_ids,
            sup_batch.attention_mask,
            sup_batch.labels,
        )
        output = self(**inputs)
        sup_loss = output.loss

        if self.hparams.tsa_schedule is not None:
            tsa_thresh = self.get_tsa_thresh(self.hparams.tsa_schedule, start=1.0 / output.logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh  # prob = exp(log_prob), prob > tsa_threshold
            loss_mask = torch.ones_like(sup_batch.labels, dtype=torch.float32) * (
                1 - larger_than_threshold.type(torch.float32)
            )
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.clamp(torch.sum(loss_mask, dim=-1), min=1.0)
            self.log("uda/tsa_threshold", tsa_thresh, logger=True)

        self.log("train/sup_loss", sup_loss, logger=True)
        return sup_loss

    def get_tsa_thresh(self, tsa, start, end) -> torch.Tensor:
        training_progress = torch.tensor(float(self.global_step) / float(self.total_steps()))
        if tsa == "linear":
            threshold = training_progress
        elif tsa == "exp":
            scale = 5
            threshold = torch.exp((training_progress - 1) * scale)
        elif tsa == "log":
            scale = 5
            threshold = 1 - torch.exp((-training_progress) * scale)
        else:
            raise KeyError(f"Unknown tsa algorithm: {tsa}")

        output = threshold * (end - start) + start
        return output

    def total_steps(self) -> int:
        if self.hparams.num_augments > 0:
            num_devices = max(1, self.get_number_of_gpus())
            effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
            effective_dataset_size = len(self.train_dataset) / effective_batch_size
            effective_augmented_dataset_size = len(self.augmented_train_dataset) / effective_batch_size
            return effective_dataset_size * self.hparams.augment_start_epoch + effective_augmented_dataset_size * (
                self.hparams.max_epochs - self.hparams.augment_start_epoch
            )
        else:
            return super().total_steps()

    def setup(self, mode):
        super().setup(mode)

        if mode == "fit":
            if self.hparams.num_augments > 0:
                args = self.hparams
                processor = GLUEv2Processor(args.task, num_augments=-1)
                self._filter_augments(processor.get_train_examples(args.data_dir))
                self.augmented_train_dataset = self.get_dataset("augmented_train")
                rank_zero_info(
                    f"****** Size of supervised data: {len(self.augmented_train_dataset.sup_indices)} "
                    f"out of {len(self.augmented_train_dataset.examples)}"
                )

    def get_dataset(self, mode: str):
        is_train = mode.endswith("train")
        if not is_train:
            self.hparams.test_mode = mode

        args = self.hparams

        processor = GLUEv2Processor(
            args.task,
            num_augments=-1,
        )
        self.labels = processor.get_labels()

        if mode in ("dev", "test"):
            examples = processor.get_dev_examples(args.data_dir)
        elif mode == "infer":
            examples = processor.get_test_examples(args.data_dir)
        elif mode == "test2":
            examples = processor.get_labelled_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)

        return SemiGLUEDataset(
            examples,
            self.labels,
            args.glue_output_mode,
            args.max_seq_length,
            self.tokenizer,
            args.num_augments if mode == "augmented_train" else 0,
            self.hparams.sup_ratio,
            self.hparams.sup_size,
            torch.load(self._get_cached_allowed_indices_file()) if mode == "augmented_train" else None,
            self.hparams.vanilla_augment,
            self.hparams.padding,
        )

    def get_collator(self, mode: str):
        return SemiFeaturesCollator(
            self.tokenizer,
            pad_to_multiple_of=self.hparams.pad_to_multiple_of,
            group_features=mode == "augmented_train" and self._is_glitter_on(),
            num_augment_ranks=(self.hparams.num_augments + 1)
            if mode == "augmented_train" and self._is_glitter_on()
            else 0,
        )

    def train_dataloader(self):
        if self.hparams.num_augments > 0 and self.current_epoch >= self.hparams.augment_start_epoch:
            return self.get_dataloader(
                "augmented_train", self.hparams.train_batch_size, shuffle=True, dataset=self.augmented_train_dataset
            )

        return super().train_dataloader()

    def validation_step(self, batch, batch_idx):
        batch = SemiGLUEBatch(*batch).sup
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

    def _get_cached_allowed_indices_file(self):
        return os.path.join(self.hparams.output_dir, f".allowed_indices.bin")

    def _filter_augments(
        self,
        examples: List[AugmentedInputExample],
    ):
        cache_file = self._get_cached_allowed_indices_file()

        if os.path.exists(cache_file):
            return

        num_augmented_examples = 0
        num_retained_augmented_examples = 0
        allowed_indices = []

        rank_zero_info("*** Filtering based on distance constraints")

        for i, ex in enumerate(examples):
            num_augmented_examples += len(ex.augmented_examples)
            n_augs = self.hparams.num_aug_candidates if self._is_glitter_on() else self.hparams.num_augments
            retained_indices = np.arange(
                min(n_augs, len(ex.augmented_examples)),
            )
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
    def add_model_specific_args(parser):
        GLUETransformer.add_model_specific_args(parser)

        parser.add_argument(
            "--num_augments",
            default=0,
            type=int,
            help="Number of augmentations per train samples used during training",
        )
        parser.add_argument(
            "--num_aug_candidates",
            default=8,
            type=int,
            help="Number of candidates per train samples to select augmentation samples from",
        )
        parser.add_argument(
            "--augment_start_epoch",
            default=0,
            type=int,
            help="First epoch (starting from zero) in which augmented data incorporated into training. "
            "Earlier epochs are warm-up and run standard KD.",
        )
        parser.add_argument(
            "--update_interval",
            default=1,
            type=int,
            help="Update interval for the fixed copy",
        )
        parser.add_argument(
            "--vanilla_augment",
            action="store_true",
            default=False,
            help="Ignores Glitter and naively include the whole augmented data in training. Only when `num_augments` > 0",
        )

        # Consistency-training (CT) parameters (optional)
        parser.add_argument(
            "--alpha_uda",
            default=1.0,
            type=float,
            help="Coefficient on the UDA loss. Defaults to 1 following the original repo.",
        )
        parser.add_argument(
            "--tsa_schedule",
            default=None,
            choices=("linear", "log", "exp"),
            type=str,
            help="Anneal schedule of training signal annealing.",
        )
        parser.add_argument(
            "--uda_softmax_temp",
            default=1.0,
            type=float,
            help="The temperature of the Softmax when making prediction on unlabeled examples.",
        )
        parser.add_argument(
            "--uda_confidence_threshold",
            default=-1.0,
            type=float,
            help="The threshold on predicted probability on unsupervised data. "
            "If set, UDA loss will only be calculated on unlabeled examples whose largest probability "
            "is larger than the threshold",
        )
        parser.add_argument(
            "--sup_ratio",
            default=1.0,
            type=float,
            help="Ratio of labelled data over the entire dataset",
        )
        parser.add_argument(
            "--sup_size",
            default=0,
            type=int,
            help="Size of labelled data (Overrides `--sup_ratio`)",
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
        f"_epsadam{args.adam_epsilon}"
        f"_b2adam{args.adam_beta2}"
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

    if args.sup_size > 0:
        arg_summary += f"_supsz{args.sup_size}"
    elif args.sup_ratio > 0:
        arg_summary += f"_supr{args.sup_ratio}"

    if args.num_augments > 0:
        if args.tsa_schedule is not None:
            arg_summary += f"_tsa{args.tsa_schedule.upper()}"

        if args.uda_confidence_threshold > 0.0:
            arg_summary += f"_confd{args.uda_confidence_threshold}"

        if args.uda_softmax_temp != 1.0:
            arg_summary += f"_temp{args.uda_softmax_temp}"
        arg_summary += f"_uda{args.alpha_uda}"

        arg_summary += f"_naug{args.num_augments}"
        if not args.vanilla_augment:
            arg_summary += f"of{args.num_aug_candidates}"

        arg_summary += f"_upd{args.update_interval}"
        if args.augment_start_epoch > 0:
            arg_summary += f"_augst{args.augment_start_epoch}"

        if args.vanilla_augment:
            arg_summary += "_vanilla"

        arg_summary += "_consistency"

    return arg_summary


def _sanity_check(args):
    if args.do_train:
        assert 0.0 <= args.warmup_ratio <= 1.0, "`--warmup_ratio` must be in [0, 1]"

        if not args.vanilla_augment:
            assert args.uda_softmax_temp > 0, "`--uda_softmax_temp` must be greater than 0"
            assert (
                args.num_augments < args.num_aug_candidates
            ), f"`--num_augments` ({args.num_augments}) must be less than `--num_aug_candidates` ({args.num_aug_candidates})"

        assert not (
            args.do_infer or args.do_test or args.do_eval
        ), "`--do_infer`, `--do_test` and `--do_eval` cannot be done if training is enabled"


def _load_default_args(args):
    if args.available_gpus is not None:
        available_gpus = [g for g in args.available_gpus if g < torch.cuda.device_count()]
        if len(available_gpus) < len(args.available_gpus):
            logger.warning(f"Only these GPUs are available: {available_gpus}")
            args.available_gpus = available_gpus
        if args.gpus > len(args.available_gpus) or args.gpus < 0:
            args.gpus = len(args.available_gpus)

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

    if args.do_train:
        args.output_dir = os.path.join(args.output_dir, _get_default_output_dir(args))

    model = SemiGLUETransformer(args)

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

    if args.save_last:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            save_last=True,
        )
    elif args.val_check_interval > 0:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            monitor="val_metric",
            mode="max",
            save_top_k=1,
        )
    else:
        checkpoint_callback = False

    trainer_kwargs = dict(
        reload_dataloaders_every_epoch=args.num_augments > 0,
        profiler="simple",
    )

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
        if args.do_train:
            if args.save_last:
                model_checkpoint_path = checkpoint_callback.last_model_path
            elif checkpoint_callback is not None:
                model_checkpoint_path = checkpoint_callback.best_model_path
            else:
                model_checkpoint_path = None

            if model_checkpoint_path is not None:
                model = model.load_from_checkpoint(model_checkpoint_path)

        if args.do_eval:
            trainer.test(model)

        if args.do_infer:
            trainer.test(model, model.get_dataloader("infer", args.eval_batch_size))

        if args.do_test:
            trainer.test(model, model.get_dataloader("test2", args.eval_batch_size))
