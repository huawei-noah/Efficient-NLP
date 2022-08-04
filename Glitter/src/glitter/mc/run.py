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
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_info
from torch.nn import CrossEntropyLoss

from .base_model import get_model_inputs, MultipleChoiceTransformer
from .cached import cached_model_outputs
from .dataset import MultiChoiceExample, MultiChoiceDataset, DataCollatorWithPadding, MultiChoiceBatch
from ..lightning_base import generic_train
from ..log_utils import set_global_logging_error
from ..modeling_glitter import GlitterForMultipleChoice
from ..modeling_kd import KDForMultipleChoice

transformers.logging.set_verbosity_error()
set_global_logging_error(["tensorflow", "tensorboard", "urllib3.connectionpool"])
logger = logging.getLogger(__name__)


class GlitterMultipleChoiceTransformer(MultipleChoiceTransformer):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.teacher_saved_output = None
        if self.hparams.teacher_name_or_path is not None:
            self.teacher_saved_output = cached_model_outputs(
                self.hparams.teacher_name_or_path,
                self.hparams.task,
                self.hparams.data_dir,
                self.hparams.cache_batch_size or self.hparams.train_batch_size,
                self.hparams.max_seq_length,
                cache_dir=self.hparams.cache_dir,
                dataset_name=self.hparams.dataset_name,
                num_workers=self.hparams.num_workers,
            )

        if self._is_distillation_on():
            self.kd_head = KDForMultipleChoice(
                self.teacher_saved_output.logits, self.hparams.temperature
            )
        else:
            self.kd_head = None

        if self._is_glitter_on():
            self.glitter_head = GlitterForMultipleChoice(
                self.teacher_saved_output.logits,
                self.hparams.temperature,
                self.hparams.num_augments,
            )
        else:
            self.glitter_head = None

    def _set_defaults(self, hparams):
        super()._set_defaults(hparams)

        if hparams.teacher_name_or_path is not None:
            if hparams.alpha_ce is None:
                hparams.alpha_ce = 1 - hparams.alpha_true

            if hparams.num_augments > 0 and hparams.alpha_aug is None:
                hparams.alpha_aug = hparams.alpha_ce

    def _is_distillation_on(self) -> bool:
        return self.teacher_saved_output is not None

    def _is_augmentation_on(self) -> bool:
        return self._is_distillation_on() and self.hparams.num_augments > 0

    def _is_glitter_on(self) -> bool:
        return self._is_augmentation_on() and not self.hparams.vanilla_augment

    def _is_vanilla_augment_on(self) -> bool:
        return self._is_augmentation_on() and self.hparams.vanilla_augment

    def training_step(self, batch, batch_idx):
        sub_batches = MultiChoiceBatch.sub_batches(batch)

        # skip a batch
        if 0 not in sub_batches and (self._is_glitter_on() and self.current_epoch < self.hparams.augment_start_epoch):
            return None

        loss = 0.0
        batch_logits = None
        example_indices = None
        batch_labels = None

        for aug_rank, sub_batch in sub_batches.items():
            if aug_rank == 0:
                inputs = get_model_inputs(
                    self.config.model_type,
                    sub_batch.input_ids,
                    sub_batch.token_type_ids,
                    sub_batch.attention_mask,
                    sub_batch.labels,
                )

                output = self.standard_training_step(
                    inputs, sub_batch.example_indices, sub_batch.augmented_indices
                )

                batch_logits = output["logits"][sub_batch.labels >= 0, :]
                example_indices = sub_batch.example_indices[sub_batch.labels >= 0]
                batch_labels = sub_batch.labels[sub_batch.labels >= 0]

                loss += output["loss"]

                if not self._is_distillation_on():
                    break
            else:
                if self.current_epoch < self.hparams.augment_start_epoch:
                    continue

                glitter_args = get_model_inputs(
                    self.config.model_type,
                    sub_batch.augmented_input_ids,
                    sub_batch.augmented_token_types,
                    sub_batch.augmented_attention_mask,
                )
                stu_logits = self(**glitter_args).logits

                glitter_output = self.glitter_head(
                    stu_logits,
                    aug_rank,
                    sub_batch.augment_mask,
                    sub_batch.example_indices,
                    nn_ranks=sub_batch.augment_ranks,
                    augmented_indices=sub_batch.augmented_indices,
                )

                if not glitter_output.is_empty:
                    if glitter_output.has_selected_ranks:
                        self.logger.experiment.add_histogram(
                            "glitter/selected_ranks",
                            glitter_output.selected_ranks,
                            self.global_step,
                        )

                    if self.hparams.augment_with_labels:
                        loss_fct = CrossEntropyLoss()
                        cls_loss_aug = loss_fct(
                            stu_logits[glitter_output.selected_neighbors].view(-1, self.num_labels),
                            sub_batch.labels[sub_batch.augment_mask[glitter_output.selected_neighbors]].view(-1),
                        )
                        self.log("train/aug_cls_loss", cls_loss_aug, logger=True)
                    else:
                        cls_loss_aug = 0.0

                    kd_loss_aug = self.kd_head(
                        stu_logits[glitter_output.selected_neighbors],
                        glitter_output.teacher_logits,
                    )
                    loss_aug = kd_loss_aug + cls_loss_aug
                    self.log("train/aug_loss", loss_aug, logger=True)

                    loss += self.hparams.alpha_aug * loss_aug

        if loss is not None:
            self.log("train/loss", loss, prog_bar=False, logger=True)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log("train/lr", lr_scheduler.get_last_lr()[-1], prog_bar=True, logger=True)

        return dict(
            loss=loss,
            logits=batch_logits,
            labels=batch_labels,
            example_indices=example_indices,
        )

    def standard_training_step(self, inputs, example_indices, augmented_indices):
        output = self(**inputs)

        device = inputs["input_ids"].device
        cls_loss = output.loss

        # for auxiliary samples that do not have labels
        if cls_loss is None:
            cls_loss = torch.tensor(0.0, device=device)
        else:
            self.log("train/cls_loss", cls_loss, logger=True)

        if self._is_distillation_on():
            kd_loss = self.kd_head(
                output.logits,
                example_indices=example_indices,
                augmented_indices=augmented_indices,
            )
            self.log("train/kd_loss", kd_loss, logger=True)
            loss = self.hparams.alpha_ce * kd_loss + self.hparams.alpha_true * cls_loss
        else:
            loss = cls_loss

        return dict(logits=output.logits, loss=loss)

    def total_steps(self) -> int:
        if self._is_augmentation_on():
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
        if mode == "fit" and self._is_augmentation_on():
            args = self.hparams

            self._filter_augments(self.data_processor.get_train_examples(args.data_dir))
            self.augmented_train_dataset = self.get_dataset("augmented_train")

    def get_dataset(self, mode: str):
        if not mode.endswith("train"):
            self.hparams.test_mode = mode

        args = self.hparams

        processor = self.data_processor

        if mode in ("dev", "test"):
            examples = processor.get_dev_examples(args.data_dir)
        elif mode == "infer":
            examples = processor.get_test_examples(args.data_dir)
        elif mode == "test2":
            examples = processor.get_labelled_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)

        return MultiChoiceDataset(
            examples,
            args.max_seq_length,
            self.tokenizer,
            args.num_augments if mode == "augmented_train" else 0,
            torch.load(self._get_cached_allowed_indices_file()) if mode == "augmented_train" else None,
            vanilla_augment=mode == "augmented_train" and self._is_vanilla_augment_on(),
            augment_with_label=mode == "augmented_train" and self._is_augmentation_on() and self.hparams.augment_with_labels,
            padding=self.hparams.padding,
        )

    def get_collator(self, mode: str):
        return DataCollatorWithPadding(
            self.tokenizer,
            pad_to_multiple_of=self.hparams.pad_to_multiple_of,
        )

    def train_dataloader(self):
        if self._is_augmentation_on() and self.current_epoch >= self.hparams.augment_start_epoch:
            return self.get_dataloader(
                "augmented_train", self.hparams.train_batch_size, shuffle=True, dataset=self.augmented_train_dataset
            )

        return super().train_dataloader()

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False, dataset=self.valid_dataset)

    def _get_cached_allowed_indices_file(self):
        return os.path.join(self.hparams.output_dir, f".allowed_indices.bin")

    def _filter_augments(
        self,
        examples: List[MultiChoiceExample],
    ):
        cache_file = self._get_cached_allowed_indices_file()

        if os.path.exists(cache_file):
            return

        hidden_states = self.teacher_saved_output.hidden_states
        num_augmented_examples = 0
        num_retained_augmented_examples = 0
        allowed_indices = []

        cos = nn.CosineSimilarity()

        rank_zero_info("*** Filtering based on distance constraints")

        for i, ex in enumerate(examples):
            num_augmented_examples += len(ex.augmented_examples)
            n_augs = self.hparams.num_aug_candidates if self._is_glitter_on() else self.hparams.num_augments

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
    def add_model_specific_args(parser):
        MultipleChoiceTransformer.add_model_specific_args(parser)

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
        parser.add_argument(
            "--augment_start_epoch",
            default=0,
            type=int,
            help="First epoch (starting from zero) in which augmented data incorporated into training. "
            "Earlier epochs are warm-up and run standard KD.",
        )
        parser.add_argument(
            "--augment_with_labels",
            default=False,
            action="store_true",
            help="Whether to take labels into account for augmentation",
        )
        parser.add_argument(
            "--vanilla_augment",
            action="store_true",
            default=False,
            help="Ignores Glitter and naively include the whole augmented data in training. Only when `num_augments` > 0",
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
            default=None,
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

    if args.deterministic:
        arg_summary += "_det"

    if args.num_augments > 0:
        arg_summary += f"_naug{args.num_augments}"
        if not args.vanilla_augment:
            arg_summary += f"of{args.num_aug_candidates}"

    if args.teacher_name_or_path is not None:
        arg_summary += f"_tau{args.temperature}"
        if args.alpha_ce is not None:
            arg_summary += f"_ce{args.alpha_ce}"
        arg_summary += f"_tru{args.alpha_true}"

        if args.num_augments > 0:
            if args.alpha_aug is not None:
                arg_summary += f"_aug{args.alpha_aug}"

            if args.augment_start_epoch > 0:
                arg_summary += f"_augst{args.augment_start_epoch}"

            if args.augment_with_labels:
                arg_summary += "_auglbl"

            if args.vanilla_augment:
                arg_summary += "_vanilla"

        arg_summary += "_distilled"

    return arg_summary


def _sanity_check(args):
    if args.do_train:
        assert 0.0 <= args.warmup_ratio <= 1.0, "`--warmup_ratio` must be in [0, 1]"

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

    model = GlitterMultipleChoiceTransformer(args)

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

    augmentation_on = args.teacher_name_or_path is not None and args.num_augments > 0
    trainer_kwargs = dict(
        reload_dataloaders_every_epoch=augmentation_on,
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
