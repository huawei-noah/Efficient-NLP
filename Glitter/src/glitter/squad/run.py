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

import pytorch_lightning as pl
import torch
import transformers
import yaml
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from .base_model import get_model_inputs, SQuADTransformer, SQuADBatch
from .cached import cached_model_outputs
from .dataset import convert_to_features, SquadDataset, DataCollatorWithPadding
from .processors import SquadProcessor
from ..lightning_base import generic_train
from ..log_utils import set_global_logging_error
from ..modeling_glitter import GlitterForQuestionAnswering
from ..modeling_kd import KDForQuestionAnswering

transformers.logging.set_verbosity_error()
set_global_logging_error(["tensorflow", "tensorboard", "urllib3.connectionpool"])
logger = logging.getLogger(__name__)


class GlitterSQuADTransformer(SQuADTransformer):
    def __init__(self, hparams):
        super().__init__(hparams)

        if self.hparams.do_train:
            self.train_examples = SquadProcessor().get_train_examples(self.hparams.data_dir)
        else:
            self.train_examples = None

        self.teacher_saved_output = None
        if self.hparams.do_train and self.hparams.teacher_name_or_path is not None:
            self.teacher_saved_output = cached_model_outputs(
                self.train_examples,
                self.hparams.teacher_name_or_path,
                self.hparams.cache_batch_size or self.hparams.train_batch_size,
                self.hparams.max_seq_length,
                self.hparams.doc_stride,
                cache_dir=self.hparams.cache_dir,
                dataset_name=self.hparams.dataset_name,
                num_workers=self.hparams.num_workers,
            )

        if self._is_distillation_on():
            self.kd_head = KDForQuestionAnswering(
                self.teacher_saved_output.start_logits, self.teacher_saved_output.end_logits, self.hparams.temperature
            )
        else:
            self.kd_head = None

        if self._is_glitter_on():
            self.glitter_head = GlitterForQuestionAnswering(
                self.teacher_saved_output.start_logits,
                self.teacher_saved_output.end_logits,
                self.hparams.temperature,
                self.hparams.num_augments,
            )
        else:
            self.glitter_head = None

    def _set_defaults(self, hparams):
        hparams.padding = "max_length"

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

    def _num_augments_to_load(self) -> int:
        if not self._is_augmentation_on():
            return 0

        return self.hparams.num_augments if self._is_vanilla_augment_on() else self.hparams.num_aug_candidates

    def _prepare_data(self, stage: str) -> None:
        self.feature_indices_to_flatten_map = None

        if stage == "fit":
            if not self._is_augmentation_on() or self.hparams.augment_start_epoch > 0:
                self.train_features = list(
                    convert_to_features(
                        self.train_examples,
                        self.tokenizer,
                        "train",
                        self.hparams.max_seq_length,
                        self.hparams.doc_stride,
                    )
                )
            else:
                self.train_features = None

            if self._is_distillation_on():
                self.feature_indices_to_flatten_map = self.teacher_saved_output.index_map

            if self._is_augmentation_on():
                self.augmented_train_features = list(
                    convert_to_features(
                        self.train_examples,
                        self.tokenizer,
                        "train",
                        self.hparams.max_seq_length,
                        self.hparams.doc_stride,
                        self.hparams.augment_with_labels,
                        self._num_augments_to_load(),
                    )
                )
                self.augmented_train_dataset = self.get_dataset("augmented_train")
            else:
                self.augmented_train_features = None
                self.augmented_train_dataset = None

            self.train_dataset = self.get_dataset("train")
            self.valid_dataset = self.get_dataset("dev")
        elif self.hparams.do_eval:
            self.test_dataset = self.get_dataset("dev")
        elif self.hparams.do_test:
            self.test_dataset = self.get_dataset("test")

    def get_dataloader(self, mode: str, batch_size: int, shuffle: bool = False, dataset=None):
        if dataset is None:
            dataset = self.get_dataset(mode)

        if mode in ("test2", "infer", "test"):
            self.test_dataset = dataset

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.get_collator(mode),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def training_step(self, batch, batch_idx):
        sub_batches = SQuADBatch.sub_batches(batch)

        # skip a batch
        if 0 not in sub_batches and (self._is_glitter_on() and self.current_epoch < self.hparams.augment_start_epoch):
            return None

        loss = 0.0
        batch_start_logits = None
        batch_end_logits = None
        example_indices = None
        answer_starts = None
        answer_ends = None

        for aug_rank, sub_batch in sub_batches.items():
            inputs = get_model_inputs(
                self.config.model_type,
                sub_batch.input_ids,
                sub_batch.token_type_ids,
                sub_batch.attention_mask,
                sub_batch.start_positions,
                sub_batch.end_positions,
            )

            if aug_rank == 0:
                noaug_loss, start_logits, end_logits = self.standard_training_step(inputs, sub_batch.cache_indices)

                batch_start_logits = start_logits[sub_batch.start_positions >= 0, :]
                batch_end_logits = end_logits[sub_batch.end_positions >= 0, :]
                example_indices = sub_batch.example_indices[sub_batch.start_positions >= 0]
                answer_starts = sub_batch.start_positions[sub_batch.start_positions >= 0]
                answer_ends = sub_batch.end_positions[sub_batch.end_positions >= 0]

                loss += noaug_loss

                if not self._is_distillation_on():
                    break
            else:
                if self.current_epoch < self.hparams.augment_start_epoch:
                    continue

                glitter_args = dict(inputs)
                glitter_args.pop("start_positions", None)
                glitter_args.pop("end_positions", None)
                output = self(**glitter_args)

                glitter_output = self.glitter_head(
                    output.start_logits,
                    output.end_logits,
                    aug_rank,
                    sub_batch.augment_mask,
                    sub_batch.cache_indices,
                    nn_ranks=sub_batch.augment_ranks,
                )

                if not glitter_output.is_empty:
                    if glitter_output.has_selected_ranks:
                        self.logger.experiment.add_histogram(
                            "glitter/selected_ranks",
                            glitter_output.selected_ranks,
                            self.global_step,
                        )

                    aug_output = self(
                        input_ids=sub_batch.select_subset("input_ids", glitter_output.selected_neighbors),
                        attention_mask=sub_batch.select_subset(
                            "attention_mask",
                            glitter_output.selected_neighbors,
                        ),
                        token_type_ids=sub_batch.select_subset(
                            inputs.get("token_type_ids", None),
                            glitter_output.selected_neighbors,
                        ),
                    )

                    loss_aug = self.kd_head(
                        aug_output.start_logits,
                        aug_output.end_logits,
                        teacher_start_logits=glitter_output.teacher_start_logits,
                        teacher_end_logits=glitter_output.teacher_end_logits,
                    )
                    self.log("train/aug_loss", loss_aug, logger=True)

                    loss += self.hparams.alpha_aug * loss_aug

        if loss is not None:
            self.log("train/loss", loss, prog_bar=False, logger=True)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log("train/lr", lr_scheduler.get_last_lr()[-1], prog_bar=True, logger=True)

        return dict(
            loss=loss,
            start_logits=batch_start_logits,
            end_logits=batch_end_logits,
            answer_starts=answer_starts,
            answer_ends=answer_ends,
            example_indices=example_indices,
        )

    def standard_training_step(self, inputs, cache_indices):
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
                output.start_logits,
                output.end_logits,
                cache_indices,
            )
            self.log("train/kd_loss", kd_loss, logger=True)
            loss = self.hparams.alpha_ce * kd_loss + self.hparams.alpha_true * cls_loss
        else:
            loss = cls_loss

        return loss, output.start_logits, output.end_logits

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

    def get_dataset(self, mode: str):
        if not mode.endswith("train"):
            self.hparams.test_mode = mode

        args = self.hparams

        processor = SquadProcessor()

        if mode in ("dev", "test"):
            examples = processor.get_dev_examples(args.data_dir)
        elif mode == "infer":
            examples = processor.get_test_examples(args.data_dir)
        elif mode == "test2":
            examples = processor.get_labelled_test_examples(args.data_dir)
        else:
            examples = self.train_examples

        features = None
        if mode == "train":
            features = self.train_features
        elif mode == "augmented_train":
            features = self.augmented_train_features

        return SquadDataset(
            examples,
            args.max_seq_length,
            self.tokenizer,
            args.doc_stride,
            mode,
            features,
            args.num_augments if mode == "augmented_train" and self._is_augmentation_on() else 0,
            self.feature_indices_to_flatten_map if mode == "augmented_train" and self._is_distillation_on() else None,
            vanilla_augment=mode == "augmented_train" and self._is_vanilla_augment_on(),
            padding=self.hparams.padding,
        )

    def get_collator(self, mode: str):
        return DataCollatorWithPadding(
            self.tokenizer,
            pad_to_multiple_of=self.hparams.pad_to_multiple_of,
            padding=self.hparams.padding,
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

    @staticmethod
    def add_model_specific_args(parser):
        SQuADTransformer.add_model_specific_args(parser)

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
            default=5.0,
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
    arg_summary = f"qa_{Path(args.model_name_or_path).name}"

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

    # if args.padding is not None:
    #     arg_summary += f"_pa{args.padding[:3]}"
    if args.pad_to_multiple_of is not None:
        arg_summary += f"_pam{args.pad_to_multiple_of}"

    if args.fp16:
        arg_summary += "_amp"

    if args.deterministic:
        arg_summary += "_det"

    if args.version_2_with_negative:
        arg_summary += "_v2"

    arg_summary += f"_stride{args.doc_stride}"

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

    model = GlitterSQuADTransformer(args)

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
