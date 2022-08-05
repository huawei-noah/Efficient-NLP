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
from typing import Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_info
from torch.nn import CrossEntropyLoss

from .base_model import GLUETransformer, GLUEBatch
from .dataset import (
    AugmentedInputExample,
    GLUEv2Dataset,
)
from .processors import GLUEv2Processor
from ..data import FeaturesCollatorWithPadding
from .cached import cached_model_outputs
from ..hf_utils import (
    get_model_inputs as hf_get_model_inputs,
)
from ..lightning_base import generic_train
from ..log_utils import set_global_logging_error
from ..modeling_glitter import GlitterForSequenceClassification, RandomGlitterForSequenceClassification
from ..modeling_kd import KDForSequenceClassification

transformers.logging.set_verbosity_error()
set_global_logging_error(["tensorflow", "tensorboard", "urllib3.connectionpool"])
logger = logging.getLogger(__name__)


class GlitterGLUETransformer(GLUETransformer):
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
            self.kd_head = KDForSequenceClassification(self.teacher_saved_output.logits, self.hparams.temperature)
        else:
            self.kd_head = None

        if self._is_glitter_on():
            if self.hparams.random_glitter:
                self.glitter_head = RandomGlitterForSequenceClassification(self.teacher_saved_output.logits)
            else:
                self.glitter_head = GlitterForSequenceClassification(
                    self.teacher_saved_output.logits,
                    self.hparams.temperature,
                    self.hparams.num_augments,
                )
        else:
            self.glitter_head = None

    def _set_defaults(self, hparams):
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
                    f"glitter/{prefix}_filtered_ranks",
                    filtered_ranks,
                    self.global_step,
                )

            self.log(
                f"glitter/num_{prefix}_filtered_neighbors",
                filtered_ranks.nelement(),
                logger=True,
            )
            return filtered_ranks.nelement()

        return None

    def training_step(self, batch, batch_idx):
        minibatches = GLUEBatch.get_minibatches(batch)

        # skip a batch
        if 0 not in minibatches and (self._is_glitter_on() and self.current_epoch < self.hparams.augment_start_epoch):
            return None

        loss = 0.0
        example_logits = None
        example_indices = None
        example_labels = None

        for aug_rank, minibatch in minibatches.items():
            inputs = hf_get_model_inputs(
                self.config.model_type,
                minibatch.input_ids,
                minibatch.token_type_ids,
                minibatch.attention_mask,
                None
                if self._is_vanilla_augment_on() and self.hparams.glue_output_mode == "regression"
                else minibatch.labels,
            )

            if aug_rank == 0:
                output = self.standard_training_step(
                    inputs, minibatch.labels, minibatch.example_indices, minibatch.augmented_indices
                )

                example_logits = output["logits"][minibatch.labels >= 0, :]
                example_indices = minibatch.example_indices[minibatch.labels >= 0]
                example_labels = minibatch.labels[minibatch.labels >= 0]

                loss += output["loss"]

                if not self._is_distillation_on():
                    break
            else:
                if self.current_epoch < self.hparams.augment_start_epoch:
                    continue

                assert self._is_glitter_on() and minibatch.augmented_input_ids is not None

                augmented_inputs = hf_get_model_inputs(
                    self.config.model_type,
                    minibatch.augmented_input_ids,
                    minibatch.augmented_token_types,
                    minibatch.augmented_attention_mask,
                )

                stu_logits = self(**augmented_inputs).logits

                glitter_output = self.glitter_head(
                    stu_logits,
                    aug_rank,
                    minibatch.augmented_mask,
                    minibatch.example_indices,
                    nn_ranks=minibatch.augmented_ranks,
                    augmented_indices=minibatch.augmented_indices,
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
                            minibatch.labels[minibatch.augmented_mask[glitter_output.selected_neighbors]].view(-1),
                        )
                        self.log("train/aug_cls_loss", cls_loss_aug, logger=True)
                    else:
                        cls_loss_aug = 0.0

                    kd_loss_aug = self.kd_head(
                        stu_logits[glitter_output.selected_neighbors],
                        glitter_output.teacher_logits,
                        weights=glitter_output.weights,
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
            example_logits=example_logits,
            example_labels=example_labels,
            example_indices=example_indices,
        )

    def standard_training_step(self, inputs, labels, example_indices, augmented_indices):
        output = self(**inputs)

        device = inputs["input_ids"].device

        if self._is_vanilla_augment_on() and self.hparams.glue_output_mode == "regression":
            loss_fct = nn.MSELoss()
            if (labels != -100.0).int().sum() > 0:
                cls_loss = loss_fct(
                    output.logits[labels != -100.0, ...].view(-1),
                    labels[labels != -100.0].view(-1),
                )
            else:
                cls_loss = torch.tensor(0.0, device=device)
        else:
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

    def _prepare_data(self, stage) -> None:
        if stage == "fit":
            if self._is_augmentation_on():
                args = self.hparams

                processor = GLUEv2Processor(args.task, num_augments=-1)
                self._filter_augments(processor.get_train_examples(args.data_dir))
                self.augmented_train_dataset = self.get_dataset("augmented_train")
            self.train_dataset = self.get_dataset("train")
            self.valid_dataset = self.get_dataset("dev")

    def get_dataset(self, mode: str):
        if not mode.endswith("train"):
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

        return GLUEv2Dataset(
            examples,
            self.labels,
            args.glue_output_mode,
            args.max_seq_length,
            self.tokenizer,
            args.num_augments if mode == "augmented_train" else 0,
            torch.load(self._get_cached_allowed_indices_file()) if mode == "augmented_train" else None,
            vanilla_augment=mode == "augmented_train" and self._is_vanilla_augment_on(),
            padding=self.hparams.padding,
        )

    def get_collator(self, mode: str):
        return FeaturesCollatorWithPadding(
            self.tokenizer,
            pad_to_multiple_of=self.hparams.pad_to_multiple_of,
            group_features=mode == "augmented_train" and self._is_glitter_on(),
            num_augment_ranks=(self.hparams.num_augments + 1) if mode == "augmented_train" else 0,
        )

    def train_dataloader(self):
        if self._is_augmentation_on() and self.current_epoch >= self.hparams.augment_start_epoch:
            return self.get_dataloader(
                "augmented_train", self.hparams.train_batch_size, shuffle=True, dataset=self.augmented_train_dataset
            )

        return super().train_dataloader()

    def validation_step(self, batch, batch_idx):
        batch = GLUEBatch.get_minibatches(batch)[0]
        inputs = hf_get_model_inputs(
            self.config.model_type, batch.input_ids, batch.token_type_ids, batch.attention_mask, batch.labels
        )
        outputs = self(**inputs)

        preds = outputs.logits

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
        return os.path.join(self.hparams.output_dir, f".allowed_indices_{self.current_epoch}.bin")

    def _filter_augments(
        self,
        examples: List[AugmentedInputExample],
    ):
        cache_file = self._get_cached_allowed_indices_file()

        if os.path.exists(cache_file):
            return

        hidden_states = self.teacher_saved_output.hidden_states
        logits = self.teacher_saved_output.logits

        num_augmented_examples = 0
        num_retained_augmented_examples = 0
        num_noaug_examples = 0
        allowed_indices = []

        cos = nn.CosineSimilarity()
        softmax = nn.Softmax(dim=-1)

        rank_zero_info("*** Filtering based on distance constraints")

        for i, ex in enumerate(examples):
            num_augmented_examples += len(ex.augmented_examples)
            n_augs = self.hparams.num_aug_candidates if self._is_glitter_on() else self.hparams.num_augments

            augment_indices = np.arange(len(ex.augmented_examples))

            if (
                self.hparams.min_confidence_threshold > 0
                or self.hparams.max_confidence_threshold > 0
                or self.hparams.consistent_augment
            ):
                aug_probs = softmax(logits[i, augment_indices + 1])

                if self.hparams.min_confidence_threshold > 0 or self.hparams.max_confidence_threshold > 0:
                    aug_predict_probs, _ = torch.max(aug_probs, dim=-1)
                    if self.hparams.min_confidence_threshold > 0:
                        augment_indices = augment_indices[
                            np.where(aug_predict_probs.numpy() >= self.hparams.min_confidence_threshold)
                        ]

                    if augment_indices.size > 0 and self.hparams.max_confidence_threshold > 0:
                        augment_indices = augment_indices[
                            np.where(aug_predict_probs.numpy() <= self.hparams.max_confidence_threshold)
                        ]

                if augment_indices.size > 0 and self.hparams.consistent_augment:
                    predicted_label = torch.argmax(softmax(logits[i, 0]))
                    _, aug_predicted_labels = torch.max(aug_probs, dim=-1)
                    if len(augment_indices) == 1:
                        augment_indices = augment_indices[
                            [aug_predicted_labels[augment_indices] == predicted_label]
                        ]
                    else:
                        augment_indices = augment_indices[aug_predicted_labels[augment_indices] == predicted_label]

            if augment_indices.size == 0:
                retained_indices = []
            else:
                distances = (
                    torch.acos(
                        torch.clamp(
                            cos(
                                hidden_states[i, 0].reshape(1, -1),
                                hidden_states[i, augment_indices + 1],
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
            if len(retained_indices) == 0:
                num_noaug_examples += 1

            if len(retained_indices) == 1:
                allowed_indices.append([augment_indices[retained_indices].tolist()])
            else:
                allowed_indices.append(augment_indices[retained_indices].tolist())

        rank_zero_info(f">>> original examples: {len(examples)}")
        rank_zero_info(
            f">>> With augmentation: {num_augmented_examples} ({num_augmented_examples / len(examples):.2f}x)"
        )
        rank_zero_info(
            f">>> Examples without Aug: {num_noaug_examples} ({100. * num_noaug_examples / len(examples):.2f}%)"
        )
        rank_zero_info(
            f">>> After all constraints: {num_retained_augmented_examples} "
            f"({100. * num_retained_augmented_examples / num_augmented_examples:.2f}% of augs or "
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
            "--augment_with_labels",
            action="store_true",
            default=False,
            help="Use labels for augmented samples from the original data samples. Only when `num_augments` > 0",
        )
        parser.add_argument(
            "--vanilla_augment",
            action="store_true",
            default=False,
            help="Ignores Glitter and naively include the whole augmented data in training. Only when `num_augments` > 0",
        )
        parser.add_argument(
            "--min_confidence_threshold",
            default=0.0,
            type=float,
            help="The threshold on min predicted probability of the augmented data based on the teacher, intended for filtering.",
        )
        parser.add_argument(
            "--max_confidence_threshold",
            default=0.0,
            type=float,
            help="The threshold on max predicted probability of the augmented data based on the teacher, intended for filtering.",
        )
        parser.add_argument(
            "--consistent_augment",
            action="store_true",
            default=False,
            help="Only retain augmented data that teacher predicts the same labels as the original data.",
        )
        parser.add_argument(
            "--random_glitter",
            action="store_true",
            default=False,
            help="Select augmented data based on random sampling rather than maximum KL-div between teacher and student",
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

    if args.reinit_pooler:
        arg_summary += "_rep"

    if args.reinit_layers > 0:
        arg_summary += f"_rel{args.reinit_layers}"

    if args.deterministic:
        arg_summary += "_det"

    if args.num_augments > 0:
        arg_summary += f"_naug{args.num_augments}"
        if not args.vanilla_augment:
            arg_summary += f"of{args.num_aug_candidates}"

            if args.random_glitter:
                arg_summary += f"_randglitter"

        if args.min_confidence_threshold > 0:
            arg_summary += f"_minconfd{args.min_confidence_threshold}"

        if args.max_confidence_threshold > 0:
            arg_summary += f"_maxconfd{args.max_confidence_threshold}"

        if args.consistent_augment:
            arg_summary += f"_consist"

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
        logger.warning(f"Output dir: {args.output_dir}")

    model = GlitterGLUETransformer(args)

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

    tb_logger = (
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
        tb_logger,
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
