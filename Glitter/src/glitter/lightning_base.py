# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Changes:
# 2021.12.10 - More arguments added (e.g., early stopping, and tensorboard logging)
#               Huawei Technologies Co., Ltd. <foss@huawei.com>
# 2021.12.10 - Mixed precision training updated
#               Huawei Technologies Co., Ltd. <foss@huawei.com>
# Copyright 2018 HuggingFace Inc..
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
"""
Modified from: https://github.com/huggingface/transformers/blob/v4.0.1/examples/lightning_base.py
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForMultipleChoice,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)


logger = logging.getLogger(__name__)


MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "multiple-choice": AutoModelForMultipleChoice,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        mode="base",
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs,
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()

        self.save_hyperparameters(hparams)
        self.train_dataset = None
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir

        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]

        total_steps = self.total_steps()
        if self.hparams.warmup_ratio > 0:
            warmup_steps = self.hparams.warmup_ratio * total_steps
        else:
            warmup_steps = self.hparams.warmup_steps

        sch_kwargs = dict(
            num_warmup_steps=warmup_steps,
        )

        if self.hparams.lr_scheduler != "constant":
            sch_kwargs["num_training_steps"] = total_steps

        scheduler = get_schedule_func(self.opt, **sch_kwargs)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer_grouped_parameters = self.get_trainable_parameters()
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def get_trainable_parameters(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def get_number_of_gpus(self):
        if self.hparams.gpus == -1:
            if self.hparams.available_gpus:
                return len(self.hparams.available_gpus)
            else:
                return torch.cuda.device_count()
        else:
            if self.hparams.available_gpus:
                return min(len(self.hparams.available_gpus), self.hparams.gpus)
            else:
                return self.hparams.gpus

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        if self.train_dataset is None:
            return 0
        else:
            num_devices = max(1, self.get_number_of_gpus())
            effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
            dataset_size = len(self.train_dataset)
            return (dataset_size / effective_batch_size) * self.hparams.max_epochs

    def _prepare_data(self, stage: str) -> None:
        if self.hparams.do_train:
            self.train_dataset = self.get_dataset("train")
            self.valid_dataset = self.get_dataset("dev")

        if self.hparams.do_eval:
            self.test_dataset = self.get_dataset("dev")

        if self.hparams.do_test:
            self.test_dataset = self.get_dataset("test")

    def setup(self, stage: str):
        self._prepare_data(stage)
        self.configure_metrics(stage)

    def configure_metrics(self, stage: str) -> Optional[Any]:
        """
        Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module,
        and initialize any data specific metrics.
        """
        pass

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def get_dataloader(self, mode: str, batch_size: int, shuffle: bool = False, dataset=None):
        if dataset is None:
            dataset = self.get_dataset(mode)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.get_collator(mode),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def get_collator(self, mode: str):
        return None

    def get_dataset(self, mode: str):
        raise NotImplementedError("This method must be implemented for each task")

    def train_dataloader(self):
        return self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True, dataset=self.train_dataset)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.global_step
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default=None,
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--adam_beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
        parser.add_argument("--adam_beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument(
            "--warmup_ratio",
            default=0.0,
            type=float,
            help="Linear warmup over a ratio of train steps. Overrides `--warmup_steps`.",
        )
        parser.add_argument("--num_workers", default=6, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument("--padding", default=None, type=str, choices=("longest", "max_length"))
        parser.add_argument(
            "--pad_to_multiple_of",
            default=None,
            type=int,
            help="See https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/",
        )


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr/group_{i}": lr for i, lr in enumerate(lr_scheduler.get_last_lr())}
        pl_module.logger.log_metrics(lrs, trainer.global_step)

    @pl.utilities.rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        for cl in trainer.callbacks:
            if isinstance(cl, pl.callbacks.EarlyStopping):
                rank_zero_info(f"early_stop {cl.wait_count}/{cl.patience} best = {cl.best_score.item()}")
                break

        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}".format(key, str(metrics[key])))

    @pl.utilities.rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics

        if getattr(pl_module.hparams, "test_mode", False):
            results_file_prefix = pl_module.hparams.test_mode
        else:
            results_file_prefix = "dev"

        if any(key not in ("log", "progress_bar") for key in metrics):
            rank_zero_info(f"\n***** {results_file_prefix} results *****")

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, f"{results_file_prefix}_results.txt")
        try:
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))
        except PermissionError:
            rank_zero_warn(f"Cannot save results due to premission error at `{output_test_results_file}`")
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser):
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        default=False,
        help="Overwrite the content of the output directory.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument(
        "--gpus",
        default=-1,
        type=int,
        help="The number of GPUs allocated for this, it is by default -1 meaning all GPUs",
    )
    parser.add_argument(
        "--available_gpus",
        default=None,
        nargs="*",
        type=int,
        help="The indices of available GPUs to avoid modifying CUDA_VISIBLE_DEVICES (by default all GPUs will be used)",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        choices=("dp", "ddp", "ddp2", "ddp_cpu", "ddp_spawn"),
        type=str,
        help="The accelerator backend for multi-GPU/CPU environment",
    )
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run predictions on the dev set.")
    parser.add_argument("--do_infer", action="store_true", help="Whether to run inference on the test set.")
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run predictions on the test set if it includes labels."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--eval_interval",
        dest="val_check_interval",
        type=float,
        default=1.0,
        help="Run an evaluation every X steps (int) or X times (float) within an epoch.",
    )
    parser.add_argument(
        "--log_interval",
        dest="log_every_n_steps",
        type=int,
        default=10,
        help="Controls logging frequency during training",
    )
    parser.add_argument(
        "--flush_log_interval",
        dest="flush_logs_every_n_steps",
        type=int,
        default=100,
        help="Controls log writing frequency during training",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Number of evaluation runs with no improvement after which training will be stopped (For early stopping).",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="An absolute change of less than `min_delta`, will count as no improvement (For early stopping).",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="The input test file. Overrides test file in `--data_dir`",
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=False, help="Enables cudnn.deterministic for reproducibility."
    )
    parser.add_argument(
        "--save_last",
        action="store_true",
        default=False,
        help="Saves only the last model and skips periodical saves during training",
    )


def generic_train(
    model: BaseTransformer,
    args: argparse.Namespace,
    pl_logger=True,  # can pass WandbLogger() here
    extra_callbacks=None,
    checkpoint_callback=None,
    logging_callback=None,
    weights_summary=None,
    **extra_train_kwargs,
):
    pl.seed_everything(args.seed)

    extra_callbacks = extra_callbacks or []

    # init model
    odir = Path(args.output_dir)
    if args.do_train and args.overwrite_output_dir and odir.exists():
        shutil.rmtree(odir)
    odir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output saved into `{odir}`")

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = extra_train_kwargs or {}

    if args.fp16:
        train_params["precision"] = 16

    if args.available_gpus:
        if args.gpus > 0:
            args.gpus = args.available_gpus[: args.gpus]
        elif args.gpus < 0:
            args.gpus = args.available_gpus

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=weights_summary,
        callbacks=[logging_callback] + extra_callbacks,
        logger=pl_logger,
        checkpoint_callback=checkpoint_callback,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)

    return trainer
