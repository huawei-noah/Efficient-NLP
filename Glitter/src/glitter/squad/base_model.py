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

import collections
import json
import logging
import os
from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from typing import Optional, Dict, Tuple, Union

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from .metric import SquadMetric
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
    start_positions: Optional[torch.Tensor] = None
    end_positions: Optional[torch.Tensor] = None


@dataclass
class SQuADBatch:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    token_type_ids: Optional[torch.Tensor] = None
    start_positions: Optional[torch.Tensor] = None
    end_positions: Optional[torch.Tensor] = None
    example_indices: Optional[torch.Tensor] = None
    cache_indices: Optional[torch.Tensor] = None
    augment_mask: Optional[torch.Tensor] = None
    augment_indices: Optional[torch.Tensor] = None
    augment_ranks: Optional[torch.Tensor] = None

    def select_subset(self, attr: Union[str, Optional[torch.Tensor]], selected_indices) -> Optional[torch.Tensor]:
        if attr is not None and isinstance(attr, str):
            val = getattr(self, attr, None)
        else:
            val = attr

        return val[selected_indices] if val is not None else val

    @classmethod
    def sub_batches(cls, batch: Dict[int, Dict[str, Optional[torch.Tensor]]]) -> Dict[int, "SQuADBatch"]:
        return {r: cls(**sub_batch) for r, sub_batch in batch.items() if sub_batch}


def get_model_inputs(
    model_type: str, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None
) -> Dict[str, Optional[torch.Tensor]]:
    inputs = dict(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
    )

    if model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
        del inputs["token_type_ids"]

    return inputs


class SQuADTransformer(BaseTransformer):
    mode = "question-answering"

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
        pass

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_metrics(self, stage: str):
        if stage == "test":
            dataset = self.test_dataset
        else:
            dataset = self.valid_dataset

        postprocess_func = partial(
            postprocess_qa_predictions,
            examples=dataset.examples,
            features=dataset.features,
            output_dir=self.hparams.output_dir,
        )
        self.metric = SquadMetric(postprocess_func, self.hparams.version_2_with_negative)

    def validation_step(self, batch, batch_idx):
        batch = SQuADBatch.sub_batches(batch)[0]

        inputs = get_model_inputs(
            self.config.model_type,
            batch.input_ids,
            batch.token_type_ids,
            batch.attention_mask,
            batch.start_positions,
            batch.end_positions,
        )
        outputs = self(**inputs)

        self.metric.update(batch.example_indices, outputs.start_logits, outputs.end_logits)

        return {
            "val_loss": outputs.loss.cpu().numpy(),
        }

    def _eval_end(self, outputs) -> tuple:
        metric_dict = self.metric.compute()

        val_loss_mean = None
        if all("val_loss" in x for x in outputs):
            val_loss_mean = np.stack([x["val_loss"] for x in outputs]).mean()

        return val_loss_mean, metric_dict

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()

    def on_test_epoch_start(self) -> None:
        self.metric.reset()

    def validation_epoch_end(self, outputs: list):
        val_loss, metrics = self._eval_end(outputs)
        self.log("valid/loss", torch.FloatTensor([val_loss])[0], prog_bar=True, logger=True)

        val_metric = next(iter(metrics.values()))
        self.log("val_metric", torch.FloatTensor([val_metric])[0], logger=True)

        self.log_dict(metrics, prog_bar=False)

    def test_epoch_end(self, outputs):
        test_loss, metrics = self._eval_end(outputs)

        for met_name, met_val in metrics.items():
            self.log(f"valid/{met_name}", met_val, prog_bar=False, logger=True)
            self.log(met_name, torch.FloatTensor([met_val])[0], prog_bar=True, logger=False)

        if test_loss is not None:
            self.log("test/loss", test_loss)

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
            "--doc_stride",
            type=int,
            default=128,
            help="When splitting up a long document into chunks how much stride to take between chunks.",
        )
        parser.add_argument(
            "--n_best_size",
            type=int,
            default=20,
            help="The total number of n-best predictions to generate when looking for an answer.",
        )
        parser.add_argument(
            "--null_score_diff_threshold",
            type=float,
            default=0.0,
            help="The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`.",
        )
        parser.add_argument(
            "--version_2_with_negative",
            action="store_true",
            default=False,
            help="If true, some of the examples do not have an answer.",
        )
        parser.add_argument(
            "--max_answer_length",
            type=int,
            default=30,
            help="The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another.",
        )

        return parser


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    example_indices=None,
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Verbatim from: https://github.com/huggingface/transformers/blob/v4.5.1/examples/question-answering/utils_qa.py#L31
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    # assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    if example_indices is not None:
        examples = [ex for i, ex in enumerate(examples) if i in example_indices]
        features = [
            f for f in features if f["example_index"] in example_indices
        ]

    # example_id_to_index = {ex.id: i for i, ex in enumerate(examples)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_index"]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example.context
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example.id] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example.id] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example.id] = ""
            else:
                all_predictions[example.id] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example.id] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    if version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in all_predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in all_predictions.items()]

    references = [{"id": ex.id, "answers": ex.answers_json} for ex in examples]

    return formatted_predictions, references
