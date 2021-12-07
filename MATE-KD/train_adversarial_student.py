# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Changes made to the HugginFace script: 
# - We keep the dataloader and argparse from Huggingface and implement our MATE-KD

# Copyright 2020 The HuggingFace Team. All rights reserved.
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


import argparse
import json
import os
import pickle
import shutil
import csv
import logging
import random
import glob
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from dataclasses import dataclass, field
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
    DataCollatorForLanguageModeling
)

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM
from transformers.modeling_roberta import RobertaClassificationHead

import tensorflow.compat.v1 as tf


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataProcessingArguments:
    task_name: str = field(
        metadata={"help": "The name of the task to train selected in the list: " + ", ".join(processors.keys())}
    )
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, rand_vec=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.rand_vec = rand_vec


class MNLIDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, set_type):
        'Initialization'
        self.examples = self._create_examples(self._read_tsv(data_path), set_type)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence1_index = 0
        sentence2_index = 0
        for (i, line) in enumerate(lines):

          if i == 0:
            # Identify the sentence index
            for j, token in enumerate(line):
              if token.strip() == "sentence1":
                sentence1_index = j
              elif token.strip() == "sentence2":
                sentence2_index = j
            continue

          guid = int(line[0]) #"%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
          text_a = line[sentence1_index] #tokenization.convert_to_unicode(line[sentence1_index])
          text_b = line[sentence2_index] #tokenization.convert_to_unicode(line[sentence2_index])
          if set_type == "train":
            label = [float(line[-1]==l) for l in self._get_label()]
          else:
            label = [float(line[-1]==l) for l in self._get_label()]

          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, rand_vec=None))
        return examples

    def _get_label(self):
        return ['contradiction', 'neutral', 'entailment']

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        example = self.examples[index]

        return example


class RTEDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, set_type):
        'Initialization'
        self.examples = self._create_examples(self._read_tsv(data_path), set_type)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence1_index = 0
        sentence2_index = 0
        for (i, line) in enumerate(lines):

          if i == 0:
            # Identify the sentence index
            for j, token in enumerate(line):
              if token.strip() == "sentence1":
                sentence1_index = j
              elif token.strip() == "sentence2":
                sentence2_index = j
            continue

          guid = int(line[0]) #"%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
          text_a = line[sentence1_index] #tokenization.convert_to_unicode(line[sentence1_index])
          text_b = line[sentence2_index] #tokenization.convert_to_unicode(line[sentence2_index])
          if set_type == "train":
            label = [float(line[-1]==l) for l in self._get_label()]
          else:
            label = [float(line[-1]==l) for l in self._get_label()]

          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _get_label(self):
        return ['entailment', 'not_entailment']

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        example = self.examples[index]

        return example


class QQPDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, set_type):
        'Initialization'
        self.examples = self._create_examples(self._read_tsv(data_path), set_type)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence1_index = 0
        sentence2_index = 0
        for (i, line) in enumerate(lines):

          if i == 0:
            # Identify the sentence index
            for j, token in enumerate(line):
              if token.strip() == "question1":
                sentence1_index = j
              elif token.strip() == "question2":
                sentence2_index = j
            continue


          try:

             guid = i #int(line[0]) #"%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
             text_a = line[sentence1_index].strip() #tokenization.convert_to_unicode(line[sentence1_index])
             text_b = line[sentence2_index].strip() #tokenization.convert_to_unicode(line[sentence2_index])
             if not text_a or not text_b:
               continue

             if set_type == "train":
               label = [float(line[-1]==l) for l in self._get_label()]
             else:
               label = [float(line[-1]==l) for l in self._get_label()]

             examples.append(
                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
          except:
             continue


        return examples

    def _get_label(self):
        return ['0', '1']

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        example = self.examples[index]

        return example


class WNLIDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, set_type):
        'Initialization'
        self.examples = self._create_examples(self._read_tsv(data_path), set_type)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence1_index = 0
        sentence2_index = 0
        for (i, line) in enumerate(lines):

          if i == 0:
            # Identify the sentence index
            for j, token in enumerate(line):
              if token.strip() == "sentence1":
                sentence1_index = j
              elif token.strip() == "sentence2":
                sentence2_index = j
            continue


          try:

             guid = i #int(line[0]) #"%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
             text_a = line[sentence1_index] #tokenization.convert_to_unicode(line[sentence1_index])
             text_b = line[sentence2_index] #tokenization.convert_to_unicode(line[sentence2_index])
             if set_type == "train":
               label = [float(line[-1]==l) for l in self._get_label()]
             else:
               label = [float(line[-1]==l) for l in self._get_label()]

             examples.append(
                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
          except:
             continue


        return examples

    def _get_label(self):
        return ['0', '1']

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        example = self.examples[index]

        return example

class STSBDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, set_type):
        'Initialization'
        self.examples = self._create_examples(self._read_tsv(data_path), set_type)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence1_index = 0
        sentence2_index = 0
        for (i, line) in enumerate(lines):

          if i == 0:
            # Identify the sentence index
            for j, token in enumerate(line):
              if token.strip() == "sentence1":
                sentence1_index = j
              elif token.strip() == "sentence2":
                sentence2_index = j
            continue


          try:

             guid = i #int(line[0]) #"%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
             text_a = line[sentence1_index] #tokenization.convert_to_unicode(line[sentence1_index])
             text_b = line[sentence2_index] #tokenization.convert_to_unicode(line[sentence2_index])
             if set_type == "train":
               label = float(line[-1])
             else:
               label = float(line[-1])

             examples.append(
                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
          except:
             continue


        return examples

    def _get_label(self):
        return ['0', '1']

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        example = self.examples[index]

        return example

class QNLIDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, set_type):
        'Initialization'
        self.examples = self._create_examples(self._read_tsv(data_path), set_type)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence1_index = 0
        sentence2_index = 0
        for (i, line) in enumerate(lines):

          if i == 0:
            # Identify the sentence index
            for j, token in enumerate(line):
              if token.strip() == "question":
                sentence1_index = j
              elif token.strip() == "sentence":
                sentence2_index = j
            continue

          guid = int(line[0]) #"%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
          text_a = line[sentence1_index] #tokenization.convert_to_unicode(line[sentence1_index])
          text_b = line[sentence2_index] #tokenization.convert_to_unicode(line[sentence2_index])
          if set_type == "train":
            label = [float(line[-1]==l) for l in self._get_label()]
          else:
            label = [float(line[-1]==l) for l in self._get_label()]

          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _get_label(self):
        return ['entailment', 'not_entailment']

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        example = self.examples[index]

        return example

class MRPCDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, set_type):
        'Initialization'
        self.examples = self._create_examples(self._read_tsv(data_path), set_type)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence1_index = 3
        sentence2_index = 4
        for (i, line) in enumerate(lines):

          if i == 0:
            # Identify the sentence index
            #for j, token in enumerate(line):
            #  if token.strip() == "question":
            #    sentence1_index = j
            #  elif token.strip() == "sentence":
            #    sentence2_index = j
            continue

          guid = int(line[0]) #"%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
          text_a = line[sentence1_index] #tokenization.convert_to_unicode(line[sentence1_index])
          text_b = line[sentence2_index] #tokenization.convert_to_unicode(line[sentence2_index])
          if set_type == "train":
            label = [float(line[0]==l) for l in self._get_label()]
          else:
            label = [float(line[0]==l) for l in self._get_label()]

          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _get_label(self):
        return ['0', '1']

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        example = self.examples[index]

        return example

class SST2Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, set_type):
        'Initialization'
        self.examples = self._create_examples(self._read_tsv(data_path), set_type)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence_index = 0
        for (i, line) in enumerate(lines):

          if i == 0:
            # Identify the sentence index
            for j, token in enumerate(line):
              if token.strip() == "sentence":
                sentence_index = j
            continue

          guid = i #"%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
          if set_type == "train":
            text_a = line[sentence_index]
            label = [float(line[-1]==l) for l in self._get_label()]
          else:
            text_a = line[sentence_index]
            label = [float(line[-1]==l) for l in self._get_label()]

          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label,
              rand_vec=torch.normal(0, std_z, (1, 128, 256))))
        return examples

    def _get_label(self):
        return ["0","1"]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        example = self.examples[index]

        return example


class ColaDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_path, set_type):
        'Initialization'
        self.examples = self._create_examples(self._read_tsv(data_path), set_type)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence_index = 3
        for (i, line) in enumerate(lines):

          guid = i #"%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
          if set_type == "train":
            text_a = line[sentence_index]
            label = [float(line[1]==l) for l in self._get_label()]
          else:
            text_a = line[sentence_index]
            label = [float(line[1]==l) for l in self._get_label()]

          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _get_label(self):
        return ["0","1"]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        example = self.examples[index]

        return example


def mask_tokens(inputs, tokenizer, mlm_probability):
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels, masked_indices.float()


def create_collate_fn(args, tokenizer, max_length):
    def collate_fn(data):
        text = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) if example.text_b is not None else example.text_a for example in data], max_length=max_length, truncation=True,
            pad_to_max_length=True,
        )

        input_ids = text['input_ids']

        #token_type_ids = text['token_type_ids']
        attention_mask = text['attention_mask']

        guids = [example.guid for example in data]

        labels = [example.label for example in data]

        input_ids = torch.tensor(input_ids)
        #token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)

        input_ids_permuted, labels_permuted, mask_permuted = mask_tokens(input_ids.clone(), tokenizer, args.p_value)

        labels = torch.tensor(labels)

        guids = torch.tensor(guids)

        return guids, input_ids, attention_mask, labels, input_ids_permuted, mask_permuted, labels_permuted #, token_type_ids

    return collate_fn

def freeze_model(model, freeze):
    for param in model.parameters():
        param.requires_grad = not freeze

from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape, device="cuda")
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = F.log_softmax(logits, dim=-1) + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

KL_temperature = 1.0
std_z = 0.01

def divergence(student_logits, teacher_logits):
    divergence = -torch.sum(F.log_softmax(student_logits / KL_temperature, dim=-1) * F.softmax(teacher_logits / KL_temperature, dim=-1), dim=-1)  # forward KL
    return torch.mean(divergence)


def train(args, train_dataset, eval_dataset, model, teacher_model, generator, tokenizer):

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)

    train_sampler = data.RandomSampler(train_dataset) if args.local_rank == -1 else data.distributed.DistributedSampler(train_dataset)

    train_dataloader = data.DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=create_collate_fn(args, tokenizer, 128))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=2138, #args.warmup_steps,
        num_training_steps=t_total
    )


    gen_optimizer = AdamW(generator.parameters(), lr=5e-7, eps=args.adam_epsilon)
    gen_scheduler = get_linear_schedule_with_warmup(
        gen_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    gen_pre_optimizer = AdamW(generator.parameters(), lr=5e-6, eps=args.adam_epsilon)
    gen_pre_scheduler = get_linear_schedule_with_warmup(
        gen_pre_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        generator, [gen_optimizer, gen_pre_optimizer] = amp.initialize(generator, [gen_optimizer, gen_pre_optimizer], opt_level=args.fp16_opt_level)
        teacher_model = amp.initialize(teacher_model, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_train = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    max_met = -1
    max_met_step = -1
    model.zero_grad()
    generator.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproducibility

    n_generator_iter = 10
    n_student_iter = 100

    idx_pseudo = 0
    n_repeat_batch = n_generator_iter + n_student_iter

    logger.info("  Pre-train Generator")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            generator.train()
            batch = tuple(t.to(args.device) for t in batch)

            input_ids = batch[1]
            attention_mask = batch[2]
            labels = batch[3]
            token_type_ids = None #batch[5]
            input_ids_permuted = batch[4]
            masked_permuted = batch[5]
            # data augmentation
            outputs = generator(input_ids=input_ids_permuted, attention_mask=attention_mask, token_type_ids=token_type_ids)
               
            prediction_scores = outputs[0]

            prediction_scores = gumbel_softmax(prediction_scores, 1.0)

            teacher_inp = torch.matmul(prediction_scores, teacher_model.roberta.embeddings.word_embeddings.weight) * masked_permuted.unsqueeze(-1)
            student_inp = torch.matmul(prediction_scores, model.roberta.embeddings.word_embeddings.weight) * masked_permuted.unsqueeze(-1)

            teacher_inp = teacher_inp + (teacher_model.roberta.embeddings.word_embeddings(input_ids) * (1 - masked_permuted.unsqueeze(-1)))
            student_inp = student_inp + (model.roberta.embeddings.word_embeddings(input_ids) * (1 - masked_permuted.unsqueeze(-1)))

            teacher_logits = teacher_model(attention_mask=attention_mask, inputs_embeds=teacher_inp, token_type_ids=token_type_ids)[0]
            student_logits = model(attention_mask=attention_mask, inputs_embeds=student_inp, token_type_ids=token_type_ids)[0]

            # generator training loss
            if args.task_name == "sts-b":
                loss = F.mse_loss(student_logits, teacher_logits)
            else:
                loss = divergence(student_logits, teacher_logits)


            if idx_pseudo % n_repeat_batch < n_generator_iter:
                loss = -loss

                gen_optimizer.zero_grad()

                if args.fp16:
                    with amp.scale_loss(loss, gen_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(gen_optimizer), 5)
                else:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(),5)

                gen_optimizer.step()
                gen_scheduler.step()

            elif idx_pseudo % n_repeat_batch < (n_generator_iter + n_student_iter):

                if args.task_name == "sts-b":
                    teacher_logits = teacher_model(attention_mask=attention_mask, input_ids=input_ids, token_type_ids=token_type_ids)[0]
                    student_logits = model(attention_mask=attention_mask, input_ids=input_ids, token_type_ids=token_type_ids)[0]
                    loss_teach = F.mse_loss(student_logits, teacher_logits)
                    loss_good = F.mse_loss(student_logits.squeeze(-1), labels)
                else:
                    teacher_logits = teacher_model(attention_mask=attention_mask, input_ids=input_ids, token_type_ids=token_type_ids)[0]
                    student_logits = model(attention_mask=attention_mask, input_ids=input_ids, token_type_ids=token_type_ids)[0]
                    loss_teach = divergence(student_logits, teacher_logits)
                    loss_good = torch.mean(-torch.sum(F.log_softmax(student_logits / KL_temperature, dim=-1) * labels, dim=-1))

                # loss in the paper
                loss = loss_good * (1/3) + (1/3) * loss + (1/3) * loss_teach
                # new loss
                #loss = loss_good * 0.5 + 0.5 * loss_teach

                optimizer.zero_grad()

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5)

                optimizer.step()
                scheduler.step()

                tr_loss += loss.item()
                steps_train += 1


            if global_step % args.logging_steps == 0:
                logs = {}
                if (
                    args.local_rank == -1 and args.evaluate_during_training
                ):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, eval_dataset, model, tokenizer)
                    for key, value in results.items():
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value

                loss_scalar = tr_loss / steps_train if steps_train > 0 else 0 #(tr_loss - logging_loss) / args.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                logging_loss = tr_loss

                acc_tasks = ['sst-2', 'rte', 'qnli', 'qqp', 'mrpc', 'wnli']
                mnliacc_tasks = ['mnli']
                mnlimmacc_tasks = ['mnli-mm']
                mcc_tasks = ['cola']
                accf1_tasks = [] #['qqp', 'mrpc']
                corr_tasks = ['sts-b']
                current_met = 0

                if args.task_name in acc_tasks:
                    current_met = logs["eval_acc"]
                    if max_met <= logs["eval_acc"]:
                        max_met = logs["eval_acc"]
                        max_met_step = global_step
                elif args.task_name in mnliacc_tasks:
                    current_met = logs["eval_mnli/acc"]
                    if max_met <= logs["eval_mnli/acc"]:
                        max_met = logs["eval_mnli/acc"]
                        max_met_step = global_step
                elif args.task_name in mnlimmacc_tasks:
                    current_met = logs["eval_mnli-mm/acc"]
                    if max_met <= logs["eval_mnli-mm/acc"]:
                        max_met = logs["eval_mnli-mm/acc"]
                        max_met_step = global_step
                elif args.task_name in mcc_tasks:
                    current_met = logs["eval_mcc"]
                    if max_met <= logs["eval_mcc"]:
                        max_met = logs["eval_mcc"]
                        max_met_step = global_step
                elif args.task_name in accf1_tasks:
                    current_met = logs["eval_acc_and_f1"]
                    if max_met <= logs["eval_acc_and_f1"]:
                        max_met = logs["eval_acc_and_f1"]
                        max_met_step = global_step
                elif args.task_name in corr_tasks:
                    current_met = logs["eval_corr"]
                    if max_met <= logs["eval_corr"]:
                        max_met = logs["eval_corr"]
                        max_met_step = global_step

                logs["eval_met_max"] = max_met
                logs["eval_met_max_step"] = max_met_step

                print(json.dumps({**logs, **{"step": global_step}}))
                # restore best model
                if current_met == max_met:
                    output_dir = os.path.join(args.output_dir, "best")
                    shutil.rmtree(output_dir, ignore_errors=True)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            idx_pseudo += 1
            global_step += 1


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break


    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,) #("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir,) #(args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        #eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = data.SequentialSampler(eval_dataset)
        eval_dataloader = data.DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=create_collate_fn(args, tokenizer, 128))

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
                labels = batch[3]
                inputs["token_type_ids"] = None

                #if args.model_type != "distilbert":
                #    inputs["token_type_ids"] = (
                #        batch[5] if args.model_type in ["bert", "xlnet", "albert"] else None
                #    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                logits = model(**inputs)[0]
                if args.task_name == "sts-b":
                    tmp_eval_loss = F.mse_loss(logits.squeeze(-1), labels)
                else:
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    per_example_loss = -torch.sum(labels * log_probs, dim=-1)
                    tmp_eval_loss = torch.mean(per_example_loss)

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
            out_label_ids = np.argmax(out_label_ids, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def main():
    parser = HfArgumentParser((ModelArguments, DataProcessingArguments, TrainingArguments))

    #parser.add_argument(
    #	"--bert_pretrained_path",
    #	type=str,
    #	required=True,
    #	help="The path corresponding to the pre-trained BERT model."
    #)

    parser.add_argument(
    	"--p_value",
    	type=float,
        default=0.3,
    	required=False,
    	help="Enable if you initialize student with an MNLI checkpoint for tasks such as RTE and STS-B."
    )

    parser.add_argument(
    	"--use_mnli_ckpt",
    	type=bool,
    	required=False,
    	help="Enable if you initialize student with an MNLI checkpoint for tasks such as RTE and STS-B."
    )

    parser.add_argument(
    	"--teacher_path",
    	type=str,
    	required=True,
    	help="The path corresponding to the teacher BERT model."
    )

    model_args, dataprocessing_args, training_args, rest_args = parser.parse_args_into_dataclasses()

    args = argparse.Namespace(**vars(model_args), **vars(dataprocessing_args), **vars(training_args), **vars(rest_args))

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    #args.n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    print(num_labels)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = AutoTokenizer.from_pretrained(
       	args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    #args.model_type = args.model_type.lower()
    if not args.use_mnli_ckpt:
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
        )
    else:
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=3,
            finetuning_task=args.task_name,
        )

        config.num_labels = 3

    model = AutoModelForSequenceClassification.from_pretrained( #AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config
    )

    if args.use_mnli_ckpt:
        config.num_labels = num_labels
        logger.info('Reintializing model classifier layer...')
        model.num_labels = num_labels
        model.classifier = RobertaClassificationHead(config)


    teacher_config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.teacher_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
       	args.tokenizer_name if args.tokenizer_name else args.teacher_path,
    )
    teacher_model = AutoModelForSequenceClassification.from_pretrained( #AutoModelForSequenceClassification.from_pretrained(
        args.teacher_path,
        from_tf=False,
        config=teacher_config
    )

    gen_bert_tokenizer = RobertaTokenizer.from_pretrained('bert_models/distilroberta-base')#, cache_dir="downloaded")
    generator = RobertaForMaskedLM.from_pretrained('bert_models/distilroberta-base')#, cache_dir="downloaded")

    #model = BertForSequenceClassification(config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    teacher_model.to(args.device)
    generator.to(args.device)


    logger.info("Training/evaluation parameters %s", args)

    freeze_model(teacher_model, True)
    teacher_model.eval()

    print(model)
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", model_total_params)

    # Training
    if args.do_train:
        if args.task_name == "mnli":
            training_set = MNLIDataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = MNLIDataset(os.path.join(args.data_dir, "dev_matched.tsv"), "test")
        elif args.task_name == "mnli-mm":
            training_set = MNLIDataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = MNLIDataset(os.path.join(args.data_dir, "dev_mismatched.tsv"), "test")
        elif args.task_name == "sst-2":
            training_set = SST2Dataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = SST2Dataset(os.path.join(args.data_dir, "dev.tsv"), "test")
        elif args.task_name == "rte":
            training_set = RTEDataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = RTEDataset(os.path.join(args.data_dir, "dev.tsv"), "test")
        elif args.task_name == "qnli":
            training_set = QNLIDataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = QNLIDataset(os.path.join(args.data_dir, "dev.tsv"), "test")
        elif args.task_name == "qqp":
            training_set = QQPDataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = QQPDataset(os.path.join(args.data_dir, "dev.tsv"), "test")
        elif args.task_name == "mrpc":
            training_set = MRPCDataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = MRPCDataset(os.path.join(args.data_dir, "dev.tsv"), "test")
        elif args.task_name == "cola":
            training_set = ColaDataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = ColaDataset(os.path.join(args.data_dir, "dev.tsv"), "test")
        elif args.task_name == "wnli":
            training_set = WNLIDataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = WNLIDataset(os.path.join(args.data_dir, "dev.tsv"), "test")
        elif args.task_name == "sts-b":
            training_set = STSBDataset(os.path.join(args.data_dir, "train.tsv"), "train")
            eval_set = STSBDataset(os.path.join(args.data_dir, "dev.tsv"), "test")

        global_step, tr_loss = train(args, training_set, eval_set, model, teacher_model, generator, tokenizer)
        logger.info("  global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
