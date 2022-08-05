# Glitter :sparkles:
This repo hosts the implmentation of our paper: [When Chosen Wisely, More Data Is What You Need: A Universal Sample-Efficient Strategy For Data Augmentation](https://aclanthology.org/2022.findings-acl.84/), published at ACL 2022 Findings.
The codebase follows our previous work: [Minimax-kNN](https://github.com/huawei-noah/KD-NLP/tree/main/Minimax-kNN).

## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [Data](#data)
    - [Generate Augmented Data](#generate-augmented-data)
    - [Augmented Data Format](#augmented-data-format)
  - [Training](#training)
    - [Fine-Tuning](#fine-tuning-no-data-augmentation)
    - [Knowledge Distillation](#knowledge-distillation-kd-no-data-augmentation)
    - [Data Augmentation](#data-augmentation)
    - [Consistency Training](#consistency-training)
  - [Evaluation](#evaluation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [License](#license)
  - [Citation](#citation)


## Overview
We present a sample efficient technique for incorporating augmented data into training.
The key ingredient of our approach is to find worst-case examples based on a selection criterion, inspired by adversarial training.

## Getting Started

### Requirements
First, create a conda environment (Python 3.6+), and install [PyTorch](https://pytorch.org/) 1.6+.
Note that the repo is tested with Python 3.8 and PyTorch 1.7 (CUDA 10.1):

```bash
conda create -n glitter python=3.8
conda activate glitter
conda install pytorch==1.7.1 cudatoolkit=10.1 -c pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
```
 
To install the dependencies:

```bash
pip install -e .
```

### Data
Our experiments are conducted on GLUE, SQuAD, and HellaSwag. To prepare the data, simply run the following commands:

```bash
python -m glitter.preprocess.glue --dataset glue/rte --output_dir /path/to/RTE
python -m glitter.preprocess.squad --output_dir /path/to/squad
python -m glitter.preprocess.hellaswag --output_dir /path/to/HellaSwag
```

The data directory should contain three files: `train.jsonl`, `dev.jsonl`, and `test.jsonl`. 
For MNLI, the dev set and the test set should be named: `dev_matched.jsonl`, `dev_mismatched.jsonl`, `test_matched.jsonl`, and `test_mismatched.jsonl`.
Tab-separated `.tsv` files are also supported.

### Generate Augmented Data
We use three Data Augmentation (DA) strategies:

1. __Back-translation:__ [fairseq](https://github.com/facebookresearch/fairseq/) is required to generate augmented data using back-translation. The scripts do not support multiple GPUs.
```bash
CUDA_VISIBLE_DEVICES="0" python aug/backt.py --task rte --data_dir /path/to/RTE --output_dir /path/to/RTE_augmented --do_sample
CUDA_VISIBLE_DEVICES="0" python aug/qa_backt.py --data_dir /path/to/squad --output_dir /path/to/squad_augmented --do_sample
CUDA_VISIBLE_DEVICES="0" python aug/multichoice_backt.py --data_dir /path/to/HellaSwag --output_dir /path/to/HellaSwag_augmented --do_sample
```
2. __Easy Data Augmentation ([Wei & Zou, EMNLP 2019](https://aclanthology.org/D19-1670/)):__ Additional dependencies to generate EDA examples are nltk and [nlpaug](https://github.com/makcedward/nlpaug):
```bash
python aug/eda.py --task rte --data_dir /path/to/RTE --output_dir /path/to/RTE_augmented
```
3. __Mask-and-reconstruct:__
```bash
CUDA_VISIBLE_DEVICES="0,1" python aug/mlm.py --model_name_or_path bert-large-uncased-whole-word-masking --task rte --data_dir /path/to/RTE --output_dir /path/to/RTE_augmented --do_sample --topk 10
```

The size of augmented data is controlled by `--num_augments` (default: 8) in the above commends.

### Augmented Data Format
The data should be stored in a _jsonl_ format where each line corresponds to an example in a json format:

```json
{
   "guid": "8",
   "label": "entailment",
   "text_a": "(Read  for Slate 's take on Jackson's findings.)",
   "text_b": "Slate had an opinion on Jackson's findings.",
   "augmented_samples":[
      {
         "guid":"aug-1",
         "text_a": "( Read for Slate's notes on Jackson's findings. )",
         "text_b": "Slate had an opinion on Jackson's findings."
      },
      ...
   ]
}
```

_Note that the above example is reformatted for better visualization._

## Training

### Fine-Tuning (no data augmentation)

__GLUE__
```bash
CUDA_VISIBLE_DEVICES="0,1" python -m glitter.glue --model_name_or_path distilroberta-base \
                      --gradient_accumulation_steps 4 --learning_rate 3e-5 --warmup_ratio 0.08 \
                      --num_train_epochs 20 --train_batch_size 32 --eval_batch_size 32 \
                      --gpus -1 --fp16 --do_train --max_seq_length 256 --task mnli \
                      --patience 20 --weight_decay 0.0 --eval_interval 0.5 --overwrite_output_dir \ 
                      --data_dir /path/to/MNLI --output_dir /path/to/saved_model
```

The supported tasks are: `cola`, `sst-2`, `mrpc`, `sts-b`, `qqp`, `mnli`, `mnli-mm`, `qnli`, and `rte`.

__SQuAD__
```bash
CUDA_VISIBLE_DEVICES="0,1" python -m glitter.squad --model_name_or_path distilroberta-base \
                      --gradient_accumulation_steps 4 --learning_rate 1.5e-5 --warmup_ratio 0.06 \
                      --num_train_epochs 3 --train_batch_size 8 --eval_batch_size 16 \
                      --gpus -1 --fp16 --do_train --max_seq_length 512 \
                      --weight_decay 0.01 --overwrite_output_dir \ 
                      --data_dir /path/to/squad --output_dir /path/to/saved_model
```

__HellaSwag__
```bash
CUDA_VISIBLE_DEVICES="0,1" python -m glitter.mc --model_name_or_path distilroberta-base \
                      --gradient_accumulation_steps 1 --learning_rate 1.5e-5 --warmup_ratio 0.06 \
                      --adam_epsilon 1e-6 --adam_beta2 0.98 --padding longest --pad_to_multiple_of 8 \
                      --num_train_epochs 20 --train_batch_size 16 --eval_batch_size 16 \
                      --gpus -1 --fp16 --do_train --max_seq_length 512 --task hellaswag \
                      --weight_decay 0.01 --overwrite_output_dir --save_last \ 
                      --data_dir /path/to/HellaSwag --output_dir /path/to/saved_model
```

### Knowledge Distillation (KD; no data augmentation)
Same as fine-tuning, but with an additional argument `--teacher_name_or_path /path/to/teacher_model`.
To reduce the runtime, we build a cache from the outputs of the teacher model before start of the training.

Other options, summarized below, are also available for distillation:

| Argument | Default | Description |
| -------- | ------- | ----------- |
|teacher_name_or_path | None | Path to a directory containing the teacher model (i.e., pytorch_model.bin). Setting this argument triggers the distillation process. | 
|alpha_ce | 0.5 | Linear weight corresponding to cross entropy loss between teacher and student |
|alpha_true | 0.5 | Linear weight corresponding to cross entropy loss between true labels and predictions |
|temperature | 5.0 | Softmax temperature that determines softness of output probabilities |

An example command on a GLUE task looks like:
```bash
CUDA_VISIBLE_DEVICES="0,1" python -m glitter.glue --model_name_or_path distilroberta-base \
                      --gradient_accumulation_steps 4 --learning_rate 3e-5 --warmup_ratio 0.08 \
                      --num_train_epochs 20 --train_batch_size 32 --eval_batch_size 32 \
                      --gpus -1 --fp16 --do_train --max_seq_length 256 --task mnli \
                      --patience 20 --weight_decay 0.0 --eval_interval 0.5 --overwrite_output_dir \ 
                      --data_dir /path/to/MNLI --output_dir /path/to/saved_model
                      --teacher_name_or_path /path/to/teacher/best_tfmr/ --temperature 12.0
```

### Data Augmentation
The command is the same as KD, but with the following options:

| Argument            | Default | Description                                                                                                                                                                                                                |
|---------------------| ------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| num_augments        | 0 | Number of augmentations per train samples used during training                                                                                                                                                             |
| num_aug_candidates  | 0 | Number of candidates per train samples to select augmentation samples from (works only when `--num_augments` > 0 and `--vanilla_augment` is not set)                                                                       |
| alpha_aug           | None | Linear weight corresponding to cross-entropy loss of augmented samples between teacher and student (works only when `--num_augments` > 0 and `--vanilla_augment` is not set). By default, it would be same as `--alpha_ce` |
| vanilla_augment     | False | Ignores Glitter-DA and includes the whole augmented data in training (works only when `--num_augments` > 0)                                                                                                                |

_Note that `--data_dir` should refer to an augmented dataset as described above._

For Glitter-DA :sparkles:, the following command can be run:
```bash
CUDA_VISIBLE_DEVICES="0,1" python -m glitter.glue --model_name_or_path distilroberta-base \
                      --gradient_accumulation_steps 4 --learning_rate 3e-5 --warmup_ratio 0.08 \
                      --num_train_epochs 20 --train_batch_size 32 --eval_batch_size 32 \
                      --gpus -1 --fp16 --do_train --max_seq_length 256 --task mnli \
                      --patience 10 --weight_decay 0.0 --eval_interval 0.5 --overwrite_output_dir \ 
                      --data_dir /path/to/MNLI_augmented --output_dir /path/to/saved_model
                      --teacher_name_or_path /path/to/teacher/best_tfmr/ --temperature 12.0
                      --num_augments 2 --num_aug_candidates 8
```

For vanilla-DA, simply add `--vanilla_augment` to the above command (_Note that in vanilla-DA, `--num_aug_candidates` has no effect and the size of augmented data can be controlled by `--num_augments`_).

### Consistency Training

To enable [Consistency Training](https://proceedings.neurips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html), run the following command (_Works only for sequence classification tasks_): 

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m glitter.semi --model_name_or_path roberta-base \
                      --gradient_accumulation_steps 4 --learning_rate 3e-5 --warmup_ratio 0.08 \
                      --num_train_epochs 20 --train_batch_size 16 --eval_batch_size 32 \
                      --gpus -1 --fp16 --do_train --max_seq_length 256 --task mnli \
                      --patience 10 --weight_decay 0.0 --eval_interval 0.5 --overwrite_output_dir \ 
                      --data_dir /path/to/MNLI_augmented --output_dir /path/to/saved_model 
                      --num_augments 2 --num_aug_candidates 8 --alpha_uda 1.0 --uda_softmax_temp 5.0
```

For vanilla-DA, simply add `--vanilla_augment` to the above command.

## Evaluation

We recommend running evaluation commands on one GPU:

Here is the command for evaluating on the dev set:
```bash
python -m glitter.glue --model_name_or_path /path/to/saved_model/best_tfmr \
                       --do_eval --eval_batch_size 32 \
                       --task mnli --data_dir /path/to/MNLI
```

When labels in the test data are known, replace `--do_eval` with `--do_test`, to run evaluation on the test data. 
When test labels are not given, we can run inference by passing `--do_infer`.

## Bugs or questions?

If you have any questions, or encounter any problems, reach out to `kamalloo (at) ualberta dot ca`.

## License

This project's [license](LICENSE) is under the Apache 2.0 license.

## Citation
If you found our work useful, please cite us as:
```
@inproceedings{kamalloo-etal-2022-chosen,
    title = "When Chosen Wisely, More Data Is What You Need: A Universal Sample-Efficient Strategy For Data Augmentation",
    author = "Kamalloo, Ehsan  and
      Rezagholizadeh, Mehdi  and
      Ghodsi, Ali",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.84",
    doi = "10.18653/v1/2022.findings-acl.84",
    pages = "1048--1062",
}

```