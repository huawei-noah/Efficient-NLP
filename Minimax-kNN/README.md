# Minimax-kNN
This repo hosts the implmentation of our paper: [Not Far Away, Not So Close: Sample Efficient Nearest Neighbour Data Augmentation via MiniMax](https://aclanthology.org/2021.findings-acl.309/), published at ACL 2021 Findings.

We present a sample efficient semi-supervised data augmentation technique. The key ingredient of our approach is to find the most impactful examples that maximize the KL-divergence between the teacher and the student models. 
The augmentation procedure is framed as finding nearest neighbours from a massive repository of unannotated sentences using [SentAugment](https://github.com/facebookresearch/SentAugment). 
The _k_-NN selection strategy ensures that the augmented text samples fall within the vicinity of the train data points and does not allow the student to deviate too far from the teacher.


## Setup
We recommend creating a conda environment (Python 3.6+), install [PyTorch](https://pytorch.org/) 1.6+ and install [torch-scatter](https://github.com/rusty1s/pytorch_scatter):

```bash
conda create -n knn python=3.8
conda activate knn
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
```
 
Then, install the dependencies:

```bash
pip install -e .
```
Note that the repo is tested with Python 3.8 and PyTorch 1.7 (_mixed precision training_ became native to PyTorch since 1.6, so no need to install Nvidia apex).

#### Prepare the data
IMP and CR can be downloaded from the provided link. Then, the pre-process script should be run to save them in a supported format.
For the other three datasets, the script downloads and pre-processes the data.

| Dataset | Download | Pre-process Script |
| ------- | -------- | ------ |
| IMP | [Here](https://www.kaggle.com/c/detecting-insults-in-social-commentary/) | `python scripts/imp.py /path/to/file --mode train (or dev or test) --output_dir /path/to/IMP` |
| CR | [Here](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) | `python scripts/cr.py --pos_file /path/to/file --neg_file /path/to/file --output_dir /path/to/CR` |
| SST-5 | - | `python scripts/sst5.py --output_dir /path/to/SST-5` |
| SST-2 | - | `python scripts/hf.py glue/sst2 --output_dir /path/to/SST-2` |
| TREC | - | `python scripts/hf.py trec --split_size 500 --output_dir /path/to/TREC` |

#### kNN Data Augmentation
The crucial aspect of kNN augmentation is interpretability as augmented examples are written in natural language.
To augment a dataset using nearest neighbour search, we leverage [SentAugment](https://github.com/facebookresearch/SentAugment) and save the augmented data in a jsonl file, with the following format:

```
{
   "guid":"1",
   "label":0,
   "text_a":"hide new secretions from the parental units",
   "augmented_samples":[
      {
         "guid":"aug-1",
         "text_a":"Introducing the Newest Members of our Family",
      },
      {
         "guid":"aug-2",
         "text_a":"The position of the new sheet in its parent group.",
      },
      {
         "guid":"aug-3",
         "text_a":"You can read this blog post about how we deliver our puppies to their new families.",
      },
   ]
}
```

_Note that the above example is reformatted for better visualization and each sample should actually be stored in a single line._

## Getting Started

### Fine-Tuning
The following command fine-tunes a pre-trained model on a given task:

```bash
python -m knnkd.cls   --model_name_or_path distilroberta-base \
                      --gradient_accumulation_steps 1 --learning_rate 1e-5 --warmup_ratio 0.06 \
                      --num_train_epochs 4 --train_batch_size 32 --eval_batch_size 16 \
                      --gpus -1 --fp16 --do_train --max_seq_length 256 --task rte \
                      --patience 10 --overwrite_output_dir \ 
                      --data_dir /path/to/data --output_dir /path/to/saved_model
```

Supported tasks are _sst-2_, _sst-5_, _trec_, _imp_, and _cr_. For more information, check out [glue.py](src/knnkd/data/glue.py#L263).

### Knowledge Distillation (KD)
The command is the same as fine-tuning, but with an additional argument `--teacher_name_or_path /path/to/teacher_model`.
We build a cache from the outputs of the teacher model before the training starts to reduce the runtime.

Other options, summarized below, are also available for distillation:

| Argument | Default | Description |
| -------- | ------- | ----------- |
|teacher_name_or_path | None | Path to a directory containing the teacher model (i.e., pytorch_model.bin). Setting this argument triggers the distillation process. | 
|alpha_ce | 0.5 | Linear weight corresponding to cross entropy loss between teacher and student |
|alpha_true | 0.5 | Linear weight corresponding to cross entropy loss between true labels and predictions |
|temperature | 5.0 | Softmax temperature that determines softness of output probabilities |

An example command looks like:
```bash
python -m knnkd.cls   --model_name_or_path distilroberta-base \
                      --gradient_accumulation_steps 4 --learning_rate 3e-5 --warmup_steps 800 \
                      --num_train_epochs 6 --train_batch_size 32 --eval_batch_size 32 \
                      --gpus -1 --fp16 --do_train --max_seq_length 256 --task qnli \
                      --patience 10 --overwrite_output_dir \ 
                      --data_dir /path/to/data --output_dir /path/to/saved_model \
                      --teacher_name_or_path /path/to/teacher/best_tfmr/
```
### KD + Data Augmentation
The command is the same as KD, but with the following options:

| Argument | Default | Description |
| -------- | ------- | ----------- |
|num_augments | 0 | Number of augmentations per train samples used during training |
|num_aug_candidates | 0 | Number of candidates per train samples to select augmentation samples from (works only when `--num_augments` > 0 and `--naive_augment` is not set) |
|alpha_aug | None | Linear weight corresponding to cross-entropy loss of augmented samples between teacher and student (works only when `--num_augments` > 0 and `--naive_augment` is not set). By default, it would be same as `--alpha_ce` |
|max_aug_length | 0 | Maximum length of augmented text sequence after tokenization |
|naive_augment | False | Ignores minimax and includes the whole augmented data in training (works only when `--num_augments` > 0) |
|min_distance | 0.0 | Minimum acceptable distance between train example and augmented example within teacher's space |
|max_distance | 0.0 | Maximum acceptable distance between train example and augmented example within teacher's space |
|augment_start_epoch | 0 | First epoch (starting from zero) in which augmented data incorporated into training |
|preserve_order | False | Preserves the order of augmented examples by skipping computing the distance in the teacher's space |

_Note that `--data_dir` should refer to an augmented dataset as described above._

Run the following to run KD with __naive augmentation__:
```bash
python -m knnkd.cls   --model_name_or_path distilroberta-base \
                      --gradient_accumulation_steps 4 --learning_rate 3e-5 --warmup_steps 800 \
                      --num_train_epochs 6 --train_batch_size 32 --eval_batch_size 32 \
                      --gpus -1 --fp16 --do_train --max_seq_length 256 --task qnli \
                      --patience 10 --overwrite_output_dir \ 
                      --data_dir /path/to/augmented_data --output_dir /path/to/saved_model \
                      --teacher_name_or_path /path/to/teacher/best_tfmr/ \
                      --naive_augment --augment_start_epoch 3 --num_augments 8
```
To maintain the original order of NNs, you need to pass `--preserve_order`.
By default, NNs would be reranked within teacher's embedding space.

Here is an example command to run KD with __minimax augmentation__:
```bash
python -m knnkd.cls  --model_name_or_path distilroberta-base \
                     --gradient_accumulation_steps 1 --learning_rate 1e-5 --warmup_ratio 0.06 \
                     --num_train_epochs 6 --train_batch_size 32 --eval_batch_size 32 \
                     --gpus -1 --fp16 --do_train --max_seq_length 256 --task sst-5 \
                     --patience 10 --overwrite_output_dir \ 
                     --data_dir /path/to/augmented_data --output_dir /path/to/saved_model \
                     --teacher_name_or_path /path/to/teacher/best_tfmr/ \
                     --augment_start_epoch 7 --num_augments 1 --num_aug_candidates 8 --max_distance 0.22
```


#### Evaluation

Testing should be run only one GPU otherwise distributed training would duplicate data across GPUs, which in turn, causes misleading test results.

Here is the command for inference:
```bash
python -m knnkd.cls  --model_name_or_path /path/to/saved_model/best_tfmr \
                     --do_infer --eval_batch_size 48 \
                     --task rte --data_dir /path/to/data
```

When labels in test data is known, replace `--do_infer` with `--do_test`:
```bash
python -m knnkd.cls  --model_name_or_path /path/to/saved_model/best_tfmr \
                     --do_test --eval_batch_size 48 \
                     --task trec --data_dir /path/to/data
```

And this is the command for evaluating on the dev data:
```bash
python -m knnkd.cls  --model_name_or_path /path/to/saved_model/best_tfmr \
                     --do_eval --eval_batch_size 48 \
                     --task sst-5 --data_dir /path/to/data
```

## License

This project's [license](LICENSE) is under the Apache 2.0 license.

## Citation
If you found our work useful, please cite us as:
```
@inproceedings{kamalloo-etal-2021-far,
    title = "Not Far Away, Not So Close: Sample Efficient Nearest Neighbour Data Augmentation via {M}ini{M}ax",
    author = "Kamalloo, Ehsan  and
      Rezagholizadeh, Mehdi  and
      Passban, Peyman  and
      Ghodsi, Ali",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.309",
    doi = "10.18653/v1/2021.findings-acl.309",
    pages = "3522--3533",
}

```