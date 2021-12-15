<p align="center">
  <img src="https://avatars.githubusercontent.com/u/12619994?s=200&v=4" width="150">
  <br />
  <br />
  <a href="LICENSE"><img alt="Apache License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
</p>

--------------------------------------------------------------------------------

# MATE-KD 

This project is about MATE-KD long paper: *MATE-KD: Masked Adversarial Text, a Companion to Knowledge Distillation*, which is accepted by ACL 2021. 
In this project, we are interested in KD for model compression and study the use of adversarial training to improve student accuracy using just the logits of the teacher as in standard KD.
The paper can be found at https://arxiv.org/abs/2105.05912v1.

### Features:

* Does MATE-KD

# Requirements and Installation

* Python version >= 3.6
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* [HuggingFace](https://huggingface.co/) version == 3.1.0
* **To set up this code** and develop locally:

1.  Set the conda environment
      ```
      conda env create -f environment.yml
      pip install transformers==3.1.0
      ```

2. Restore the pretrained model in following directories (can be downloaded from Huggingface e.g., https://huggingface.co/distilbert-base-uncased):
    -- ./bert_models/distilbert-base
    -- ./bert_models/distilroberta-base
    -- ./bert_models/uncased_L-6_H-768_A-12
    -- ./bert_models/uncased_L-4_H-256_A-4

3. Restore the finetuned teacher model in following directories:
   -- ./ckpts/teachers_roberta_large
   -- ./ckpts/teachers_bert_base

4. Download the [GLUE benchmark](https://gluebenchmark.com/) in following directory:
 	 -- ./glue_data

# Getting Started

## Model training
```
bash adversarial_train_mnli.sh                 # for BERT-base model
bash run_adversarial_train_mnli_roberta.sh     # for roberta-large model
```

# License

This project's [license](LICENSE) is under the Apache 2.0 license.

# Citation

Please cite as:

``` bibtex
@inproceedings{rashid-etal-2021-mate,
    title = "{MATE}-{KD}: Masked Adversarial {TE}xt, a Companion to Knowledge Distillation",
    author = "Rashid, Ahmad  and
      Lioutas, Vasileios  and
      Rezagholizadeh, Mehdi",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.86",
    doi = "10.18653/v1/2021.acl-long.86",
    pages = "1062--1071",
    abstract = "The advent of large pre-trained language models has given rise to rapid progress in the field of Natural Language Processing (NLP). While the performance of these models on standard benchmarks has scaled with size, compression techniques such as knowledge distillation have been key in making them practical. We present MATE-KD, a novel text-based adversarial training algorithm which improves the performance of knowledge distillation. MATE-KD first trains a masked language model-based generator to perturb text by maximizing the divergence between teacher and student logits. Then using knowledge distillation a student is trained on both the original and the perturbed training samples. We evaluate our algorithm, using BERT-based models, on the GLUE benchmark and demonstrate that MATE-KD outperforms competitive adversarial learning and data augmentation baselines. On the GLUE test set our 6 layer RoBERTa based model outperforms BERT-large.",
}
```
