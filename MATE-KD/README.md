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
* ...

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
@inproceedings{something,
  title = {Fill in the title here},
  author = {Authors here},
  booktitle = {Proceedings here},
  year = {2019},
}
```
