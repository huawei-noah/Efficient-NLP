<p align="center">
  <img src="logo.png" width="150">
  <br />
  <img src="https://avatars.githubusercontent.com/u/12619994?s=200&v=4" width="150">
  <br />
  <br />
  <a href="LICENSE"><img alt="Apache License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
</p>

--------------------------------------------------------------------------------

# Combined-KD

This project is about Combined-KD long paper: *How to Select One Among All ? An Extensive Empirical Study Towards the Robustness of Knowledge Distillation* in Natural Language Understanding which is accepted by EMNLP findings 2021.
In this project, we proposed a Combined-KD (ComKD) by taking advantage of data-augmentation and progressive training. Results show that our proposed ComKD not only achieves a new state-of-the-art (SOTA) on the GLUE benchmark, but also more robust than other KD methods under OOD evaluation and adversarial attacks. Paper link:https://arxiv.org/abs/2109.05696v1


# Requirements and Installation

* Python version >= 3.6
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* [HuggingFace](https://huggingface.co/) version == 3.1.0


1.  Set the conda environment
      ```
      conda env create -f environment.yml
      ```
      
2. Restore the pretrained model from huggingface in following dirs (Can be downloaded from huggingface e.g., https://huggingface.co/distilbert-base-uncased):
    -- ./bert_models/distilbert-base
    -- ./bert_models/distilroberta-base
    -- ./bert_models/uncased_L-6_H-768_A-12
    -- ./bert_models/uncased_L-4_H-256_A-4
    
3. Restore the finetuned teacher model in following dirs:
   -- ./ckpts/teachers_roberta_large
   -- ./ckpts/teachers_bert_base

4. Download glue benchmark in following dir (https://gluebenchmark.com/):
 	 -- ./glue_data

# Getting Started

The [full documentation](https://www.huawei.com/) contains cool things.


## Train the models:
```
bash  run_combinekd_mnli_bert.sh           for BERT model
bash  run_combinekd_mnli_roberta.sh        for RoBERTa model
```

## Something else doing

# Join the Huawei Noah's Ark community

* You can omit this section if you want, it's just if needed in the future.
* Main page: https://www.noahlab.com.hk/
* Github: https://github.com/huawei-noah

# License

This project's [license](LICENSE) is under the Apache 2.0 license.

``` bibtex
@misc{li2021select,
      title={How to Select One Among All? An Extensive Empirical Study Towards the Robustness of Knowledge Distillation in Natural Language Understanding}, 
      author={Tianda Li and Ahmad Rashid and Aref Jafari and Pranav Sharma and Ali Ghodsi and Mehdi Rezagholizadeh},
      year={2021},
      eprint={2109.05696},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
