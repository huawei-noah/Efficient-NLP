<p align="center">
  <img src="https://avatars.githubusercontent.com/u/12619994?s=200&v=4" width="150">
  <br />
  <br />
  <a href="LICENSE"><img alt="Apache License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
</p>

--------------------------------------------------------------------------------

This repository is the official code for the following paper: *DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low Rank Adaptation*. 

<details><summary>This repository is based on the following repository, we basically integrated DyLoRA on top of LoRA:</summary><p>

* **Low-Rank Adapation of Large Language Models**
  + [LoRA](https://github.com/microsoft/LoRA)

</p></details>

## Repository Overview

There are several directories in this repo:
* [loralib/](loralib) contains the source code for the package 'DyLoRA', which needs to be installed to run the examples we provide;
* [examples/NLU/](examples/NLU) contains an example implementation of DyLoRA in RoBERTa using our package, which produces competitive results on the GLUE benchmark;
* See how we use `loralib` in [RoBERTa](examples/NLU/src/transformers/models/roberta/modeling_roberta.py)

# Requirements and Installation

1. Download LoRA: https://github.com/microsoft/LoRA
2. Download DyLoRA repository (this) and replace the files with the LoRA files
3. cd examples/NLU
4. Setup the conda environment
```bash
conda env create -f environment.yml
```
5. Install the transformers library
```
pip install -e .
```
6. Install DyLoRA integrated loralib
```
cd ../..
pip install -e .
```

# Getting Started

## Model Fine-TUNING
```
cd examples/NLU
bash roberta_base_mnli.sh
```

You can use the following pattern to do other experiments: roberta_base_{task_name}.sh

## Contact
Please contact us or post an issue if you have any questions.

* Mojtaba Valipour (mojtaba.valipour@huawei.com)

# License

This project's [license](LICENSE) is under the Apache 2.0 license.

# Citation

You can cite this project as:

``` bibtex
@misc{va2022DyLoRA,
    title={DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low Rank Adaptation},
    author={Valipour, Mojtaba; Rezagholizadeh, Mehdi; Kobyzev, Ivan and Ghodsi, Ali},
    year={2022},
    eprint={0},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```