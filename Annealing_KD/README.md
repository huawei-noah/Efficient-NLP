<p align="center">
  <br />
  <img src="https://avatars.githubusercontent.com/u/12619994?s=200&v=4" width="150">
  <br />
  <br />
  <a href="LICENSE"><img alt="Apache License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
</p>

--------------------------------------------------------------------------------

# Annealing Knowledge Distillation
This project is about Annealing-KD long paper: Annealing Knowledge Distillation, which is accepted by EACL 2021.
In this project, we tried to address the capacity gap problem in knowledge distillation and provide an improved knowledge distillation method that is robust against the capacity gap.

The paper can be found at https://arxiv.org/abs/2104.07163 .

## Requirements and Installation

- Install requirements.

- Set the conda environment
```bash
conda env create -f environment.yml
 ```


- Download glue benchmark datasets using the following script: https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e by running 

```bash
python download_glue_data.py --data_dir glue_data --tasks all
```

- For BERT experiments, download the models using the following commands (Bert-base-uncased):

```bash
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
wget -O vocab.txt https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
```

For other experiments' models, look up the name from this link: https://huggingface.co/models and replace bert-base-uncased with the name of the model. 
Different vocab files are required in some models  (e.g RoBERTa has vocab.json and merges.txt instead of vocab.txt), to determine files, go to transformers' github repository: https://github.com/huggingface/tran$

## Running experiments

### Glue tasks:

Befor running the code, you need to train a teacher model. For this purpose use Huggingface run_glue.py script to train teacher model (https://huggingface.co/transformers/v2.3.0/examples.html#glue)
and save the trained teacher checkpoint in a directory.


You can use the following script to run glue experiments:

```bash
export TASK_NAME= {CoLA, RTE, MRPC, STS-B, SST-2, MNLI, QNLI, QQP, WNLI}       # one of these tasks
python ./NLP_glue_tasks/annealing_kd_glue.py 
	--model_type  roberta 
	--teacher_model_name_or_path /path/to/traned/teacher/check-point 
	--student_model_name_or_path /path/to/the/downloaded/pretrained/student/model 
	--task_name $TASK_NAME 
	--data_dir /path/to/glue/task/data 
	--max_seq_length 128 
	--evaluate_during_training 
	--per_gpu_eval_batch_size=32 
	--per_gpu_train_batch_size=32 
	--learning_rate 2e-5 
	--output_dir /path/to/output/directory  
	--num_train_epochs_phase_1 14 
	--num_train_epochs_phase_2 4 
	--max_temperature 7 
	--do_train 
	--do_eval 
	--do_lower_case 
```


### Computer vision tasks:

You can use the following script to run computer vision experiments. (The following script trains both teacher and student)

```bash
python ./computer_vision_tasks/annealing_kd_train.py
    --dataset cifar10 
    --teacher resnet110
    --student resnet8
    --teacher_epochs 200 
    --student_epochs_p1 200 
    --student_epochs_p2 200 
    --max_temperature 10
    --batch-size 128
    --learning-rate 0.1
    --dataset-dir ./path/to/dataset/cache/directory
    --cuda 
```

## Citations


EACL:
```
@inproceedings{jafari2021annealing,
  title={Annealing Knowledge Distillation},
  author={Jafari, Aref and Rezagholizadeh, Mehdi and Sharma, Pranav and Ghodsi, Ali},
  booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
  pages={2493--2504},
  year={2021}
}
```

arXiv:
```
@article{jafari2021annealing,
  title={Annealing Knowledge Distillation},
  author={Jafari, Aref and Rezagholizadeh, Mehdi and Sharma, Pranav and Ghodsi, Ali},
  journal={arXiv preprint arXiv:2104.07163},
  year={2021}
}
```






