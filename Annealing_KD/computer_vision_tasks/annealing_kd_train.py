# coding=utf-8
# 2022.6.8-Changed for Annealing knowledage distillation
#      Huawei Technologies Co., Ltd. <aref.jafari@huawei.com>
# Copyright 2022 Huawei Technologies Co., Ltd.
# Copyright 2019 Iman Mirzadeh (https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation).
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

import os
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_cifar
from model_factory import create_cnn_model, is_resnet


# Use this code to specify the gpu device.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False
	
	
def parse_arguments():
	parser = argparse.ArgumentParser(description='Annealing Knowledge Distillation Code')
	parser.add_argument('--teacher_epochs', default=200, type=int,  help='number of total epochs to run')
	parser.add_argument('--student_epochs_p1', default=200, type=int, help='number of total epochs to run')
	parser.add_argument('--student_epochs_p2', default=100, type=int, help='number of total epochs to run')
	parser.add_argument('--dataset', default='cifar100', type=str, help='dataset. can be either cifar10 or cifar100')
	parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
	parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
	parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
	parser.add_argument('--teacher', default='resnet110', type=str, help='teacher student name')
	parser.add_argument('--student', '--model', default='resnet8', type=str, help='teacher student name')
	parser.add_argument('--teacher-checkpoint', default='', type=str, help='optinal pretrained checkpoint for teacher')
	parser.add_argument(
		"--cuda", action="store_true", help="whether or not use cuda(train on GPU)",
	)
	parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
	parser.add_argument('--max_temperature', default=10, type=int, help='KD temperature')
	args = parser.parse_args()
	return args


def load_checkpoint(model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn student
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn student with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	model.load_state_dict(model_ckp['model_state_dict'])
	return model


class TrainManager(object):
	def __init__(self, max_temprature, phase, student, teacher=None, train_loader=None, test_loader=None, train_config={}):
		self.max_temperature = max_temprature
		self.phase = phase
		self.student = student
		self.teacher = teacher
		self.have_teacher = bool(self.teacher)
		self.device = train_config['device']
		self.name = train_config['name']
		self.optimizer = optim.SGD(self.student.parameters(),
								   lr=train_config['learning_rate'],
								   momentum=train_config['momentum'],
								   weight_decay=train_config['weight_decay'])

		self.student.to(self.device)
		if self.have_teacher:
			self.teacher.eval()
			self.teacher.train(mode=False)
			self.teacher.to(self.device)

		self.train_loader = train_loader
		self.test_loader = test_loader
		self.config = train_config
	
	def train(self):
		lambda_ = self.config['lambda_student']
		T = 1 if self.have_teacher else self.config['T_student']

		if not self.have_teacher:
			epochs = self.config['epochs']
		elif  self.phase == 1:
			epochs = self.config['epochs_p1']
		else:
			epochs = self.config['epochs_p2']

		iteration = 0
		best_acc = 0
		for epoch in range(epochs):
			if self.have_teacher and self.phase == 1:
				if epoch % int(epochs / self.max_temperature) == 0 and epoch > 0:
					T += 1
					print(f"temperature is {T}")

			self.student.train()
			self.adjust_learning_rate(self.optimizer, epoch, epochs)
			loss = 0

			for batch_idx, (data, target) in enumerate(self.train_loader):

				iteration += 1
				data = data.to(self.device)
				target = target.to(self.device)
				self.optimizer.zero_grad()
				output = self.student(data)

				if self.have_teacher:
					if self.phase==1:
						teacher_outputs = self.teacher(data)
						loss = F.mse_loss(output, teacher_outputs * T/self.max_temperature)
					else:
						loss = F.cross_entropy(output, target)

				else:
					loss = F.cross_entropy(output, target)

				loss.backward()
				self.optimizer.step()
			
			print("epoch {}/{}".format(epoch+1, epochs))
			val_acc = self.validate(step=epoch)
			if val_acc > best_acc:
				best_acc = val_acc
				self.save(epoch, name=self.config["checkpoint_name"])
		
		return best_acc
	
	def validate(self, step=0):
		self.student.eval()
		with torch.no_grad():
			correct = 0
			total = 0
			acc = 0
			for images, labels in self.test_loader:
				images = images.to(self.device)
				labels = labels.to(self.device)
				outputs = self.student(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

			acc = 100 * correct / total
			
			print('{{"metric": "{}_val_accuracy", "value": {}}}'.format(self.name, acc))
			return acc
	
	def save(self, epoch, name):
		trial_id = self.config['trial_id']
		torch.save({
			'model_state_dict': self.student.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'epoch': epoch,
		}, name)
	
	def adjust_learning_rate(self, optimizer, epoch, epochs):
		models_are_plane = self.config['is_plane']
		
		# depending on dataset
		if models_are_plane:
			lr = 0.01
		else:
			if epoch < int(epochs/2.0):
				lr = 0.1
			elif epoch < int(epochs*3/4.0):
				lr = 0.1 * 0.1
			else:
				lr = 0.1 * 0.01
		
		# update optimizer's learning rate
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr


if __name__ == "__main__":
	# Parsing arguments and prepare settings for training
	args = parse_arguments()
	print(args)
	config = { "lambda_student": 0.5, "T_student": 5, "seed": 20 } #nni.get_next_parameter()
	torch.manual_seed(config['seed'])
	torch.cuda.manual_seed(config['seed'])
	trial_id = os.environ.get('NNI_TRIAL_JOB_ID')
	dataset = args.dataset
	num_classes = 100 if dataset == 'cifar100' else 10
	teacher_model = None
	student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)

	assert args.student_epochs_p1 >= args.max_temperature, \
		"Numper of epoches of first phase can not be smaller than max-temperature"
	

	if args.teacher:
		teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda)
		if args.teacher_checkpoint:
			print("---------- Loading Teacher -------")
			teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
		else:

			teacher_train_config = {
				'epochs': args.teacher_epochs,
				'learning_rate': args.learning_rate,
				'momentum': args.momentum,
				'weight_decay': args.weight_decay,
				'device': 'cuda' if args.cuda else 'cpu',
				'is_plane': not is_resnet(args.student),
				'trial_id': trial_id,
				'T_student': config.get('T_student'),
				'lambda_student': config.get('lambda_student'),
				'name': args.teacher,
				'checkpoint_name': 'teacher_{}_{}_best_checkpoint.tar'.format(args.teacher, args.dataset)

			}

			print("---------- Training Teacher -------")
			train_loader, test_loader = get_cifar(num_classes, crop=True)
			teacher_trainer = TrainManager(args.max_temperature, 1, teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config)
			teacher_trainer.train()

			teacher_model = load_checkpoint(teacher_model, teacher_train_config['checkpoint_name'])
			
	# Student training
	print("---------- Training Student -------")
	student_train_config = {
		'epochs_p1': args.student_epochs_p1,
		'epochs_p2': args.student_epochs_p2,
		'learning_rate': args.learning_rate,
		'momentum': args.momentum,
		'weight_decay': args.weight_decay,
		'device': 'cuda' if args.cuda else 'cpu',
		'is_plane': not is_resnet(args.student),
		'trial_id': trial_id,
		'T_student': config.get('T_student'),
		'lambda_student': config.get('lambda_student'),
		'name': args.student,
		'checkpoint_name': 'student_{}_{}_best_checkpoint.tar'.format(args.student, args.dataset),
	}

	train_loader, test_loader = get_cifar(num_classes, crop=True)
	print('#' * 90)
	print(" " * 41 + f"phase 1" + " " * 42)
	print('#' * 90)
	student_trainer_p1 = TrainManager(args.max_temperature, 1, student_model, teacher=teacher_model, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config)
	best_student_acc = student_trainer_p1.train()
	student_model = load_checkpoint(student_model, student_train_config['checkpoint_name'])
	print('#' * 90)
	print(" " * 41 + f"phase 2" + " " * 42)
	print('#' * 90)
	student_trainer_p2 = TrainManager(args.max_temperature, 2, student_model, teacher=teacher_model,
									  train_loader=train_loader, test_loader=test_loader,
									  train_config=student_train_config)
	best_student_acc = student_trainer_p2.train()

