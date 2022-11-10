# MIT License

# Copyright (c) 2019 Iman Mirzadeh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import torch
import torchvision
import torchvision.transforms as transforms

NUM_WORKERS = 2


def get_cifar(num_classes=100, dataset_dir='./data', batch_size=128, crop=False):
	"""
	:param num_classes: 10 for cifar10, 100 for cifar100
	:param dataset_dir: location of datasets, default is a directory named 'data'
	:param batch_size: batchsize, default to 128
	:param crop: whether or not use randomized horizontal crop, default to False
	:return:
	"""
	normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
	simple_transform = transforms.Compose([transforms.ToTensor(), normalize])
	
	if crop is True:
		train_transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		])
	else:
		train_transform = simple_transform
	
	if num_classes == 100:
		trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True,
												 download=True, transform=train_transform)
		
		testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False,
												download=True, transform=simple_transform)
	else:
		trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
												 download=True, transform=train_transform)
		
		testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
												download=True, transform=simple_transform)
		
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=NUM_WORKERS,
											  pin_memory=True, shuffle=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=NUM_WORKERS,
											 pin_memory=True, shuffle=False)
	return trainloader, testloader


if __name__ == "__main__":
	print("CIFAR10")
	print(get_cifar(10))
	print("---"*20)
	print("---"*20)
	print("CIFAR100")
	print(get_cifar(100))
