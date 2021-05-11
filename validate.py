from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import argparse
import copy

plt.ion()   # interactive mode

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn

import pathlib
from model import Net

from IPython import embed

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='PyTorch GAIL')

parser.add_argument('--aug', '-aug', type=int, default=0)
parser.add_argument('--k', '-k', type=int, default=1)
parser.add_argument('--file_path', '-sfp', type=str, default='saved_mlp.pt')
args = parser.parse_args()

def evaluateModelOnValidationSet():

	# Create a pytorch dataset
	data_dir = pathlib.Path('./data/tiny-imagenet-200')
	# image_count = len(list(data_dir.glob('**/*.JPEG')))
	CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
	# print('Discovered {} images'.format(image_count))

	assert(len(CLASS_NAMES) == 200)

	# Create the validation data generator
	batch_size = 64
	im_height = 64
	im_width = 64

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	# Should data augmentation be performed on the training data?
	if args.aug == 1:
		print("Validation with Aug")
		validation_data_transforms = transforms.Compose([
			# transforms.ColorJitter(brightness = 1, contrast = 1, saturation = 1, hue = [-0.2,0.2]),
			# transforms.RandomAffine(degrees = 20, translate = [0.2, 0.2], scale = None, shear = [-5,5]),
			transforms.RandomGrayscale(p = 0.15),
			transforms.RandomHorizontalFlip(p = 0.35),
			transforms.RandomVerticalFlip(p = 0.35),
			transforms.RandomRotation(degrees = 5),
			transforms.RandomPerspective(p = 0.2),


			transforms.ToTensor(),
			normalize
		])
	else:
		print("Validation without Aug")
		validation_data_transforms = transforms.Compose([
			transforms.ToTensor(),
			normalize
		])

	dataPathString = './data/tiny-imagenet-200'

	# Create the validation data generator
	validation_set = torchvision.datasets.ImageFolder(dataPathString + '/val/data', validation_data_transforms)
	validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size,
										   shuffle = True, num_workers = 1, pin_memory = True)

	# Size of Validation Data Set
	validationDataLength = len(validation_set)
	assert(validationDataLength == 10000)

	# Load Model and its Weights
	ckpt = torch.load(args.file_path)
	model = Net(200, im_height, im_width)
	model.load_state_dict(ckpt['net'])
	model = model.cuda()

	# Put the model in evaluation mode (to test on validation data)
	model.eval()

	running_loss = 0.0
	running_corrects = 0
	# Loop through validation batches
	for idx, (inputs, targets) in enumerate(tqdm(validation_loader)):

		inputs = inputs.to(device)
		targets = targets.to(device)

		# Run the model on the validation batch
		outputs = model(inputs)

		# Get validation loss and validation accuracy on this batch
		criterion = nn.CrossEntropyLoss()
		loss = criterion(outputs, targets)
		_, preds = torch.max(outputs, 1)

		# Keep tracking of running statistics on validation loss and accuracy

		values,indices = outputs.topk(args.k)

		running_loss += loss.item() * inputs.size(0)
		for i in range(len(targets.data)):
			if targets.data[i].cpu().item() in indices[i]:
				running_corrects += 1

	validationLoss = running_loss / validationDataLength
	validationAccuracy = running_corrects / validationDataLength

	return validationLoss, validationAccuracy


if __name__ == '__main__':
	validationLoss, validationAccuracy = evaluateModelOnValidationSet()
	print("validationLoss is: " + str(validationLoss))
	print("validationAccuracy is: " + str(validationAccuracy))
