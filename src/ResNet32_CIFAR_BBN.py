import os
import sys
import math
import torch
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Custom_Library.util_loss import *
from Custom_Library.util_sampler import *
from Custom_Library.util_architecture import *

def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path",   type = str,   default = "/home/sailoer123/Earth/Dataset/CIFAR/CIFAR10_LT_100/")
	parser.add_argument("--source_file",   type = str,   default = "")
	parser.add_argument("--target_path",   type = str,   default = "/home/sailoer123/Earth/Result/")
	parser.add_argument("--target_file",   type = str,   default = "")
	parser.add_argument("--batch_size",    type = int,   default = 128)
	parser.add_argument("--num_epochs",    type = int,   default = 200)
	parser.add_argument("--milestone_0",   type = int,   default = 120)
	parser.add_argument("--milestone_1",   type = int,   default = 160)
	parser.add_argument("--learning_rate", type = float, default = 0.1)
	parser.add_argument("--gpu",           type = int,   default = 2)
	parser.add_argument("--random_seed",   type = int,   default = 0)
	parser.add_argument("--pretrained",    type = str,   default = "False")
	parser.add_argument("--model_path",    type = str,   default = "")
	parser.add_argument("--model_file",    type = str,   default = "model.pth")
	parser.add_argument("--weight_file",   type = str,   default = "weight.pth")
	parser.add_argument("--model",         type = str,   default = "ResNet32_CIFAR_BBN")
	parser.add_argument("--loss_function", type = str,   default = "Bilateral Branch Network")
	return parser.parse_args()


if __name__ == "__main__":
	args = set_args()

	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(args.random_seed)
	random.seed(args.random_seed)

	print("random_seed: {}".format(args.random_seed))
	print("source_path: {}".format(args.source_path))
	print("batch_size: {}".format(args.batch_size))
	print("milestone_0: {}".format(args.milestone_0))
	print("milestone_1: {}".format(args.milestone_1))
	print("learning_rate: {}".format(args.learning_rate))
	print("pre-trained: {}".format(args.pretrained))
	print("loss_function: {}".format(args.loss_function))

	start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	if os.path.exists("{}/{}/".format(args.target_path, args.model)) is False:
		os.mkdir("{}/{}/".format(args.target_path, args.model))
	writer = SummaryWriter(log_dir = "{}/{}/{}".format(args.target_path, args.model, start_time))
	with open(os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "Experiment_Description.txt"), "w") as f:

		#################################################### Data Loader ###################################################

		train_transform = transforms.Compose([transforms.ToPILImage(), transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()])
		test_transform = transforms.Compose([transforms.ToTensor()])
		train_dataset = CIFAR_Dataset(image_path = os.path.join(os.path.join(args.source_path, "Train"), "Images"), xml_path = os.path.join(os.path.join(args.source_path, "Train"), "Labels"), transform = train_transform)
		test_dataset = CIFAR_Dataset(image_path = os.path.join(os.path.join(args.source_path, "Test"), "Images"), xml_path = os.path.join(os.path.join(args.source_path, "Test"), "Labels"), transform = test_transform)
		conventional_train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size = args.batch_size, pin_memory = True)
		sample_counts = [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
		weights = [max(sample_counts) / sample_count for sample_count in sample_counts]
		reversed_sampler = ReversedSampler(sample_counts, weights, num_samples = sum(sample_counts), replacement = True)
		reversed_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, sampler = reversed_sampler, pin_memory = True)
		test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = False, batch_size = args.batch_size, pin_memory = True)

		################################################ Loss for Imbalance ################################################

		criterion = nn.CrossEntropyLoss()

		################################################## Model Training ##################################################

		model = ResNet32_CIFAR_BBN(args, writer).cuda(args.gpu)
		optimizer = torch.optim.SGD(model.parameters(), lr = 0.0, momentum = 0.9, weight_decay = 2e-4)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.milestone_0, args.milestone_1] , gamma = 0.01)
		model.fit(conventional_train_loader, reversed_train_loader, criterion, optimizer, scheduler, args.num_epochs, val_loader = test_loader, warmup_epoch = 5)	 

		################################################## Model Testing ###################################################

		accuracy, performance_matrix = model.predict(test_loader)
		weight = torch.tensor(performance_matrix/1000, dtype = torch.float32)
		torch.save(weight, os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "weight.pth"))
		labels_string = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
		df_cm = pd.DataFrame(performance_matrix, index = labels_string, columns = labels_string)
		sns.set(font_scale = 0.6)
		sns.heatmap(df_cm, annot = True, fmt = "d")
		plt.title("Confusion Matrix")
		plt.savefig(os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "Confusion_Matrix.png"))
		plt.close()
		confusion_matrix = Image.open(os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "Confusion_Matrix.png"))
		confusion_matrix = transforms.ToTensor()(confusion_matrix)
		writer.add_image(start_time + "_Confusion_Matrix", confusion_matrix)

		############################################### Experiment Recording ###############################################
		
		end_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		f.write("start_time: {}\n".format(start_time))
		f.write("end_time: {}\n".format(end_time))
		f.write("random_seed: {}\n".format(args.random_seed))
		f.write("source_path: {}\n".format(args.source_path))
		f.write("pre-trained: {}\n".format(args.pretrained))
		f.write("model_path: {}\n".format(args.model_path))
		f.write("model_file: {}\n".format(args.model_file))
		f.write("weight_file: {}\n".format(args.weight_file))
		f.write("architecture: {}\n".format(args.model))
		f.write("batch_size: {}\n".format(args.batch_size))
		f.write("num_epochs: {}\n".format(args.num_epochs))
		f.write("learning_rate: {}\n".format(args.learning_rate))
		f.write("milestone_0: {}\n".format(args.milestone_0))
		f.write("milestone_1: {}\n".format(args.milestone_1))
		f.write("accuracy: {}\n".format(accuracy))
		f.write("loss function: {}\n".format(args.loss_function))
		f.write("performance matrix:\n {}\n".format(performance_matrix/1000))
		torch.save(model.state_dict(), os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "model.pth"))
	
