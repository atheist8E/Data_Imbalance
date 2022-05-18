import os
import sys
import math
import torch
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

from lib.util_sampler import *
from lib.util_architecture import *

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path",   type = str,   default = "../dat/CIFAR10_LT_100/")
    parser.add_argument("--source_file",   type = str,   default = "")
    parser.add_argument("--target_path",   type = str,   default = "../log/")
    parser.add_argument("--target_file",   type = str,   default = "")
    parser.add_argument("--batch_size",    type = int,   default = 256)
    parser.add_argument("--num_epochs",    type = int,   default = 200)
    parser.add_argument("--milestone_0",   type = int,   default = 120)
    parser.add_argument("--milestone_1",   type = int,   default = 160)
    parser.add_argument("--learning_rate", type = float, default = 0.1)
    parser.add_argument("--gpu",           type = int,   default = 0)
    parser.add_argument("--pretrained",    type = str,   default = "False")
    parser.add_argument("--model_path",    type = str,   default = "")
    parser.add_argument("--model",         type = str,   default = "ResNet32_CIFAR_Naive")
    parser.add_argument("--description",   type = str,   default = "Softmax Cross Entropy Loss")
    return parser.parse_args()


if __name__ == "__main__":

    args = set_args()

    print("source_path: {}".format(args.source_path))
    print("batch_size: {}".format(args.batch_size))
    print("num_epochs: {}".format(args.num_epochs))
    print("milestone_0: {}".format(args.milestone_0))
    print("milestone_1: {}".format(args.milestone_1))
    print("description: {}".format(args.description))

    start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if os.path.exists("{}/{}/".format(args.target_path, args.model)) is False:
        os.mkdir("{}/{}/".format(args.target_path, args.model))
    writer = SummaryWriter(log_dir = "{}/{}/{}".format(args.target_path, args.model, start_time))
    with open(os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "Experiment_Description.txt"), "w") as f:

		#################################################### Data Loader ###################################################

        train_transform = transforms.Compose([transforms.ToPILImage(), transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = CIFAR_Dataset(image_path = os.path.join(os.path.join(args.source_path, "Train"), "Images"), \
                                      xml_path = os.path.join(os.path.join(args.source_path, "Train"), "Labels"), transform = train_transform)
        test_dataset = CIFAR_Dataset(image_path = os.path.join(os.path.join(args.source_path, "Test"), "Images"), \
                                      xml_path = os.path.join(os.path.join(args.source_path, "Test"), "Labels"), transform = test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size = args.batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = False, batch_size = args.batch_size)

		################################################## Model Training ##################################################

        model = resnet32(args, writer).cuda(args.gpu)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.0, momentum = 0.9, weight_decay = 2e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.milestone_0, args.milestone_1] , gamma = 0.01)
        model.fit(train_loader, criterion, optimizer, scheduler, args.num_epochs)	 

		################################################## Model Testing ###################################################

        accuracy, confusion_matrix_ = model.evaluate(test_loader)
        weight = torch.tensor(confusion_matrix_, dtype = torch.float32)
        torch.save(weight, os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "weight.pth"))
        labels_string = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        df_cm = pd.DataFrame(confusion_matrix_, index = labels_string, columns = labels_string)
        sns.set(font_scale = 0.6)
        sns.heatmap(df_cm, annot = True, fmt = "d")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "Confusion_Matrix.png"))
        plt.close()
        confusion_matrix_ = Image.open(os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "Confusion_Matrix.png"))
        confusion_matrix_ = transforms.ToTensor()(confusion_matrix_)
        writer.add_image(start_time + "_Confusion_Matrix", confusion_matrix_)

		############################################### Experiment Recording ###############################################
		
        end_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        f.write("source_path: {}\n".format(args.source_path))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("num_epochs: {}\n".format(args.num_epochs))
        f.write("milestone_0: {}\n".format(args.milestone_0))
        f.write("milestone_1: {}\n".format(args.milestone_1))
        f.write("learning_rate: {}\n".format(args.learning_rate))
        f.write("description: {}\n".format(args.description))
        f.write("accuracy: {}\n".format(accuracy))
        f.write("confusion matrix:\n {}\n".format(confusion_matrix_))
        torch.save(model.state_dict(), os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "model.pth"))
