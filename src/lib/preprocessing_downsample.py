import os
import math
import shutil
import random
import argparse


def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path",      type = str,   default = "C:\\Users\\atheist8E\\Documents\\Satyagraha5\\Earth\\Vanellope\\Dataset\\CIFAR_Custom\\CIFAR10\\Train\\")
	parser.add_argument("--source_file",      type = str,   default = "")
	parser.add_argument("--target_path",      type = str,   default = "C:\\Users\\atheist8E\\Documents\\Satyagraha5\\Earth\\Vanellope\\Dataset\\CIFAR_Custom\\CIFAR10_DS_2997\\Train\\")
	parser.add_argument("--target_file",      type = str,   default = "")
	parser.add_argument("--imbalance_factor", type = int,   default = 1)
	parser.add_argument("--num_labels",       type = int,   default = 10)
	parser.add_argument("--n_base",           type = int,   default = 5000)
	return parser.parse_args()

if __name__ == "__main__":
	args = set_args()
	labels = ["0", "1", "2", "5", "7", "3", "6", "4", "8", "9"]
	mu = 10 ** (math.log10(1/args.imbalance_factor) / (args.num_labels - 1))
	for i, label in enumerate(labels):
		num_images = 2997
		print("n_base: {} | mu: {:.4f} | num_images: {}".format(args.n_base, mu, num_images))
		for image in random.sample(os.listdir(os.path.join(args.source_path, label)), k = num_images):
			if os.path.exists(os.path.join(args.target_path, label)) is False:
				os.mkdir(os.path.join(args.target_path, label))
			shutil.copy(os.path.join(os.path.join(args.source_path, label),image), os.path.join(os.path.join(args.target_path, label), image))