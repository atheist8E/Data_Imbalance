import os
import cv2
import math
import shutil
import random
import argparse
from xml.etree.ElementTree import parse
from collections import defaultdict


def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path",      type = str,   default = "/home/sailoer123/Earth/Dataset/OAI/Sclerosis/Trabecular_Bone/Original/Train_SCTL/")
	parser.add_argument("--source_file",      type = str,   default = "")
	parser.add_argument("--target_path",      type = str,   default = "")
	parser.add_argument("--target_file",      type = str,   default = "")
	parser.add_argument("--imbalance_factor", type = int,   default = 1)
	parser.add_argument("--num_labels",       type = int,   default = 10)
	parser.add_argument("--n_base",           type = int,   default = 5000)
	return parser.parse_args()

if __name__ == "__main__":
	args = set_args()
	frequency = defaultdict(int)
	img_path = os.path.join(args.source_path, "Images")
	xml_path = os.path.join(args.source_path, "Labels")
	for label in os.listdir(xml_path):
		fname = label[:-4]
		image = cv2.imread(os.path.join(img_path, fname + ".jpg"), flags = cv2.IMREAD_COLOR)
		#image = cv2.resize(image, (32, 32))
		try:
			image = cv2.resize(image, (32, 32))
		except:
			print(label)
			continue
		else:
			tree = parse(os.path.join(xml_path, label))
			root = tree.getroot()
			label = int(root.find("Medial").find("SCTM").text)
			frequency[label] += 1
		"""
		tree = parse(os.path.join(xml_path, label))
		root = tree.getroot()
		label = int(root.find("Medial").find("SCTM").text)
		frequency[label] += 1
		"""
	print(frequency)
