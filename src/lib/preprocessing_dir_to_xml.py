import os
import shutil
import argparse
from xml.etree.ElementTree import Element, SubElement, ElementTree

def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path", type = str, default = "C:\\Users\\atheist8E\\Documents\\Satyagraha5\\Earth\\Vanellope\\Dataset\\CIFAR_Custom\\CIFAR10_LT_100\\Train\\")
	parser.add_argument("--source_file", type = str, default = "")
	parser.add_argument("--target_path", type = str, default = "")
	parser.add_argument("--target_file", type = str, default = "")
	return parser.parse_args()

if __name__ == "__main__":
	args = set_args()
	for label in os.listdir(args.source_path):
		for image in os.listdir(os.path.join(args.source_path, label)):
			fname = image[:-4] + ".xml"
			root = Element("Label")
			SubElement(root, "label").text = label
			tree = ElementTree(root)
			if os.path.exists(os.path.join(args.source_path, "Labels")) is False:
				os.mkdir(os.path.join(args.source_path, "Labels"))
			tree.write(os.path.join(os.path.join(args.source_path, "Labels"), fname))
			if os.path.exists(os.path.join(args.source_path, "Images")) is False:
				os.mkdir(os.path.join(args.source_path, "Images"))
			shutil.copy(os.path.join(os.path.join(args.source_path, label), image), os.path.join(os.path.join(args.source_path, "Images"), image))
