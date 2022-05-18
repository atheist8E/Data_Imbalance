import os
import pickle
import argparse
import numpy as np
from PIL import Image


def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path", type = str, default = "C:\\Users\\atheist8E\\Documents\\Satyagraha5\\Earth\\Vanellope\\Dataset\\")
	parser.add_argument("--source_file", type = str, default = "data.pickle")
	parser.add_argument("--target_path", type = str, default = "C:\\Users\\atheist8E\\Documents\\Satyagraha5\\Earth\\Vanellope\\Dataset\\CIFAR10\\")
	parser.add_argument("--target_file", type = str, default = "")
	return parser.parse_args()

if __name__ == "__main__":
	args = set_args()
	with open(os.path.join(args.source_path, args.source_file), "rb") as f:
		raw_data = pickle.load(f, encoding="bytes")
		print(raw_data.keys())
		for label, data, fname in zip(raw_data[b"labels"], raw_data[b"data"], raw_data[b"filenames"]):
			image = data.reshape((32,32,3), order = "F").swapaxes(0, 1)
			image = Image.fromarray(image, mode = "RGB")
			if os.path.exists(os.path.join(args.target_path, str(label))) is False:
				os.mkdir(os.path.join(args.target_path, str(label)))
			image.save(os.path.join(os.path.join(args.target_path, str(label)), fname.decode("utf-8")), format="PNG")
