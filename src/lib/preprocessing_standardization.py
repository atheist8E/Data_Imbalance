import os
import shutil
import random
import argparse
from PIL import Image


def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path",      type = str,   default = "C:\\Users\\atheist8E\\Documents\\Satyagraha5\\Earth\\Vanellope\\Dataset\\Fundus\\Original\\")
	parser.add_argument("--source_file",      type = str,   default = "")
	parser.add_argument("--target_path",      type = str,   default = "C:\\Users\\atheist8E\\Documents\\Satyagraha5\\Earth\\Vanellope\\Dataset\\Fundus\\Processed\\")
	parser.add_argument("--target_file",      type = str,   default = "")
	parser.add_argument("--training_samples", type = int,   default = 5000)
	parser.add_argument("--test_samples",     type = int,   default = 1000)
	parser.add_argument("--class_index",      type = int,   default = 10)
	return parser.parse_args()

if __name__ == "__main__":
    args = set_args()

    if os.path.exists(os.path.join(args.target_path, "Train")) is False:
        os.mkdir(os.path.join(args.target_path, "Train"))
    if os.path.exists(os.path.join(args.target_path, "Test")) is False:
        os.mkdir(os.path.join(args.target_path, "Test"))
    if os.path.exists(os.path.join(os.path.join(args.target_path, "Train"), str(args.class_index))) is False:
        os.mkdir(os.path.join(os.path.join(args.target_path, "Train"), str(args.class_index)))
    if os.path.exists(os.path.join(os.path.join(args.target_path, "Test"), str(args.class_index))) is False:
        os.mkdir(os.path.join(os.path.join(args.target_path, "Test"), str(args.class_index)))
        
    image_fname = list()
    for fname in os.listdir(args.source_path):
        if fname[-3:] == "jpg":
            image_fname.append(fname)
    random.shuffle(image_fname)
    for fname in image_fname[:args.training_samples]:
        image = Image.open(os.path.join(args.source_path, fname))
        image_downsampled = image.resize((32, 32), resample = Image.LANCZOS)
        image_downsampled.save(os.path.join(os.path.join(os.path.join(args.target_path, "Train"), str(args.class_index)), fname))
    for fname in image_fname[-1 * args.test_samples:]:
        image = Image.open(os.path.join(args.source_path, fname))
        image_downsampled = image.resize((32, 32), resample = Image.LANCZOS)
        image_downsampled.save(os.path.join(os.path.join(os.path.join(args.target_path, "Test"), str(args.class_index)), fname))