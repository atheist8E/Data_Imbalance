import os
import argparse
import numpy as np
import collections
from PIL import Image
import tensorflow as tf

def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path", type = str, default = "C:\\Users\\atheist8E\\Documents\\Satyagraha5\\Earth\\Vanellope\\Dataset\\CIFAR_\\CIFAR10_LT_20\\")
	parser.add_argument("--source_file", type = str, default = "eval.tfrecords")
	parser.add_argument("--target_path", type = str, default = "C:\\Users\\atheist8E\\Documents\\Satyagraha5\\Earth\\Vanellope\\Dataset\\CIFAR_\\CIFAR10_LT_20\\Test")
	parser.add_argument("--target_file", type = str, default = "")
	return parser.parse_args()

def _parse_image_function(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

if __name__ == "__main__":
    args = set_args()
    raw_dataset = tf.data.TFRecordDataset(os.path.join(args.source_path, args.source_file))
    parsed_image_dataset = raw_dataset.map(_parse_image_function)
    count = collections.defaultdict(int)
    labels_string = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for features in parsed_image_dataset:
        label = features["label"]
        count[label.numpy()] += 1
        if os.path.exists(os.path.join(args.target_path, str(label.numpy()))) is False:
            os.mkdir(os.path.join(args.target_path, str(label.numpy())))
        image = features["image"].numpy()
        image = np.frombuffer(image, dtype = np.int8)
        image = image.reshape((32,32,3), order = "F").swapaxes(0, 1)
        image = Image.fromarray(image, mode = "RGB")
        image.save(os.path.join(os.path.join(args.target_path, str(label.numpy())), labels_string[label.numpy()] + "_" + str(count[label.numpy()]) + ".png"), format="PNG")
    print(count)
