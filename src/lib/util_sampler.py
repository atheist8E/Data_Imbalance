import os
import torch
import numpy as np
from skimage import io
import torch.nn.functional as F
import torchvision.transforms as T
from xml.etree.ElementTree import parse


class CIFAR_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, xml_path, transform):
        super().__init__()
        self.xml_path = xml_path
        self.image_path = image_path
        self.xml_fnames = sorted(os.listdir(xml_path))
        self.image_fnames = sorted(os.listdir(image_path))
        self.transform = transform
   
    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
        image = torch.tensor(io.imread(os.path.join(self.image_path, self.image_fnames[idx])), dtype = torch.float)
        image = torch.permute(image, (2, 0, 1))
        root = parse(os.path.join(self.xml_path, self.xml_fnames[idx])).getroot()
        label = int(root.find("label").text)
        return image, label
