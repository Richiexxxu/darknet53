import os
import json
import torch

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import darknet53


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.106],
                                     std=[0.229, 0.224, 0.225])

    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    # load image

    # create model



