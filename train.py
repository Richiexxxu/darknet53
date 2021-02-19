import os
import argparse
import time
import shutil

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
import torchvision.datasets as datasets

from model import darknet53


parser = argparse.ArgumentParser(description='Darknet53 ImageNet Training')
parser.add_argument('--data', metavar='DIR', help='path to datset')
parser.add_argument('--resume', default='', type=str, metavar='Path', help='path to latest checkpoint (default:none)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch')
parser.add_argument('--workers')
parser.add_argument('--lr')
parser.add_argument('--momentum')
parser.add_argument('--wd')
parser.add_argument('--b')
parser.add_argument('--p')
parser.add_argument('--gpu')



def main():
    args = parser.parse_args()
    if args.gpu is not None:
        print("Use, GPU : {} for training".format(args.gpu))


    model = darknet53(1000) # change 1000 to num_classes, put the 1000 classes into parser.add_argument default

    if args.gpu in not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    # define loss function (criterion) and optimizer

















def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.pth.tar')


