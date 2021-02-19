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

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            # if already has  checkpoint
            print("-> loading checkping '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found ad '{}'".format(args.resume))

        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir =  os.path.join(args.data, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.106],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle = True,
            num_workers=args.workers, pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True
        )
        best_acc1= 0

        for epoch in range(args.start_epoch, args.epochs):

            # adjust learning rate
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch

            train()







def train(train_loader, model ,criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (intput, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy()





def accuracy(output, target, topk = (1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k """
    with torch.no_grad():

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # eq: equal;  expand_as 变成和括号内tensor一样的形状
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res




def adjust_learning_rate(optimizer, epoch, args):
    """ Sets the learning rate to the initial LR decayed by 10 every 30 epochs """
    lr = args.lr * ( 0.1 ** (epoch //30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val= val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count








def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.pth.tar')


