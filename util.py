from __future__ import print_function

import os
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from typing import Sequence
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset


class SubsetWithTargets(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)
        self.targets = list(torch.tensor(dataset.targets)[indices])


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class DoubleTransform:
    """Return array with two transforms of an image"""
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average, current value, and history"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = [] 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val) 


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):

    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):

    print('\n[INFO] ==> Saving...')

    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    torch.save(state, save_file)

    del state


#Based on: https://github.com/eriksonJAguiar/RADAR_MIX/blob/main/utils/utils.py#L530
class CustomDatasetFromCSV(Dataset):
    """
    Custom dataset for loading images from a CSV file with columns: image_path, label, fold
    Automatically splits into train/val based on a fold.

    Returns:
        X (Image): Image data
        y (int): Label
    """    
        
    def __init__(self, path_root, tf_image=None, csv_name=None, val_fold=None, train=True):
        """_summary_

        Args:
            path_root (_type_): _description_
            tf_image (_type_, optional): _description_. Defaults to None.
            csv_name (_type_, optional): _description_. Defaults to None.
            val_fold (_type_, optional): _description_. Defaults to None.
            train (bool, optional): _description_. Defaults to True.
        """

        self.data = pd.read_csv(csv_name)
        self.tf_image = tf_image
        self.root = path_root
        self.cl_name = {c: i for i, c in enumerate(np.unique(self.data["label"]))}
        if val_fold is not None:
            if train:
                self.data = self.data[self.data["fold"] != val_fold]  #Use all folds except the validation fold
            else:
                self.data = self.data[self.data["fold"] == val_fold]  #Use only the validation fold
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x_path = os.path.join(self.root, self.data.iloc[idx, 0])
        y = self.cl_name[self.data.iloc[idx, 1]]
        
        X = Image.open(x_path).convert("RGB")  # Always convert to RGB

        if self.tf_image:
            X = self.tf_image(X)
                
        return X, y