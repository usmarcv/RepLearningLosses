from __future__ import print_function

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset

import os
import pandas as pd
from PIL import Image


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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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




# class CustomDatasetFromCSV(Dataset):
#     """Custom dataset para importar imagens a partir de um CSV, com suporte a folds."""
    
#     def __init__(self, path_root, tf_image, csv_name, as_rgb=False, val_fold=None, is_val=False):
#         """
#         Args:
#             path_root (str): Diretório raiz onde as imagens estão armazenadas.
#             tf_image (callable): Transformações a serem aplicadas às imagens.
#             csv_name (str): Caminho para o arquivo CSV contendo os metadados.
#             as_rgb (bool, optional): Se True, converte as imagens para RGB. Default é False.
#             val_fold (int, optional): Número do fold usado para validação.
#             is_val (bool, optional): Define se o dataset será de validação.
#         """
#         self.full_data = pd.read_csv(csv_name)
#         self.as_rgb = as_rgb
#         self.tf_image = tf_image
#         self.root = path_root
#         self.cl_name = {c: i for i, c in enumerate(np.unique(self.full_data['label']))}
#         self.BARVALUE = "/" if not os.name == "nt" else "\\"

#         # Filtra apenas os dados relevantes para treino ou validação
#         if val_fold is not None:
#             if is_val:
#                 self.data = self.full_data[self.full_data["fold"] == val_fold]  # Somente validação
#             # else:
#                 self.data = self.full_data[self.full_data["fold"] != val_fold] # Somente treino
#         else:
#             self.data = self.full_data  # Usa todos os dados se nenhum fold for passado

#     def __len__(self):
#         return len(self.full_data)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         x_path = os.path.join(self.root, self.full_data.iloc[idx, 0])
#         y = self.cl_name[self.full_data.iloc[idx, 1]]
        
#         X = Image.open(x_path)
#         if self.as_rgb:
#             X = X.convert("RGB")
        
#         if self.tf_image:
#             X = self.tf_image(X)
        
#         return X, y


# class CustomDatasetFromCSV(Dataset):
#     """Custom dataset para importar imagens a partir de um CSV, com suporte a folds."""
    
#     def __init__(self, path_root, tf_image, csv_name, as_rgb=False, val_fold=None):
#         """
#         Args:
#             path_root (str): Diretório raiz onde as imagens estão armazenadas.
#             tf_image (callable): Transformações a serem aplicadas às imagens.
#             csv_name (str): Caminho para o arquivo CSV contendo os metadados.
#             as_rgb (bool, optional): Se True, converte as imagens para RGB. Default é False.
#         """
#         self.data = pd.read_csv(csv_name)
#         self.as_rgb = as_rgb
#         self.tf_image = tf_image
#         self.root = path_root
#         self.cl_name = {c: i for i, c in enumerate(np.unique(self.data['label']))}
#         self.BARVALUE = "/" if not os.name == "nt" else "\\"

#         if val_fold is not None:
#             self.train_data = self.data[self.data["fold"] != val_fold]  # Dados de treino (exclui o fold de validação)
#             self.val_data = self.data[self.data["fold"] == val_fold]    # Dados de validação (apenas o fold selecionado)
#         # else:
#         #     self.train_data = self.data  # Se nenhum fold for passado, usa todos os dados para treino
#         #     self.val_data = pd.DataFrame(columns=self.data.columns)  # DataFrame vazio para evitar erros
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         x_path = os.path.join(self.root, self.data.iloc[idx, 0])
#         y = self.cl_name[self.data.iloc[idx, 1]]
        
#         X = Image.open(x_path)
#         if self.as_rgb:
#             X = X.convert("RGB")
        
#         if self.tf_image:
#             X = self.tf_image(X)
        
#         return X, y

#Copy pasted from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataset
# class CustomImageDataset(Dataset):

#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# class CustomDatasetFromCSV(Dataset):
#     """
#     Custom dataset to load images from a CSV file.
#     CSV structure: image_path,label,fold
#     """    
#     def __init__(self, path_root, tf_image=None, csv_name=None, as_rgb=False, fold=None, exclude_fold=False):
#         self.data = pd.read_csv(csv_name)
#         self.as_rgb = as_rgb
#         if fold is not None:
#             if exclude_fold:
#                 self.data = self.data[self.data["fold"] != fold]
#             else:
#                 self.data = self.data[self.data["fold"] == fold]
        
#         self.tf_image = tf_image
#         self.root = path_root
        
#         # Map class labels to indices
#         self.cl_name = {c: i for i, c in enumerate(np.unique(self.data["label"]))}
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         x_path = os.path.join(self.root, self.data.iloc[idx, 0])
#         y = self.cl_name[self.data.iloc[idx, 1]]
        
#         X = Image.open(x_path)
#         if self.as_rgb:
#             X = X.convert("RGB")
        
#         if self.tf_image:
#             X = self.tf_image(X)
        
#         return X, y

class CustomDatasetFromCSV(Dataset):
    """
    Custom dataset for loading images from a CSV file with columns: image_path, label, fold
    Automatically splits into train/val based on a fold.
    """    
    def __init__(self, path_root, tf_image=None, csv_name=None, val_fold=None, train=True):
        self.data = pd.read_csv(csv_name)
        
        if val_fold is not None:
            if train:
                self.data = self.data[self.data["fold"] != val_fold]  # Usar todos os folds exceto o de validação
            else:
                self.data = self.data[self.data["fold"] == val_fold]  # Usar apenas o fold de validação
        
        self.tf_image = tf_image
        self.root = path_root
        self.cl_name = {c: i for i, c in enumerate(np.unique(self.data["label"]))}
    
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
        
        # print(f"Shape da imagem carregada: {X.shape}")  # Debugando o tamanho da imagem
        
        return X, y


# class CustomDatasetFromCSV(Dataset):
#     """Generating custom dataset for importing images from csv
#     """    
#     def __init__(self, path_root, tf_image, csv_name, as_rgb=False, task=None):

#         self.data = pd.read_csv(csv_name)
#         self.as_rgb = as_rgb
#         if task is not None:
#             self.data.query("Fold == @task", inplace=True)
#         self.tf_image = tf_image
#         self.root = path_root
#         self.cl_name = {c: i for i, c in enumerate(np.unique(self.data["label"]))}
#         self.BARVALUE = "/" if not os.name == "nt" else "\\"
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         x_path = os.path.join(self.root, self.data.iloc[idx, 0])
#         y = self.cl_name[self.data.iloc[idx, 1]]
        
#         X = Image.open(x_path).convert("RGB") 
        
#         if self.tf_image:
#             X = self.tf_image(X)
        
#         return X, y
    
# class CustomDatasetFromCSV(Dataset):
#     """Custom dataset para importar imagens a partir de um CSV, com suporte a folds."""
    
#     def __init__(self, path_root, tf_image, csv_name, as_rgb=False, fold=None):
#         """
#         Args:
#             path_root (str): Diretório raiz onde as imagens estão armazenadas.
#             tf_image (callable): Transformações a serem aplicadas às imagens.
#             csv_name (str): Caminho para o arquivo CSV contendo os metadados.
#             as_rgb (bool, optional): Se True, converte as imagens para RGB. Default é False.
#             task (str, optional): Filtra os dados para uma tarefa específica.
#             fold (int, optional): Filtra os dados para um fold específico.
#         """
#         self.data = pd.read_csv(csv_name)
#         self.as_rgb = as_rgb
        
#         if fold is not None:
#             self.train_data = self.data[self.data["fold"] != fold].index.tolist() # train data is all folds except fold validation
#             self.val_data = self.data[self.data["fold"] == fold].index # only validation fold data is used
        
#         self.tf_image = tf_image
#         self.root = path_root
#         self.cl_name = {c: i for i, c in enumerate(np.unique(self.data["label"]))}
#         self.BARVALUE = "/" if not os.name == "nt" else "\\"
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         x_path = os.path.join(self.root, self.data.iloc[idx, 0])
#         y = self.cl_name[self.data.iloc[idx, 1]]
        
#         X = Image.open(x_path)
#         if self.as_rgb:
#             X = X.convert("RGB")
        
#         if self.tf_image:
#             X = self.tf_image(X)
        
#         return X, y

