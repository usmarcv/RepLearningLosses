from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml

import copy

import torch
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn

from pathlib import Path
from networks.resnet_big import SupCEResNet
import networks.vit as vits
from main_ce import set_loader
from util import AverageMeter, accuracy, fix_random_seeds, CustomDatasetFromCSV
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import torch.nn.functional as F

from networks.pretrained_models_cls_head import load_model_with_classification_head

import numpy as np
from argparse import Namespace

torch.is_inference_mode_enabled()
torch.serialization.add_safe_globals([Namespace])
torch.serialization.add_safe_globals([np._core.multiarray.scalar])
torch.serialization.add_safe_globals([np.dtype])
torch.serialization.add_safe_globals([np.dtypes.Float64DType])




__all_models = ['resnet50', 
                'vit_small', 'vit_base', 
                'dino_vit_small_p_16', 'dino_vit_small_p_8', 'dino_vit_base_p_16', 'dino_vit_base_p_8',
                'dinov2_vit_small_p_14', 'dinov2_vit_base_p_14']



def parse_option():

    parser = argparse.ArgumentParser('Arguments for Inferences...')
    
    # parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    # parser.add_argument('--save_freq', type=int, default=25, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    # parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # model dataset
    # Aqui tem as alterações para o modelo e dataset // mexer aqui
    parser.add_argument('--model', type=str, default=None, choices=['resnet50', 
                                                                          'vit_small', 'vit_base',
                                                                          'dino_vit_small_p_16', 'dino_vit_small_p_8', 
                                                                          'dino_vit_base_p_16', 'dino_vit_base_p_8'], 
                        help='Choose your backbone')
    parser.add_argument('--n_cls', type=int, default=10, help='number of classes') #For CIFAR10
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet100', 'imagenet', 'cifar2',
                                 'aircraft', 'cars', 'food101', 'pet', 'dtd', 'flowers', 'path'],
                        help='dataset')
    parser.add_argument('--size', type=int, default=224, help='size of images after resizing')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')


    # other setting
    # parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    # parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--ckpt', type=str, default='', help='path to pre-trained model')


    parser.add_argument('--root_path', type=str, default='', help='root path to dataset')
    # We have used the cross-validation mode to search our hyperparameters
    # parser.add_argument('--train_mode', type=str, default='holdout', 
    #                     choices=['holdout', 'cross-validation', 'contrastive-mode'],
    #                     help='Choose your training mode and set the csv file accordingly')
    
    parser.add_argument('--train_file', type=str, default=None, help='csv file for training')
    # parser.add_argument('--val_file', type=str, default=None, help='csv file for validation')
    parser.add_argument('--test_file', type=str, default=None, help='csv file for testing')
    # parser.add_argument('--num_folds', type=int, default=None, help='Number of folds for cross-validation based on your csv file')
    parser.add_argument('--seed', default=31, type=int, help='Random seed')

  
    opt = parser.parse_args()

    # check if settings are correct
    # assert opt.ckpt is not None  # need pre trained model
    

    # Priting arguments for logging
    print("\n[INFO] Printing arguments for Inferences stage...")
    print(opt)
    
    print("\n[INFO] Inferences with gpu: {}".format(torch.cuda.get_device_name()))



    return opt


def set_loader(opt:str, contrast_trans:bool=False, for_test:bool=True):

    #Normalization based on ImageNet dataset: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if for_test:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize([opt.size, opt.size]),
                transforms.ToTensor(),
                normalize,
            ])
    
        val_transform = test_transform
    
    train_dataset = CustomDatasetFromCSV(
            path_root=opt.root_path,
            tf_image=train_transform,
            csv_name=opt.train_file,
            val_fold=None,
            train=True
        )
    
    val_dataset = CustomDatasetFromCSV(
            path_root=opt.root_path,
            tf_image=val_transform,
            csv_name=opt.train_file,
            val_fold=None,
            train=True
        )
    
    test_dataset = CustomDatasetFromCSV(
            path_root=opt.root_path,
            tf_image=test_transform,
            csv_name=opt.test_file,
            val_fold=None,
            train=False
        )

    train_loader = DataLoader(train_dataset, 
                              batch_size=opt.batch_size, 
                              shuffle=True, 
                              num_workers=opt.num_workers,
                              pin_memory=True,
                              prefetch_factor=2)
  
    valid_loader = DataLoader(val_dataset, 
                                batch_size=opt.batch_size, 
                                shuffle=False, 
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                prefetch_factor=2)
    
    test_loader = DataLoader(test_dataset, 
                                batch_size=opt.batch_size, 
                                shuffle=False, 
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                prefetch_factor=2)
        
    # print('[INFO] Training with cross-validation mode...')
        
    return train_loader, valid_loader, test_loader



def set_model(opt:str):

    print('\n[INFO] Setting model and criterion with linear classifier...')

    # model = None
    # classifier = None
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model
    if opt.model == "vit_small" or opt.model == "dino_vit_small_p_16":
        # model = vits.SupCEViT(name=opt.model, feat_dim=384, num_classes=opt.n_cls)
        model = vits.SupConViT(name=opt.model, feat_dim=384)
        embed_dim = model.encoder.embed_dim 
        classifier = vits.LinearClassifier(dim=embed_dim, num_labels=opt.n_cls)
    elif opt.model == "vit_base" or opt.model == "dino_vit_base_p_16":
        model = vits.SupCEViT(name=opt.model, feat_dim=768, num_classes=opt.n_cls)
    elif opt.model == "resnet50": 
        model = SupCEResNet(name=opt.model)
    else:
        raise ValueError('Model not supported: {}'.format(opt.model))
    
    # model.eval()
    # classifier.eval()
    # Load the checkpoint if available (Basically, if we are using a pre-trained model or runned from Stage 1)
    # original_state_dict = copy.deepcopy(model.state_dict())
    

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        classifier = classifier.cuda()
        # cudnn.benchmark = True
    
    state_dict = torch.load(opt.ckpt, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state_dict, strict=True)
    # model.eval()

    # # Após carregar
    # new_state_dict = model.state_dict()

    # # Verificar se houve mudança
    # for key in original_state_dict:
    #     if not torch.equal(original_state_dict[key].cpu(), new_state_dict[key].cpu()):
    #         print(f"[OK] Parâmetro '{key}' foi atualizado.")
    #     else:
    #         print(f"[WARNING] Parâmetro '{key}' permaneceu igual.")

    # return model, classifier
    return model, classifier


@torch.no_grad()
def test(test_loader, model, classifier, opt):

    """testing"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))

            # outut = classifier(images)
          
            if opt.n_cls > 4:
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top5.update(acc5[0], bsz)
            else:
                acc1 = accuracy(output, labels, topk=(1,))[0]
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0 or idx == len(test_loader) - 1:
                print(f'Test: [{idx}/{len(test_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')

    print('\t[INFO] * Average testing: Acc@1 {top1.avg:.3f} | Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return  top1.avg

# def test(test_loader, model, classifier, opt):
# def test(test_loader, model, opt):
#     print('\n[INFO] Start Testing...')

#     batch_time = AverageMeter()
#     top1 = AverageMeter()

#     if opt.n_cls > 4:
#         top5 = AverageMeter()

#     model.eval()

#     with torch.inference_mode():
#     # with torch.no_grad():
#         end = time.time()
#         for idx, (images, labels) in enumerate(test_loader):
#             images = images.float().cuda(non_blocking=True)
#             labels = labels.cuda(non_blocking=True)
#             bsz = labels.shape[0]

#             output = model(images)

#             if opt.n_cls > 4:
#                 acc1, acc5 = accuracy(output, labels, topk=(1, min(opt.n_cls, 5)))
#                 top1.update(acc1.item(), bsz)
#                 top5.update(acc5.item(), bsz)
#             else:
#                 acc1 = accuracy(output, labels, topk=(1,))[0]
#                 top1.update(acc1.item(), bsz)

#             batch_time.update(time.time() - end)
#             end = time.time()

#             if idx % 10 == 0 or idx == len(test_loader) - 1:
#                 print(f'Test: [{idx}/{len(test_loader)}]\t'
#                       f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')

#     print(f'\n[INFO] Final Testing Accuracy:')
#     print(f'   Acc@1: {top1.avg:.2f}%')
#     if opt.n_cls > 4:
#         print(f'   Acc@5: {top5.avg:.2f}%')
#         return top1.avg, top5.avg
#     else:
#         return top1.avg, None


def main():

    opt = parse_option()
    
    # fix_random_seeds(seed=opt.seed)

    # build data loader
    _, _, test_loader = set_loader(opt, contrast_trans=False, for_test=True)

    # model, classifier = set_model(opt)
    model, classifier = set_model(opt)
    
    # test(test_loader, model, classifier, opt)
    
    test(test_loader, model, classifier, opt)


if __name__ == '__main__':
    main()