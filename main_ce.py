from __future__ import print_function

import argparse
import copy
import os
import math
import sys
import time

from sklearn.model_selection import train_test_split
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torchvision import transforms, datasets

import sampler
from util import AverageMeter, DoubleTransform, SubsetWithTargets, TwoCropTransform, CustomDatasetFromCSV
from util import adjust_learning_rate, warmup_learning_rate, accuracy, set_optimizer, save_model, fix_random_seeds


import networks.vit as vits
from networks.resnet_big import SupCEResNet


def parse_option():

    parser = argparse.ArgumentParser('Argument for training')

    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=25, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='20, 30, 40, 70', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')


    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--n_cls', type=int, default=None, help='number of classes')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet100', 'imagenet', 'path'],
                        help='dataset')

    # parser.add_argument('--valid_split', type=float, default=0,
    #                     help="proportion of train data to use for validation set")
    # parser.add_argument('--mean', type=str,
    #                     help='mean of dataset in path in form of str tuple')
    # parser.add_argument('--std', type=str,
    #                     help='std of dataset in path in form of str tuple')
    # parser.add_argument('--data_folder', type=str,
    #                     default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32,
                        help='size of images after resizing')
    

    parser.add_argument('--root_path', type=str, default='', help='root path to dataset')
    parser.add_argument('--train_mode', type=str, default='holdout', 
                        choices=['holdout', 'cross-validation', 'contrastive-mode'],
                        help='Choose your training mode and set the csv file accordingly')
    
    parser.add_argument('--train_file', type=str, default='Datasets/KFolds/SKF_TRAIN_Fold_1.csv', help='csv file for training')
    parser.add_argument('--val_file', type=str, default='Datasets/KFolds/SKF_VAL_Fold_1.csv', help='csv file for validation')
    parser.add_argument('--test_file', type=str, default='Datasets/test.csv', help='csv file for testing')
    parser.add_argument('--num_folds', type=int, default=None, help='Number of folds for cross-validation based on your csv file')
        

    # other setting
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    # parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',  help='id for recording multiple runs')
    parser.add_argument('--seed', default=31, type=int, help='Random seed')

    opt = parser.parse_args()


    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        #opt.data_folder is not None
        # opt.mean is not None
        # opt.std is not None
        opt.n_cls is not None

    # set the path according to the environment
    # if opt.data_folder is None:
    #     opt.data_folder = './datasets/'

    opt.model_path = './save/CE/{}_models'.format(opt.dataset)
    opt.tb_path = './save/CE/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'path':
        pass
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
     # Priting arguments for logging
    print("\n[INFO] Printing arguments for pre-training stage...")
    print(opt)
    
    print("\n[INFO] Training with gpu: {}\n".format(torch.cuda.get_device_name()))

    return opt


# def set_loader(opt, contrast_trans=False, valid=False):
    
#     if opt.dataset == 'cifar10':
#         mean = (0.4914, 0.4822, 0.4465)
#         std = (0.2023, 0.1994, 0.2010)
#     elif opt.dataset == 'cifar100':
#         mean = (0.5071, 0.4867, 0.4408)
#         std = (0.2675, 0.2565, 0.2761)
#     elif opt.dataset == 'path':
#         #ImageNet mean and std
#         mean = (0.485, 0.456, 0.406)
#         std = (0.229, 0.224, 0.225)
#     else:
#         raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
#     normalize = transforms.Normalize(mean=mean, std=std)

#     if contrast_trans:
#         #both images heavily augmented based SupCon repo: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py#L146
#         train_transform = TwoCropTransform(transforms.Compose([
#                             transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.RandomApply([
#                                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#                             ], p=0.8),
#                             transforms.RandomGrayscale(p=0.2),
#                             transforms.ToTensor(),
#                             normalize,
#         ]))
#         if valid: #Validation Split
#             val_transform = train_transform
#     else: #Non-contrastive data transforms == False / main_ce.py or main_linear.py
#         train_transform = transforms.Compose([
#             transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#         val_transform = transforms.Compose([
#             transforms.Resize([opt.size, opt.size]),
#             transforms.ToTensor(),
#             normalize,
#         ]) 
    
#     #Contrusciton of dataset
#     if opt.dataset == 'cifar10':
#         train_dataset = datasets.CIFAR10(root=opt.data_folder,
#                                          transform=train_transform,
#                                          download=True)
#         val_dataset = datasets.CIFAR10(root=opt.data_folder,
#                                         train=False,
#                                         transform=val_transform)
#     elif opt.dataset == 'cifar100':
#         train_dataset = datasets.CIFAR100(root=opt.data_folder,
#                                           transform=train_transform,
#                                           download=True)
#         val_dataset = datasets.CIFAR100(root=opt.data_folder,
#                                          train=False,
#                                          transform=val_transform)
#     elif opt.dataset == 'path':
#         if opt.root_path is not None:
#             train_dataset = CustomDatasetFromCSV(opt.root_path, tf_image=train_transform, csv_name=opt.train_files, task=None, as_rgb=True)
#             val_dataset = CustomDatasetFromCSV(opt.root_path, tf_image=val_transform, csv_name=opt.val_files, task=None, as_rgb=True)
#         else: 
#             #Load from folders
#             train_dataset = datasets.ImageFolder(root=opt.data_folder + "/train/",
#                                                 transform=train_transform)
#             val_dataset = datasets.ImageFolder(root=opt.data_folder + "/val/",
#                                                 transform=val_transform)
#     else:
#         raise ValueError(f'Dataset', opt.dataset, 'not supported')
    
    
#     train_loader = DataLoader(train_dataset,
#                                 batch_size=opt.batch_size,
#                                 num_workers=opt.num_workers,
#                                 pin_memory=True,
#                                 shuffle=True)
    
#     val_loader = DataLoader(val_dataset,
#                                 batch_size=opt.batch_size,
#                                 num_workers=opt.num_workers,    
#                                 pin_memory=True,
#                                 shuffle=False)
                               
#     # Compute mean and STD used above (use test transform without normalization)
#     # Code from https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6

#     return train_loader, val_loader

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


# def set_model(opt):

#     if "vit" in opt.model: #If model is ViT
#         model = vits.SupCEViT(name=opt.model, num_classes=opt.n_cls)
#         # classifier = vits.LinearClassifierViT(name=opt.model, num_classes=opt.n_cls)
#     else:

#         model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    
    
#     criterion = torch.nn.CrossEntropyLoss()

#     if torch.cuda.is_available():
#         if torch.cuda.device_count() > 1:
#             model = torch.nn.DataParallel(model)
#         model = model.cuda()
#         criterion = criterion.cuda()
#         cudnn.benchmark = True

#     return model, criterion

def set_model(opt:str):

    print('\n[INFO] Setting model and criterion with linear classifier...')

    # Set model
    if opt.model == "vit_small" or opt.model == "dino_vit_small_p_16" or opt.model == 'dino_vit_small_p_8': #não ta funcionando ainda
        model = vits.SupCEViT(name=opt.model, feat_dim=384)
    elif opt.model == "vit_base" or opt.model == "dino_vit_base_p_16" or opt.model == 'dino_vit_base_p_8':
        model = vits.SupCEViT(name=opt.model, feat_dim=768)
    elif opt.model == "resnet50": #If model is resnet
        model = SupCEResNet(name=opt.model)
    else:
        raise ValueError('Model not supported: {}'.format(opt.model))
    
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""

    fix_random_seeds(seed=opt.seed)

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

    
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 or (idx + 1) == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            losses.update(loss.item(), bsz)
            if opt.n_cls > 4:
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top5.update(acc5[0], bsz)
            else:
                acc1 = accuracy(output, labels, topk=(1,))[0]
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) == len(val_loader):
                print('Valid: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print('\t[INFO] * Average validation: Acc@1 {top1.avg:.3f} | Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return losses.avg, top1.avg


def test(test_loader, model, criterion, opt):
    """testing"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            losses.update(loss.item(), bsz)
            if opt.n_cls > 4:
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top5.update(acc5[0], bsz)
            else:
                acc1 = accuracy(output, labels, topk=(1,))[0]
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print('\t[INFO] * Average testing: Acc@1 {top1.avg:.3f} | Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return losses.avg, top1.avg


def cache_outputs(val_loader, model, opt):
    # save model outputs for analysis and bootstrapping
    model.eval()
    # caches for outputs

    # Define as dimensões de embeddings para diferentes arquiteturas
    embedding_dim = {  
        'resnet50': 2048,
        'vit_small': 384, 'vit_base': 768, 
        'dino_vit_small_p_16': 384, 'dino_vit_small_p_8': 384,
        'dino_vit_base_p_16': 768, 'dino_vit_base_p_8': 768
    }.get(opt.model)
    
    # embedding_dim = embedding_dim.get(opt.model)  

    embeds = torch.empty((0, int(embedding_dim)))
    preds = torch.empty((0, opt.n_cls))
    labels = torch.empty((0,))

    with torch.no_grad():
        for b_images, b_labels in val_loader:
            b_images = b_images.float().cuda()
            b_labels = b_labels.cuda()
            # forward
            b_embeds = model.encoder(b_images)
            b_preds = model.fc(b_embeds)
            # (b_embeds)
            # cache
            embeds = torch.vstack((embeds, b_embeds.cpu()))
            preds = torch.vstack((preds, b_preds.cpu()))
            labels = torch.hstack((labels, b_labels.cpu()))
    # save caches
    torch.save(embeds, os.path.join(opt.save_folder, "embeds.pth"))
    torch.save(preds, os.path.join(opt.save_folder, "preds.pth"))
    torch.save(labels, os.path.join(opt.save_folder, "labels.pth"))

    return


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt, contrast_trans=False, for_test=True)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc = validate(val_loader, model, criterion, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)   

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        # testing
        if test_loader is not None and epoch == opt.epochs:
            loss, val_acc = test(test_loader, model, criterion, opt)
            # logger.log_value('test_loss', loss, epoch)
            # logger.log_value('test_acc', val_acc, epoch)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    cache_outputs(test_loader, model, opt)


    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
