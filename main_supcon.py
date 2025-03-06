from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

#Metrics
from contrast_acc import contrastive_acc, test_contrastive_acc, test_contrastive_acc_knn
#Utils
from util import AverageMeter, adjust_learning_rate, warmup_learning_rate, set_optimizer, save_model, TwoCropTransform, CustomDatasetFromCSV
#Networks
from networks.pretrained_models import load_pretrained_model
#Losses
from losses import SupConLoss, MultiviewSINCERELoss, MultiviewEpsSupInfoNCELoss, InfoNCELoss
#Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


__all_models = ['resnet50', 
                'vit_small', 'vit_base', 
                'dino_vit_small_p_16', 'dino_vit_small_p_8', 'dino_vit_base_p_16', 'dino_vit_base_p_8']


def parse_option():

    parser = argparse.ArgumentParser('Arguments for training...')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30, 50, 70',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 
                                                                          'vit_small', 'vit_base',
                                                                          'dino_vit_small_p_16', 'dino_vit_small_p_8', 
                                                                          'dino_vit_base_p_16', 'dino_vit_base_p_8'], 
                        help='Choose your backbone')
    #parser.add_argument('--n_cls', type=int, default=None, help='Number of classes for your dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet100', 'imagenet', 'cifar2', 'aircraft', 'cars', 'path'], 
                        help='Dataset')
    parser.add_argument('--valid_split', type=float, default=0,
                        help="proportion of train data to use for validation set")
    parser.add_argument('--mean', type=str,
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str,
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str,
                        default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32,
                        help='size of images after resizing')

    # Contrastive method
    parser.add_argument('--method', type=str, default='SINCERE',
                        choices=['SINCERE', 'SupCon', 'EpsSupInfoNCE', 'SimCLR', 'InfoNCE'],
                        help='Choose your contrastive method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='Temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    #Experimental Dataset Settings
    parser.add_argument('--root_path', type=str, default='', help='root path to dataset')
    parser.add_argument('--train_files', type=str, default='Datasets/train_folds.csv', help='csv file for training')
    parser.add_argument('--num_folds', type=int, default=5, help='number of folds for cross-validation based on csv file')
    # parser.add_argument('--val_files', type=str, default='Datasets/KFolds/SKF_VAL_Fold_1.csv', help='csv file for validation')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        #assert opt.data_folder is not None
        assert opt.mean is not None
        assert opt.std is not None
        #assert opt.n_cls is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

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

    # add time to model name
    opt.model_name += "_" + time.strftime("%Y_%m_%d-%H_%M_%S")

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    # Priting arguments for logging
    print("\n[INFO] Printing arguments for pre-training stage...")
    print(opt)
    
    print("\n[INFO] Training with gpu: {}".format(torch.cuda.get_device_name()))

    return opt


def set_loader(opt, contrast_trans=True, fold=None):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if contrast_trans:
        #Augmentations for contrastive learning based on: https://github.com/HobbitLong/SupContrast
        train_transform = TwoCropTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    #Create datasets with csv files and folds for training and validation
    train_dataset = CustomDatasetFromCSV(
        path_root=opt.root_path,
        tf_image=train_transform,
        csv_name=opt.train_files,
        val_fold=fold,
        train=True
    )

    val_dataset = CustomDatasetFromCSV(
        path_root=opt.root_path,
        tf_image=train_transform,
        csv_name=opt.train_files,
        val_fold=fold,
        train=False
    )

    
    #Create dataloaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=opt.batch_size, 
                              shuffle=True, 
                              num_workers=opt.num_workers,
                              pin_memory=True)
    
  
    valid_loader = DataLoader(val_dataset, 
                                batch_size=opt.batch_size, 
                                shuffle=False, 
                                num_workers=opt.num_workers,
                                pin_memory=True)
    
    return train_loader, valid_loader



def set_model(opt):

    print('\n[INFO] Setting model and criterion...')

    # Set the model
    model = None

    if opt.model in __all_models:
        model = load_pretrained_model(opt.model)
    
    #Set criterion 
    if opt.method == 'SINCERE':
        # original implementation does not set base_temperature, but setting here to make
        # hyperparameters comparable between implementations
        criterion = MultiviewSINCERELoss(temperature=opt.temp)
    elif opt.method == 'SupCon':
        criterion = SupConLoss(temperature=opt.temp)
    elif opt.method == 'EpsSupInfoNCE':
        criterion = MultiviewEpsSupInfoNCELoss(temperature=opt.temp)
    elif opt.method == 'SimCLR':
        criterion = SupConLoss(temperature=opt.temp)
    elif opt.method == 'InfoNCE':
        criterion = InfoNCELoss(temperature=opt.temp) 
    else:
        raise ValueError('[INFO] Contrastive method not supported on setting model: {}'.
                         format(opt.method))

    if torch.cuda.is_available():
        if "device" not in opt:
            model = model.cuda()
        else:
            model = model.to(opt.device)
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.parallel.DistributedDataParallel(model.encoder)
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, logger):
    """one epoch training"""

    #Set model to train mode
    model.train()

    #Set average meters
    av_batch_time = AverageMeter()
    av_data_time = AverageMeter()
    av_acc = AverageMeter()
    av_losses = AverageMeter()

    end = time.time()

    # #Change reshuffle split of data across GPUs
    if "device" in opt:
        train_loader.sampler.set_epoch(epoch)
    for idx, (image_aug_tuple, labels) in enumerate(train_loader):
        av_data_time.update(time.time() - end)

        images = torch.cat([image_aug_tuple[0], image_aug_tuple[1]], dim=0)
        if torch.cuda.is_available():
            if "device" not in opt:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            else:
                images = images.to(opt.device, non_blocking=True)
                labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # forward
        with torch.set_grad_enabled(True):
            flat_embeds = model(images)
        # reshape from (2B, D) to (B, 2, D)
        embeds = torch.cat([aug.unsqueeze(1) for aug in torch.split(flat_embeds, [bsz, bsz], dim=0)], dim=1)
        # compute losses
        # loss is averaged across GPU-specific batches if using multiple GPUs, as in SupCon
        # see MoCo v3 for full batch size parallelization with torch's all_gather
        if opt.method == 'SINCERE' or opt.method == 'EpsSupInfoNCE' or opt.method == 'SupCon': #Supervised contrastive learning methods
            loss = criterion(embeds, labels)
        elif opt.method == 'SimCLR' or opt.method == 'InfoNCE': #Self-supervised contrastive learning methods
            loss = criterion(embeds)
        else:
            raise ValueError('[INFO] Contrastive method not supported in training phase: {}'.
                             format(opt.method))

        av_losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute accuracy
        with torch.no_grad():
            acc = contrastive_acc(embeds, labels)
            av_acc.update(acc.item(), bsz)

        # measure elapsed time
        av_batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('[Train] Epoch: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=av_batch_time,
                    data_time=av_data_time, loss=av_losses))
            sys.stdout.flush()

    # tensorboard logger to save accuracy and loss
    if "device" not in opt or opt.device == 0:
        log_folder = "train/"
        logger.add_scalar(f"{log_folder}{str(opt.method)} loss", av_losses.avg, epoch)
        logger.add_scalar(f"{log_folder}Accuracy {str(opt.method)}: ", av_acc.avg, epoch)
        
    # log values independent of forward passes
    logger.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

    return av_acc.avg, av_losses.avg


def valid(train_loader, valid_loader, model, criterion, epoch, opt, logger):
    """validation"""

    embedding_dim = {  
        'resnet50': 128,
        'vit_small': 384, 'vit_base': 768, 
        'dino_vit_small_p_16': 384, 'dino_vit_small_p_8': 384,
        'dino_vit_base_p_16': 768, 'dino_vit_base_p_8': 768
    }.get(opt.model)

    #Caches to training/valid/test
    train_embeds = torch.empty((0, embedding_dim))
    train_labels = torch.empty((0,))

    test_embeds = torch.empty((0, embedding_dim))
    test_labels = torch.empty((0,))

    for i, loader in enumerate([train_loader, valid_loader]):
        is_train = i == 0
        model.eval()

        av_batch_time = AverageMeter()
        av_data_time = AverageMeter()
        av_losses = AverageMeter()
        av_acc_top_1 = AverageMeter()
        av_acc_top_5 = AverageMeter()

        end = time.time()
        # # # change reshuffle split of data across GPUs
        if "device" in opt:
            loader.sampler.set_epoch(epoch)
        for idx, (image_aug_tuple, labels) in enumerate(loader):
            av_data_time.update(time.time() - end)

            images = torch.cat([image_aug_tuple[0], image_aug_tuple[1]], dim=0)
            if torch.cuda.is_available():
                if "device" not in opt:
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                else:
                    images = images.to(opt.device, non_blocking=True)
                    labels = labels.to(opt.device, non_blocking=True)
            bsz = labels.shape[0]

            # forward
            with torch.no_grad():
                flat_embeds = model(images)
            # reshape from (2B, D) to (B, 2, D)
            embeds = torch.cat(
                [aug.unsqueeze(1) for aug in torch.split(flat_embeds, [bsz, bsz], dim=0)], dim=1)
            # cache train outputs
            if is_train:
                train_embeds = torch.vstack((train_embeds, embeds[:, 0].cpu()))
                train_labels = torch.hstack((train_labels, labels.cpu()))
            else:
                # cache valid/test outputs
                test_embeds = torch.vstack((test_embeds, embeds[:, 0].cpu()))
                test_labels = torch.hstack((test_labels, labels.cpu()))
                # compute validation accuracy
                av_acc_top_1.update(test_contrastive_acc(
                    train_embeds.cuda(), embeds[:, 0].cuda(),
                    train_labels.cuda(), labels.cuda()).item(), bsz)
                av_acc_top_5.update(test_contrastive_acc_knn(
                    train_embeds.cuda(), embeds[:, 0].cuda(),
                    train_labels.cuda(), labels.cuda(), 5).item(), bsz)

            # compute losses (note there's no class balancing sampler for test)
            # loss is averaged across GPU-specific batches if using multiple GPUs, as in SupCon
            # see MoCo v3 for full batch size parallelization with torch's all_gather 
            if opt.method == 'SINCERE' or opt.method == 'EpsSupInfoNCE' or opt.method == 'SupCon': #Supervised contrastive learning methods
                loss = criterion(embeds, labels)
            elif opt.method == 'SimCLR' or opt.method == 'InfoNCE': #Self-supervised contrastive learning methods
                loss = criterion(embeds)
            else:
                raise ValueError('[INFO] Contrastive method not supported in training [valid] phase: {}'.
                             format(opt.method))
            
            # update averages
            av_losses.update(loss.item(), bsz)

            # measure elapsed time
            av_batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('[Validation] Epoch: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                        epoch, idx + 1, len(train_loader), batch_time=av_batch_time,
                        data_time=av_data_time, loss=av_losses))
                sys.stdout.flush()

    if "device" not in opt or opt.device == 0:
        # tensorboard logger
        log_folder = "valid/"
        logger.add_scalar(f"{log_folder}{str(opt.method)}Valid Loss", av_losses.avg, epoch)
        logger.add_scalar(f"{log_folder}Valid Top 1 Accuracy", av_acc_top_1.avg, epoch)
        logger.add_scalar(f"{log_folder}Valid Top 5 Accuracy", av_acc_top_5.avg, epoch)
        # print output
        print(f"Valid/test {str(opt.method)} Loss: {av_losses.avg}")
        print(f"Valid/test Top 1 Accuracy: {av_acc_top_1.avg}")
        print(f"Valid/test Top 5 Accuracy: {av_acc_top_5.avg}")
        # save caches
        torch.save(train_embeds, os.path.join(opt.save_folder, "train_embeds.pth"))
        torch.save(train_labels, os.path.join(opt.save_folder, "train_labels.pth"))
        torch.save(test_embeds, os.path.join(opt.save_folder, "valid_embeds.pth"))
        torch.save(test_labels, os.path.join(opt.save_folder, "valid_labels.pth"))

    return av_losses.avg, av_acc_top_1.avg, av_acc_top_5.avg


#If I can test the model on the test set
# def test(model, criterion, opt, fold):
#     train_loader, valid_loader = set_loader(opt, contrast_trans=True, fold=fold)
#     valid(train_loader, valid_loader, model, criterion, 0, opt, logger=None)
#     #valid(train_loader, test_loader, model, criterion, 0, opt, logger=None)


def main(opt):
    accs_train = [] 
    losses_train = []
    std_devs_acc_train = []   
    std_devs_loss_train = []
    
    accs_val_ = []
    losses_val = []
    std_devs_val = []
    std_devs_loss_val = []
    accs5_val = []
    std_devs5_val = []

    for fold in range(opt.num_folds):
        print(f"\n[INFO] Training model on fold {fold}...")

        train_loader, valid_loader = set_loader(opt, contrast_trans=True, fold=fold)
        
        print(f"Size from train_loader: {len(train_loader)} batches")
        print(f"Size from val_loader: {len(valid_loader)} batches")

        model, criterion = set_model(opt)

        optimizer = set_optimizer(opt, model)

        #Tensorboard logger 
        logger = None
        if "device" not in opt or opt.device == 0:
            logger = SummaryWriter(log_dir=os.path.join(opt.tb_folder, f"fold_{fold}"))

      
        fold_acc_train = AverageMeter()
        fold_losses_train = AverageMeter()  
        fold_losses_val = []  
        fold_acc_val = []  
        fold_acc5_val = []

        # Loop de treinamento
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            time1 = time.time()
            av_train_acc, av_train_loss,  = train(train_loader, model, criterion, optimizer, epoch, opt, logger)
            time2 = time.time()

            print(f"\tloss train: {av_train_loss}")
            print(f"\tacc train: {av_train_acc}")

            fold_losses_train.update(av_train_loss)
            fold_acc_train.update(av_train_acc)

            print(f'Epoch {epoch}, Fold {fold}, Total Time: {time2 - time1:.2f}s')

            av_val_loss, av_val_acc, av_val_acc5 = valid(train_loader, valid_loader, model, criterion, epoch, opt, logger) 

            fold_losses_val.append(av_val_loss)
            fold_acc_val.append(av_val_acc)
            fold_acc5_val.append(av_val_acc5)

            # Salvar checkpoints periódicos
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(opt.save_folder, f'ckpt_fold_{fold}_epoch_{epoch}.pth')
                save_model(model, optimizer, opt, epoch, save_file)

        accs_train.append(fold_acc_train.avg)
        losses_train.append(fold_losses_train.avg)

        # Calcular desvio padrão da acurácia e loss de treinamento
        std_devs_acc_train.append(np.std(fold_acc_train.history)) 
        std_devs_loss_train.append(np.std(fold_losses_train.history))  

        accs_val_.append(np.mean(fold_acc_val))  
        losses_val.append(np.mean(fold_losses_val))
        std_devs_val.append(np.std(fold_acc_val))  
        std_devs_loss_val.append(np.std(fold_losses_val))  
        accs5_val.append(np.mean(fold_acc5_val))  
        std_devs5_val.append(np.std(fold_acc5_val))  

        # Salvar o último modelo deste fold
        save_file = os.path.join(opt.save_folder, f'last_fold_{fold}.pth')
        save_model(model, optimizer, opt, opt.epochs, save_file)

 
    mean_accuracy_train = np.mean(accs_train)
    std_dev_accuracy_train = np.std(accs_train)
    mean_loss_train = np.mean(losses_train)
    std_dev_loss_train = np.std(losses_train)

    mean_accuracy_val = np.mean(accs_val_)
    std_dev_accuracy_val = np.std(accs_val_)
    mean_loss_val = np.mean(losses_val)
    std_dev_loss_val = np.std(losses_val)
    
    mean_accuracy5_val = np.mean(accs5_val)
    std_dev_accuracy5_val = np.std(accs5_val)

  
    print(f"\n[INFO] Training Mean Accuracy across folds: {mean_accuracy_train:.4f}")
    print(f"[INFO] Standard Deviation of Training Accuracy across folds: {std_dev_accuracy_train:.4f}")

    print(f"[INFO] Training Mean Loss across folds: {mean_loss_train:.4f}")
    print(f"[INFO] Standard Deviation of Training Loss across folds: {std_dev_loss_train:.4f}\n")

    print(f"[INFO] Validation Mean Accuracy across folds: {mean_accuracy_val:.4f}")
    print(f"[INFO] Standard Deviation of Validation Accuracy across folds: {std_dev_accuracy_val:.4f}")

    print(f"[INFO] Validation Mean Loss across folds: {mean_loss_val:.4f}")
    print(f"[INFO] Standard Deviation of Validation Loss across folds: {std_dev_loss_val:.4f}")
    
    print(f"[INFO] Validation Mean Top-5 Accuracy across folds: {mean_accuracy5_val:.4f}")
    print(f"[INFO] Standard Deviation of Validation Top-5 Accuracy across folds: {std_dev_accuracy5_val:.4f}")


def launch_parallel(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # need to use gloo instead of nccl for Windows, but nccl faster on Linux
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    opt = parse_option()
    # modify options for parallel processing
    opt.device = rank  # device not in opt if not using parallel processing
    opt.batch_size = opt.batch_size // world_size
    main(opt)


if __name__ == '__main__':
    parallel = False
    if not parallel:
        main(parse_option())
    else:
        world_size = 2
        torch.multiprocessing.spawn(launch_parallel, (world_size,), world_size)