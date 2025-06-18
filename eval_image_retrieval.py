# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import pickle
import argparse

import torch
from torch import nn
# import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from PIL import Image, ImageFile
import numpy as np

import util
import networks.vit as vits
from networks.resnet_big import SupCEResNet
# import networks.resnet_big as SupCEResNet
# from util import extract_features



class OxfordParisDataset(torch.utils.data.Dataset):
    def __init__(self, dir_main, dataset, split, transform=None, imsize=None):
        if dataset not in ['roxford5k', 'rparis6k']:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')
        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])
        cfg['im_fname'] = config_imname
        cfg['qim_fname'] = config_qimname
        cfg['dataset'] = dataset
        self.cfg = cfg

        self.samples = cfg["qimlist"] if split == "query" else cfg["imlist"]
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = os.path.join(self.cfg["dir_images"], self.samples[index] + ".jpg")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.imsize is not None:
            img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
        if self.transform is not None:
            img = self.transform(img)
        return img, index


def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])


# @torch.no_grad()
# def extract_features(model, data_loader, use_cuda=True, multiscale=False):
#     metric_logger = util.MetricLogger(delimiter="  ")
#     features = None
#     for samples, index in metric_logger.log_every(data_loader, 10):
#         samples = samples.cuda(non_blocking=True)
#         index = index.cuda(non_blocking=True)
#         if multiscale:
#             feats = util.multi_scale(samples, model)
#         else:
#             feats = model(samples).clone()

#         # init storage feature matrix
#         # if dist.get_rank() == 0 and features is None:
#         #     features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
#         #     if use_cuda:
#         #         features = features.cuda(non_blocking=True)
#         #     print(f"Storing features into tensor of shape {features.shape}")


#         if features is None:
#             features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
#         if use_cuda:
#             features = features.cuda(non_blocking=True)
#         print(f"Storing features into tensor of shape {features.shape}")

#         # get indexes from all processes
#         y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
#         y_l = list(y_all.unbind(0))
#         y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
#         y_all_reduce.wait()
#         index_all = torch.cat(y_l)

#         # share features between processes
#         feats_all = torch.empty(
#             dist.get_world_size(),
#             feats.size(0),
#             feats.size(1),
#             dtype=feats.dtype,
#             device=feats.device,
#         )
#         output_l = list(feats_all.unbind(0))
#         output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
#         output_all_reduce.wait()

#         # update storage feature matrix
#         if dist.get_rank() == 0:
#             if use_cuda:
#                 features.index_copy_(0, index_all, torch.cat(output_l))
#             else:
#                 features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
#     return features



@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = util.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = util.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # update storage feature matrix
        if use_cuda:
            features.index_copy_(0, index, feats)
        else:
            features.index_copy_(0, index.cpu(), feats.cpu())
    return features


def set_model(opt:str):

    print('\n[INFO] Setting model and criterion with linear classifier...')

    # Set model
    if opt.arch == "vit_small" or opt.arch == "dino_vit_small_p_16" or opt.arch == 'dino_vit_small_p_8': #não ta funcionando ainda
        model = vits.SupCEViT(name=opt.arch, num_classes=opt.n_cls)
    elif opt.arch == "vit_base" or opt.arch == "dino_vit_base_p_16" or opt.arch == 'dino_vit_base_p_8':
        model = vits.SupCEViT(name=opt.arch, num_classes=opt.n_cls)
    elif opt.arch == "resnet50": #If model is resnet
        model = SupCEResNet(name=opt.arch, num_classes=opt.n_cls)
    else:
        raise ValueError('Model not supported: {}'.format(opt.arch))
    

    ckpt = torch.load(opt.pretrained_weights, map_location='cpu', weights_only=False)["model"]
    # state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True


    model.load_state_dict(ckpt, strict=False)
    # model.load_state_dict(state_dict)
    # model.eval()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Retrieval on revisited Paris and Oxford')
    parser.add_argument('--data_path', default='Datasets', type=str)
    parser.add_argument('--dataset', default='roxford5k', type=str, choices=['roxford5k', 'rparis6k'])
    parser.add_argument('--multiscale', default=False, type=util.bool_flag)
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=util.bool_flag)
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    # parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # parser.add_argument("--checkpoint_key", default="teacher", type=str,
    #     help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    # parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    #     distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--n_cls', type=int, default=10, help='number of classes') #For CIFAR10

    args = parser.parse_args()

    # utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = OxfordParisDataset(args.data_path, args.dataset, split="train", transform=transform, imsize=args.imsize)
    dataset_query = OxfordParisDataset(args.data_path, args.dataset, split="query", transform=transform, imsize=args.imsize)

    # sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        # sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"train: {len(dataset_train)} imgs / query: {len(dataset_query)} imgs")

    # # ============ building network ... ============
    # if "vit" in args.arch:
    #     model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    #     print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # elif "xcit" in args.arch:
    #     model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    # elif args.arch in torchvision_models.__dict__.keys():
    #     model = torchvision_models.__dict__[args.arch](num_classes=0)
    # else:
    #     print(f"Architecture {args.arch} non supported")
    #     sys.exit(1)
    # if args.use_cuda:
    #     model.cuda()
    # model.eval()

    # # load pretrained weights
    # if os.path.isfile(args.pretrained_weights):
    #     state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    #     if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
    #         print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
    #         state_dict = state_dict[args.checkpoint_key]
    #     # remove `module.` prefix
    #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #     # remove `backbone.` prefix induced by multicrop wrapper
    #     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    #     msg = model.load_state_dict(state_dict, strict=False)
    #     print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    # elif args.arch == "vit_small" and args.patch_size == 16:
    #     print("Since no pretrained weights have been provided, we load pretrained DINO weights on Google Landmark v2.")
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth"))
    # else:
    #     print("Warning: We use random weights.")


    model = set_model(args)
    model.cuda()
    model.eval()


    # ############################################################################
    # # Step 1: extract features
    # train_features = extract_features(model, data_loader_train, args.use_cuda, multiscale=args.multiscale)
    # query_features = extract_features(model, data_loader_query, args.use_cuda, multiscale=args.multiscale)

    # if util.get_rank() == 0:  # only rank 0 will work from now on
    #     # normalize features
    #     train_features = nn.functional.normalize(train_features, dim=1, p=2)
    #     query_features = nn.functional.normalize(query_features, dim=1, p=2)

    #     ############################################################################
    #     # Step 2: similarity
    #     sim = torch.mm(train_features, query_features.T)
    #     ranks = torch.argsort(-sim, dim=0).cpu().numpy()

    #     ############################################################################
    #     # Step 3: evaluate
    #     gnd = dataset_train.cfg['gnd']
    #     # evaluate ranks
    #     ks = [1, 5, 10]
    #     # search for easy & hard
    #     gnd_t = []
    #     for i in range(len(gnd)):
    #         g = {}
    #         g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
    #         g['junk'] = np.concatenate([gnd[i]['junk']])
    #         gnd_t.append(g)
    #     mapM, apsM, mprM, prsM = util.compute_map(ranks, gnd_t, ks)
    #     # search for hard
    #     gnd_t = []
    #     for i in range(len(gnd)):
    #         g = {}
    #         g['ok'] = np.concatenate([gnd[i]['hard']])
    #         g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
    #         gnd_t.append(g)
    #     mapH, apsH, mprH, prsH = util.compute_map(ranks, gnd_t, ks)
    #     print('>> {}: mAP M: {}, H: {}'.format(args.dataset, np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
    #     print('>> {}: mP@k{} M: {}, H: {}'.format(args.dataset, np.array(ks), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
    # # dist.barrier()

    train_features = extract_features(model, data_loader_train, args.use_cuda, multiscale=args.multiscale)
    query_features = extract_features(model, data_loader_query, args.use_cuda, multiscale=args.multiscale)

    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    query_features = nn.functional.normalize(query_features, dim=1, p=2)

    sim = torch.mm(train_features, query_features.T)
    ranks = torch.argsort(-sim, dim=0).cpu().numpy()

    gnd = dataset_train.cfg['gnd']
    ks = [1, 5, 10]
    gnd_t = []
    for i in range(len(gnd)):
        g = {'ok': np.concatenate([gnd[i]['easy'], gnd[i]['hard']]), 'junk': np.concatenate([gnd[i]['junk']])}
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = util.compute_map(ranks, gnd_t, ks)

    gnd_t = []
    for i in range(len(gnd)):
        g = {'ok': np.concatenate([gnd[i]['hard']]), 'junk': np.concatenate([gnd[i]['junk'], gnd[i]['easy']])}
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = util.compute_map(ranks, gnd_t, ks)

    print(f'>> {args.dataset}: mAP M: {np.around(mapM*100, decimals=2)}, H: {np.around(mapH*100, decimals=2)}')
    print(f'>> {args.dataset}: mP@k{np.array(ks)} M: {np.around(mprM*100, decimals=2)}, H: {np.around(mprH*100, decimals=2)}')