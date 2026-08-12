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
import networks.resnet_big as resnets
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


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = util.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 1000):
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


class ModelConcatenate(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        features = self.classifier(features)
        return features


def set_model(opt):

    # Set model
    if opt.arch == "vit_small" or opt.arch == "dino_vit_small_p_16":
        encoder = vits.SupConViT(name=opt.arch, feat_dim=384)
        # classifier = vits.LinearClassifierViT(name=opt.model, num_classes=opt.n_cls)
    elif opt.arch == "vit_base" or opt.arch == "dino_vit_base_p_16":
        encoder = vits.SupConViT(name=opt.arch, feat_dim=768)
        # classifier = vits.LinearClassifierViT(name=opt.model, num_classes=opt.n_cls)
    elif opt.arch == "resnet50": 
        encoder = resnets.SupConResNet(name=opt.arch)
        # classifier = resnets.LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    else:
        raise ValueError('Model not supported: {}'.format(opt.arch))
    

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            encoder = torch.nn.DataParallel(encoder)
        encoder = encoder.cuda()
        # classifier = classifier.cuda()
    
    state_dict_model = torch.load(opt.pretrained_weights, map_location="cpu", weights_only=False)["model"]
    # state_dict_cls = torch.load(opt.ckpt, map_location="cpu", weights_only=False)["classifier"]
    encoder.load_state_dict(state_dict_model, strict=False)
    # classifier.load_state_dict(state_dict_cls, strict=True)
    
    # model = ModelConcatenate(encoder.encoder, classifier.fc)
    
    return encoder.encoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Retrieval on revisited Paris and Oxford')
    parser.add_argument('--data_path', default='Datasets', type=str)
    parser.add_argument('--dataset', default='rparis6k', type=str, choices=['roxford5k', 'rparis6k'])
    parser.add_argument('--multiscale', default=False, type=util.bool_flag)
    parser.add_argument('--imsize', default=512, type=int, help='Image size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=False, type=util.bool_flag)
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    # parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # parser.add_argument("--checkpoint_key", default="teacher", type=str,
    #     help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    # parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    #     distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    # parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    #parser.add_argument('--n_cls', type=int, default=10, help='number of classes') #For CIFAR10
    parser.add_argument('--seed', default=31, type=int,    help='Random seed')
    args = parser.parse_args()

    util.fix_random_seeds(args.seed)
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

    


    model = set_model(args)
    model.cuda()
    model.eval()


    # ============ extract features ... ============
    train_features = extract_features(model, data_loader_train, args.use_cuda, multiscale=args.multiscale)
    query_features = extract_features(model, data_loader_query, args.use_cuda, multiscale=args.multiscale)

    # Normalize features
    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    query_features = nn.functional.normalize(query_features, dim=1, p=2)

    # ============ compute similarity and ranks ... ============
    sim = torch.mm(train_features, query_features.T)
    ranks = torch.argsort(-sim, dim=0).cpu().numpy()

    gnd = dataset_train.cfg['gnd']
    ks = [1, 5, 10]
    gnd_t = []
    for i in range(len(gnd)):
       g = {}
       g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
       g['junk'] = np.concatenate([gnd[i]['junk']])
       gnd_t.append(g)
    mapM, apsM, mprM, prsM = util.compute_map(ranks, gnd_t, ks)

    gnd_t = []
    for i in range(len(gnd)):
       g = {}
       g['ok'] = np.concatenate([gnd[i]['hard']])
       g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
       gnd_t.append(g)

    mapH, apsH, mprH, prsH = util.compute_map(ranks, gnd_t, ks)

    print(f'>> {args.dataset}: mAP M: {np.around(mapM*100, decimals=2)}, H: {np.around(mapH*100, decimals=2)}')
    print(f'>> {args.dataset}: mP@k{np.array(ks)} M: {np.around(mprM*100, decimals=2)}, H: {np.around(mprH*100, decimals=2)}')
