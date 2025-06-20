import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import random

import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import numpy as np

import networks.resnet_big as resnets
import networks.vit as vits

from util import SmoothedValue, MetricLogger

# ------------------------- Argumentos -------------------------
def parse_option():
    parser = argparse.ArgumentParser('Arguments for Multi-view Retrieval')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model', type=str, default='vit_small', choices=[
        'resnet50', 'vit_small', 'vit_base',
        'dino_vit_small_p_16', 'dino_vit_small_p_8',
        'dino_vit_base_p_16', 'dino_vit_base_p_8'])
    parser.add_argument('--n_cls', type=int, default=10)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--data_folder', type=str, default="ModelNet10")
    parser.add_argument('--root_path', type=str, default='ModelNet10')
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--n_view', type=int, default=None, help='Select specific view of your dataset')
    parser.add_argument('--output_folder', type=str, default="shape_retrieval_images", help='Folder to save retrieval plots')
    parser.add_argument('--query_label', type=str, default=None, help='Run retrieval only for this class label')

    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--seed', default=31, type=int)
        
    opt = parser.parse_args()

    return opt


# ------------------------- Dataset Multi-view -------------------------
class MultiViewDatasetFromCSV(Dataset):
    def __init__(self, path_root, tf_image, csv_name):
        df = pd.read_csv(csv_name)
        self.obj_to_views = {}
        self.obj_labels = {}

        for _, row in df.iterrows():
            path = os.path.join(path_root, row['path'])
            obj_id = row['obj_id']
            label = row['label']
            if obj_id not in self.obj_to_views:
                self.obj_to_views[obj_id] = []
                self.obj_labels[obj_id] = label
            self.obj_to_views[obj_id].append(path)

        self.obj_ids = list(self.obj_to_views.keys())
        self.transform = tf_image

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, idx):
        obj_id = self.obj_ids[idx]
        view_paths = self.obj_to_views[obj_id]
        label = self.obj_labels[obj_id]
        imgs = [self.transform(Image.open(p).convert("RGB")) for p in view_paths]
        return torch.stack(imgs), label, obj_id, view_paths


# ------------------------- Loader -------------------------
def set_loader(opt):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize([opt.size, opt.size]),
        transforms.ToTensor(),
        normalize
    ])

    # dataset_class = MultiViewDatasetFromCSV(opt.path_root, )

    train_dataset = MultiViewDatasetFromCSV(opt.root_path, transform, opt.train_file)
    test_dataset = MultiViewDatasetFromCSV(opt.root_path, transform, opt.test_file)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=opt.num_workers, pin_memory=True)

    return train_loader, test_loader


# ------------------------- Modelo -------------------------
class ModelConcatenate(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        return features


def set_model(opt):

    # Set model
    if opt.model == "vit_small" or opt.model == "dino_vit_small_p_16":
        encoder = vits.SupConViT(name=opt.model, feat_dim=384)
        # classifier = vits.LinearClassifierViT(name=opt.model, num_classes=opt.n_cls)
    elif opt.model == "vit_base" or opt.model == "dino_vit_base_p_16":
        encoder = vits.SupConViT(name=opt.model, feat_dim=768)
        # classifier = vits.LinearClassifierViT(name=opt.model, num_classes=opt.n_cls)
    elif opt.model == "resnet50": 
        encoder = resnets.SupConResNet(name=opt.model)
        # classifier = resnets.LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    else:
        raise ValueError('Model not supported: {}'.format(opt.model))
    

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            encoder = torch.nn.DataParallel(encoder)
        encoder = encoder.cuda()
        # classifier = classifier.cuda()
    
    state_dict_model = torch.load(opt.ckpt, map_location="cpu", weights_only=False)["model"]
    # state_dict_cls = torch.load(opt.ckpt, map_location="cpu", weights_only=False)["classifier"]
    encoder.load_state_dict(state_dict_model, strict=True)
    # classifier.load_state_dict(state_dict_cls, strict=True)
    
    # model = ModelConcatenate(encoder.encoder, classifier.fc)
    
    return encoder



@torch.no_grad
def extract_embeddings(dataloader, model, opt):

    print("extracting embeddings... ")
    
    embeddings, labels, paths = [], [], []
    metric_logger = MetricLogger(delimiter="  ")

    model.eval()

    with torch.no_grad():
        for imgs, label, obj_id, view_paths in metric_logger.log_every(dataloader, 50):
            imgs = imgs.squeeze(0).cuda()  # (V, 3, H, W)
            # labels = labels.cuda()
            feats = model(imgs)  # (V, D)
            feat_mean = feats.mean(dim=0)  # (D,)
            embeddings.append(feat_mean.cpu())
            labels.append(label)
            paths.append(view_paths[0][0])

            # paths.append(view_paths[opt.n_view+2][0])

    return torch.stack(embeddings), labels, paths


# ------------------------- Retrieval -------------------------
def retrieve(query_feat, train_feats, topk=5):

    query_feat = F.normalize(query_feat.unsqueeze(0), dim=1)
    
    train_feats = F.normalize(train_feats, dim=1)
    
    sims = torch.mm(query_feat, train_feats.T).squeeze(0)
    
    vals, indices = sims.topk(topk)
    
    return indices, vals


def plot_retrieval(query_path, retrieved_paths, retrieved_labels, query_label, save_path=None, shuffle=True):
    results = list(zip(retrieved_paths, retrieved_labels))
    if shuffle:
        random.shuffle(results)
    retrieved_paths, retrieved_labels = zip(*results)

    fig, axes = plt.subplots(1, len(retrieved_paths)+1, figsize=(2 * (len(retrieved_paths)+1), 3))
    axes[0].imshow(Image.open(query_path))
    axes[0].set_title("Query")
    axes[0].axis("off")

    for i, (img_path, lbl) in enumerate(zip(retrieved_paths, retrieved_labels)):
        img = Image.open(img_path)
        axes[i+1].imshow(img)
        axes[i+1].axis("off")
        axes[i+1].set_title(f"Top {i+1}")
        if lbl != query_label:
            rect = patches.Rectangle((0, 0), 1, 1, transform=axes[i+1].transAxes,
                                    linewidth=6, edgecolor='red', facecolor='none')
            axes[i+1].add_patch(rect)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def average_precision(query_label, retrieved_labels):
    correct = 0
    precisions = []
    for i, label in enumerate(retrieved_labels):
        if label == query_label:
            correct += 1
            precisions.append(correct / (i + 1))
    if len(precisions) == 0:
        return 0.0
    return sum(precisions) / len(precisions)


def mean_average_precision(features, labels):
    N = len(features)
    aps = []

    for i in range(N):
        query = features[i].unsqueeze(0)  # (1, D)
        sims = F.cosine_similarity(query, features)  # (N,)
        sims[i] = -1.0  # Ignore self-match

        indices = torch.argsort(sims, descending=True)
        retrieved_labels = [labels[idx] for idx in indices.tolist()]
        query_label = labels[i]
        ap = average_precision(query_label, retrieved_labels)
        aps.append(ap)

    return sum(aps) / len(aps)


def mean_average_precision_at_k(query_feats, query_labels, db_feats, db_labels, ks=[5, 10, 100]):
    
    query_feats = F.normalize(query_feats, dim=1)
    db_feats = F.normalize(db_feats, dim=1)

    map_results = {}

    for k in ks:
        aps = []
        for i in range(len(query_feats)):
            qf = query_feats[i].unsqueeze(0)
            ql = query_labels[i]
            sims = torch.mm(qf, db_feats.T).squeeze(0)
            sorted_indices = torch.argsort(sims, descending=True)
            sorted_labels = [db_labels[idx] for idx in sorted_indices[:k]]

            hits = [1 if lbl == ql else 0 for lbl in sorted_labels]
            if sum(hits) == 0:
                aps.append(0.0)
                continue

            precisions = [sum(hits[:j+1]) / (j+1) for j in range(len(hits)) if hits[j] == 1]
            ap = sum(precisions) / sum(hits)
            aps.append(ap)

        map_results[f'mAP@{k}'] = round(np.mean(aps), 4)

    return map_results



# ------------------------- Main -------------------------
def main():
    
    opt = parse_option()

    #Folder to save the output images to retrieval
    os.makedirs(opt.output_folder, exist_ok=True)

    train_loader, test_loader = set_loader(opt)
    
    model = set_model(opt)

    print("Extracting train embeddings...")
    train_feats, train_labels, train_paths = extract_embeddings(train_loader, model, opt)
    torch.save(train_feats, 'trainfeat.pth')
    torch.save(train_labels, 'trainlabels.pth')

    print("Extracting test embeddings...")
    test_feats, test_labels, test_paths = extract_embeddings(test_loader, model, opt)
    torch.save(test_feats, 'testfeat.pth')
    torch.save(test_labels, 'testlabels.pth')

    mAP = mean_average_precision(train_feats, train_labels)
    print(f"[INFO] mAP (self-retrieval on train set): {mAP:.4f}")

    mAPt = mean_average_precision(test_feats, test_labels)
    print(f"[INFO] mAP (self-retrieval on test set): {mAPt:.4f}")

    

    results = mean_average_precision_at_k(
        query_feats=test_feats,
        query_labels=test_labels,
        db_feats=train_feats,
        db_labels=train_labels,
        ks=[1, 5, 10, 100]
    )

    for k, score in results.items():
        print(f"[INFO] {k}: {score}")

    for topk in [5, 10]:
        
        if opt.query_label is not None:
            query_indices = [i for i, label in enumerate(test_labels) if str(label[0]).lower() == opt.query_label.lower()]
            if not query_indices:
                print(f"[ERROR] No test samples found with label: {opt.query_label}")
                continue

            for i in query_indices:
                
                query_feat = test_feats[i]
                query_label = test_labels[i]
                query_path = test_paths[i]

                indices, _ = retrieve(query_feat, train_feats, topk=topk)
                retrieved_paths = [train_paths[idx] for idx in indices]
                retrieved_labels = [train_labels[idx] for idx in indices]
                
                basename = os.path.basename(query_path)
                save_name = os.path.join(opt.output_folder, f"shape_retr_top{topk}_query_{i}_{basename}")

                # save_name = os.path.join(opt.output_folder, f"shape_retr_query_{i}_top{topk}_{str(retrieved_paths[0])}.png")
                plot_retrieval(
                    query_path,
                    retrieved_paths,
                    retrieved_labels,
                    query_label,
                    save_path=str(save_name),
                    shuffle=True
                )


if __name__ == '__main__':
    main()
