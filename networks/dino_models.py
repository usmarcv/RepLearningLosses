import torch
import torch.nn as nn

from torchvision import transforms
from timm.models.vision_transformer import VisionTransformer
from PIL import Image

#ViT's backbones
import networks.vit as vits

import os
import requests
import util

def download_checkpoint(model_name, save_dir="checkpoints"):
    """Baixa o checkpoint do DINO e salva localmente, se não existir."""
    # DINO MODELS URLs
    DINO_MODELS = {
        "dino_vit_small_p_16": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth",
        "dino_vit_small_p_8": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth",
        "dino_vit_base_p_16": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth",
        "dino_vit_base_p_8": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth"
    }

    if model_name not in DINO_MODELS:
        print(f"[ERROR] O modelo '{model_name}' não foi encontrado na lista de checkpoints.")
        return None

    url = DINO_MODELS[model_name]
    filename = os.path.join(save_dir, f"{model_name}.pth")

    # Verifique se o arquivo já existe
    if not os.path.exists(filename):
        print(f"[INFO] Download do checkpoint {model_name}...")
        os.makedirs(save_dir, exist_ok=True)

        # Baixando o arquivo
        with requests.get(url, stream=True) as r:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"[INFO] Checkpoint {model_name} baixado com sucesso.")
    else:
        print(f"[INFO] Checkpoint {model_name} já existe. Pulando o download.")

    return filename


def load_dino_model(model_name, checkpoint_path, checkpoint_key, linear_eval=False):
    """Carrega o modelo ViT DINO e remove a cabeça de classificação."""
    
    # Definição do tamanho da dimensão do embedding
    feat_dims = {
        "dino_vit_small_p_16": 384,
        "dino_vit_small_p_8": 384,
        "dino_vit_base_p_16": 768,
        "dino_vit_base_p_8": 768
    }
    
    if model_name not in feat_dims:
        raise ValueError(f"Modelo {model_name} não é um modelo DINOv1 suportado.")
    
    if linear_eval:
        # model = vits.SupConViT(model_name, feat_dim=feat_dims[model_name])
        model = VisionTransformer(
            img_size=224,
            patch_size=8 if "p_8" in model_name else 16,
            embed_dim=feat_dims[model_name],
            depth=12,
            num_heads=6 if "small" in model_name else 12,
            mlp_ratio=4,
        )

        # Carrega o checkpoint DINOv1
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=False)

        # Remove a cabeça de classificação
        model.head = nn.Identity()
        
    else:
        # Pretrained
        model = VisionTransformer(
            img_size=224,
            patch_size=8 if "p_8" in model_name else 16,
            embed_dim=feat_dims[model_name],
            depth=12,
            num_heads=6 if "small" in model_name else 12,
            mlp_ratio=4,
        )

        # Carrega o checkpoint DINOv1
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint[checkpoint_key], strict=False)

        # Remove a cabeça de classificação
        model.head = nn.Identity()

    return model

    

# def load_dino_model(model_name, checkpoint_path, checkpoint_key, linear_eval=False):
#     """Carrega o modelo ViT DINO e remove a cabeça de classificação."""
    
#     # Definição do tamanho da dimensão do embedding
#     feat_dims = {
#         "dino_vit_small_p_16": 384,
#         "dino_vit_small_p_8": 384,
#         "dino_vit_base_p_16": 768,
#         "dino_vit_base_p_8": 768
#     }
    
#     if model_name not in feat_dims:
#         raise ValueError(f"Modelo {model_name} não é um modelo DINOv1 suportado.")

#     if linear_eval:
#         # Carrega o modelo ViT usando timm
#         vit_model = VisionTransformer(
#             img_size=224,
#             patch_size=8 if "p_8" in model_name else 16,
#             embed_dim=feat_dims[model_name],
#             depth=12,
#             num_heads=6 if "small" in model_name else 12,
#             mlp_ratio=4,
#         )

#         checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
#         vit_model.load_state_dict(checkpoint, strict=False)
        
#         return vit_model
    
#     else:
#         # Carrega o modelo ViT usando timm
#         vit_model = VisionTransformer(
#             img_size=224,
#             patch_size=8 if "p_8" in model_name else 16,
#             embed_dim=feat_dims[model_name],
#             depth=12,
#             num_heads=6 if "small" in model_name else 12,
#             mlp_ratio=4,
#         )

#         # Carrega o checkpoint DINOv1
#         checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
#         vit_model.load_state_dict(checkpoint[checkpoint_key], strict=False)

#         # Remove a cabeça de classificação
#         vit_model.head = nn.Identity()

#         return vit_model


#DINO Repository copy-paste: https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/main_dino.py
class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            util.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            util.GaussianBlur(0.1),
            util.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            util.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops