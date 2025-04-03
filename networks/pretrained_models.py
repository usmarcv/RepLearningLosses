import timm
import torch
import torch.nn.functional as F
import networks.vit as vits
import networks.dinov2 as dinov2
from networks.resnet_big import SupConResNet
from torchvision.models import resnet50, ResNet50_Weights


def load_pretrained_model(model_name):
    """Load pretrained model from timm models or torch hub

    Args:
        model_name (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """    

    model_mapping = {
        "resnet50": ("resnet50", 128),
        "vit_small": ("vit_small_patch16_224", 384),
        "vit_base": ("vit_base_patch16_224", 768),
        "dino_vit_small_p_16": ("dino_vits16", 384),
        "dino_vit_small_p_8": ("dino_vits8", 384),
        "dino_vit_base_p_16": ("dino_vitb16", 768),
        "dino_vit_base_p_8": ("dino_vitb8", 768),
        "dinov2_vit_small_p_14": ("dinov2_vits14", 384),
        "dinov2_vit_base_p_14": ("dinov2_vitb14", 768)
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    all_model_name, feat_dim = model_mapping[model_name]
    
    if model_name == "resnet50":
        # Visit: https://github.com/HobbitLong/SupContrast/issues/146
        pretrained_net = resnet50(weights=ResNet50_Weights.DEFAULT)
        pretrained_net.fc = torch.nn.Identity()
        model = SupConResNet(name=model_name)
        model.encoder.load_state_dict(pretrained_net.state_dict(), strict=False)

        return model

    elif model_name == "vit_small" or model_name == "vit_base":
        pretrained_net = timm.create_model(all_model_name, pretrained=True)

    elif model_name == "dino_vit_small_p_16" or model_name == "dino_vit_small_p_8":
        if "p_16" in model_name:
            pretrained_net = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
            # pretrained_net = timm.create_model(all_model_name, pretrained=True)
        elif "p_8" in model_name:
            pretrained_net = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
    
    elif model_name == "dino_vit_base_p_16" or model_name == "dino_vit_base_p_8":
        if "p_16" in model_name:
            # pretrained_net = timm.create_model(all_model_name, pretrained=True)
            pretrained_net = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        elif "p_8" in model_name:
            pretrained_net = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
            # pretrained_net = timm.create_model(all_model_name, pretrained=True)
    elif model_name == "dinov2_vit_small_p_14" or model_name == "dinov2_vit_base_p_14":
        if "small" in model_name:
            pretrained_net = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14")
        elif "base" in model_name:
            pretrained_net = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14")
        
        pretrained_net.head = torch.nn.Identity()
        model = dinov2.DinoSupConViT(name=model_name, feat_dim=feat_dim)
        pretrained_pos_embed = pretrained_net.pos_embed  # Positional embeddings do modelo pré-treinado
        num_tokens = model.encoder.pos_embed.shape[1]  # Número esperado de tokens
        # Interpola para corresponder ao número esperado de tokens
        new_pos_embed = F.interpolate(pretrained_pos_embed.unsqueeze(0), 
                                        size=(num_tokens, pretrained_pos_embed.shape[2]), 
                                        mode='bilinear', align_corners=False).squeeze(0)

        # Substitui os embeddings posicionais no modelo carregado
        pretrained_net.pos_embed = torch.nn.Parameter(new_pos_embed)
        for param in pretrained_net.head.parameters():
            param.requires_grad = True
        model.encoder.load_state_dict(pretrained_net.state_dict(), strict=False)

        return model
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    pretrained_net.head = torch.nn.Identity()
    for param in pretrained_net.head.parameters():
        param.requires_grad = True
    model = vits.SupConViT(name=model_name, feat_dim=feat_dim)
    model.encoder.load_state_dict(pretrained_net.state_dict(), strict=False)
    
    return model