import torch
import timm
import networks.vit as vits
from networks.resnet_big import SupConResNet
from torchvision.models import resnet50, ResNet50_Weights


def load_pretrained_model(model_name):
    """Load a pre-trained model from the timm library or from the DINO repository.

    Args:
        model_name (str): Model name to be loaded. Supported models are:
            - resnet50
            - vit_small
            - vit_base
            - dino_vit_small_p_16
            - dino_vit_small_p_8
            - dino_vit_base_p_16
            - dino_vit_base_p_8

    Raises:
        ValueError: Unsupported model name.

    Returns:
        model (torch.nn.Module): Model with pre-trained weights based on the model_name.
    """    

    model_mapping = {
        "resnet50": ("resnet50", 128),
        "vit_small": ("vit_small_patch16_224", 384),
        "vit_base": ("vit_base_patch16_224", 768),
        "dino_vit_small_p_16": ("dino_vit_small_patch16_224", 384),
        "dino_vit_small_p_8": ("dino_vit_small_patch8_224", 384),
        "dino_vit_base_p_16": ("dino_vit_base_patch16_224", 768),
        "dino_vit_base_p_8": ("dino_vit_base_patch8_224", 768)
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    all_model_name, feat_dim = model_mapping[model_name]
    
    if model_name == "resnet50":
        # Visit: https://github.com/HobbitLong/SupContrast/issues/146
        pretrained_net = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = SupConResNet(name=model_name)
        pretrained_net.fc = torch.nn.Identity()
        model.encoder.load_state_dict(pretrained_net.state_dict(), strict=False)

        return model

    elif model_name == "vit_small" or model_name == "vit_base":
        pretrained_net = timm.create_model(all_model_name, pretrained=True)

    elif model_name == "dino_vit_small_p_16" or model_name == "dino_vit_small_p_8":
        if "p_16" in model_name:
            pretrained_net = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        elif "p_8" in model_name:
            pretrained_net = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
    
    elif model_name == "dino_vit_base_p_16" or model_name == "dino_vit_base_p_8":
        if "p_16" in model_name:
            pretrained_net = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        elif "p_8" in model_name:
            pretrained_net = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    pretrained_net.head = torch.nn.Identity()
    for param in pretrained_net.head.parameters():
        param.requires_grad = True
    model = vits.SupConViT(name=model_name, feat_dim=feat_dim)
    model.encoder.load_state_dict(pretrained_net.state_dict(), strict=False)
    
    return model