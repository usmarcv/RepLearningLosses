import timm
import torch.nn as nn

class ViTEncoder(nn.Module):
    def __init__(self, model_name="vit_small_patch16_32", pretrained=False): #vit_base_patch16_224
        super(ViTEncoder, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=10)  # num_classes=0 for feature extraction

    def forward(self, x):
        return self.vit(x)
