import timm
import torch.nn as nn

class VisionTransformer32(nn.Module):
    def __init__(self, 
                 img_size=32, 
                 patch_size=4, 
                 embed_dim=384, 
                 hidden_dim=2048, 
                 feat_dim=128, 
                 num_classes=0):
        
        super(VisionTransformer32, self).__init__()
        # Configuração do Vision Transformer para imagens pequenas
        self.vit = timm.create_model(
            'vit_small_patch16_224',  # Modelo base
            pretrained=False,
            img_size=img_size,        # Imagem de entrada 32x32
            patch_size=patch_size,    # Tamanho do patch (4x4 ou 8x8)
            num_classes=0   # Sem cabeça de classificação padrão
        )
        # Camada de projeção adicional (2048 -> 128)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),  # Projeção intermediária
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),        # Projeção final
            nn.ReLU()
        )

        # Camada final para classificação (128 -> 9 classes)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # Extração de embeddings do ViT
        vit_output = self.vit(x)  # Saída do class token
        # Projeção para espaço latente menor
        projected = self.projection_head(vit_output)
        # Classificação final
        features = self.classifier(projected)

        return features
