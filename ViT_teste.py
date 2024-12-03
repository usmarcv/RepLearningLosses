# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTConfig, ViTForImageClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# %%
# [markdown] Testando
# Testando o modelo ViT

# Define model configuration
config = ViTConfig(
    image_size=128,      # Input image size
    num_channels=1,
    patch_size=8,        # Patch size (8x8 pixels)
    num_labels=10,       # Number of output classes (e.g., for MNIST)
    hidden_size=384,     # Size of the hidden layer
    num_hidden_layers=12, # Number of Transformer layers
    num_attention_heads=6, # Number of attention heads
    intermediate_size=384 * 4, # Intermediate size in feedforward layers
)

# Initialize the model
model = ViTForImageClassification(config)

# %%
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
])

# Load the dataset
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# %% [markdown] Setting training loop

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = CrossEntropyLoss()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# %%
## Testing InfoNCE Loss

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTConfig, ViTModel
import torch.optim as optim


# Custom InfoNCE Loss
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = z_i / z_i.norm(dim=1, keepdim=True)
        z_j = z_j / z_j.norm(dim=1, keepdim=True)
        
        # Similarity matrix
        similarity_matrix = torch.mm(z_i, z_j.T) / self.temperature

        # Positive pairs are diagonal
        labels = torch.arange(batch_size).to(z_i.device)

        # Cross-entropy loss
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss


# Data Augmentation for Contrastive Learning
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(124),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomApply([transforms.ColorJitter()], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
transform = transforms.Compose([
    transforms.Resize((124, 124)),              # Ensure input size matches ViT requirements
    transforms.Grayscale(num_output_channels=3), # Convert grayscale to 3 channels (if necessary)
    transforms.ToTensor(),                      # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])


# Load MNIST Dataset (converting grayscale to 3-channel RGB)
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define ViT-S/8 Model
config = ViTConfig(
    image_size=124,
    patch_size=8,
    num_channels=3,  # RGB input
    hidden_size=384,
    num_hidden_layers=12,
    num_attention_heads=6,
    intermediate_size=384 * 4,
)

vit = ViTModel(config)


# Add Projection Head
class ContrastiveModel(nn.Module):
    def __init__(self, vit_model, projection_dim=128):
        super(ContrastiveModel, self).__init__()
        self.vit = vit_model
        self.projection_head = nn.Sequential(
            nn.Linear(config.hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, pixel_values):
        # Pass through ViT backbone
        outputs = self.vit(pixel_values=pixel_values)
        cls_token_embedding = outputs.last_hidden_state[:, 0]  # Use CLS token
        # Pass through projection head
        projection = self.projection_head(cls_token_embedding)
        return projection


model = ContrastiveModel(vit)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')


# Optimizer and Loss Function
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = InfoNCELoss()


# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        inputs, _ = batch  # Ignore labels
        inputs = inputs.to(device)

        # Generate two augmented views of the same batch
        inputs_aug1 = transform(inputs.cpu()).to(device)
        inputs_aug2 = transform(inputs.cpu()).to(device)

        # Forward pass for both views
        z_i = model(inputs_aug1)
        z_j = model(inputs_aug2)

        # Compute contrastive loss
        loss = criterion(z_i, z_j)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
# %%
