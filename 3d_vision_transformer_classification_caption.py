import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Sampler, Dataset
from torchsummary import summary
from sklearn.model_selection import GroupShuffleSplit
import nibabel as nib  # For loading 3D MRI scans
from oasis_data_loader import OASISDataLoader
from transformers import T5Tokenizer

# Set manual seed for reproducibility
torch.manual_seed(1337)

# =============================================================================
# Updated Hyperparameters for More Optimal Training
# =============================================================================
batch_size = 16
epochs = 100
learning_rate = 1e-5           # You may try a slightly higher LR if needed (e.g. 1e-4)
n_heads = 8                  # Number of attention heads
n_layers = 4                 # Number of transformer layers
dropout = 0.05               # Dropout rate
patch_size = 32
in_channels = 1
out_channels = 256           # Model capacity

# Early stopping parameters
patience = 20              # Early stopping patience

# Here we define the image dimensions.
# Note: For Conv3d, PyTorch expects (B, C, D, H, W)
# In your summary call you pass (1, 256, 256, 128) so we assume:
img_depth = 256    # e.g. along D
img_height = 256   # e.g. along H
img_width = 128    # e.g. along W
img_dims = (img_depth, img_height, img_width)

# =============================================================================
# Load the data
# =============================================================================
print("Initializing data loader")
data_loader = OASISDataLoader(batch_size=batch_size, max_text_length=300)

print("Getting dataloaders")
train_loader, test_loader, val_loader = data_loader.get_dataloaders()
print("Finished loading the data")

# =============================================================================
# Embedding and Transformer Components
# =============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, out_channels, patch_size, in_channels=1, auto_pad=True):
        """
        Uses a 3D convolution to extract non-overlapping patch embeddings.
        """
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.auto_pad = auto_pad

        self.conv = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(self.patch_size, self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size, self.patch_size),
            padding=0  # We handle padding manually if needed
        )
        self.bn = nn.BatchNorm3d(self.out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # x shape: (B, C, D, H, W)
        if self.auto_pad:
            x = self.pad_image(x)
        x = self.conv(x)  # (B, out_channels, new_D, new_H, new_W)
        x = self.bn(x)
        x = self.act(x)
        # Flatten spatial dimensions: (B, out_channels, num_patches)
        x = x.flatten(2)  
        x = x.transpose(1, 2)  # (B, num_patches, out_channels)
        return x

    def pad_image(self, x):
        # x shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        pad_d = (self.patch_size - (D % self.patch_size)) % self.patch_size
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
        # F.pad expects (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        x = F.pad(x, padding, mode='constant', value=0)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, img_dims, patch_size):
        """
        Compute positional embeddings based on the input image dimensions.
        img_dims is a tuple: (depth, height, width)
        """
        super().__init__()
        D, H, W = img_dims
        # Compute the number of patches along each dimension using ceiling division
        num_patches_d = math.ceil(D / patch_size)
        num_patches_h = math.ceil(H / patch_size)
        num_patches_w = math.ceil(W / patch_size)
        num_patches = num_patches_d * num_patches_h * num_patches_w
        total_seq_length = num_patches + 1  # +1 for the CLS token
        self.positional_embeddings = nn.Parameter(torch.randn(1, total_seq_length, d_model))

    def forward(self, x):
        # x shape: (B, seq_length, d_model)
        # Truncate or broadcast as needed
        return x + self.positional_embeddings[:, :x.size(1), :]

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(out_channels, head_size, bias=False)
        self.query = nn.Linear(out_channels, head_size, bias=False)
        self.value = nn.Linear(out_channels, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, out_channels)
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        # Scale by square root of head size
        scale = math.sqrt(self.head_size)
        wei = (q @ k.transpose(-2, -1)) / scale  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, head_size)
        return out

class FeedForward(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(out_channels, 4 * out_channels),
            nn.ReLU(),
            nn.Linear(4 * out_channels, out_channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Concatenate the outputs of all heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    def __init__(self, out_channels, n_heads):
        super().__init__()
        head_size = out_channels // n_heads
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.sa_head = MultiHeadAttention(num_heads=n_heads, head_size=head_size)
        self.ff = FeedForward(out_channels)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# =============================================================================
# The Transformer Model for Classification
# =============================================================================
class Transformer(nn.Module):
    def __init__(self, img_dims):
        super().__init__()
        self.patch_embedding = PatchEmbedding(out_channels, patch_size, in_channels)
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))
        self.pos_encoding = PositionalEncoding(out_channels, img_dims, patch_size)
        self.pos_drop = nn.Dropout(p=dropout)
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(out_channels, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(out_channels)
        # Classifier to 3 classes
        self.classifier = nn.Linear(out_channels, 3)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        # Add CLS token at the beginning of the sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        # Use only the CLS token for classification
        x = x[:, 0, :]
        x = self.ln_f(x)
        x = self.classifier(x)
        return x

# =============================================================================
# Training Setup
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Initialize and move model
model = Transformer(img_dims=img_dims).to(device)

# (Optional) Apply Xavier initialization for better training stability.
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)

# Print model summary on CPU before moving back to GPU
model_cpu = model.cpu()      # Create a CPU copy for summary
summary(model_cpu, (in_channels, img_depth, img_height, img_width), device="cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

# Use Cosine Annealing LR scheduler for smooth learning rate adjustments.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# =============================================================================
# Training Loop
# =============================================================================
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    train_loss_total = 0
    train_predictions = []
    train_targets = []
    
    for batch in train_loader:
        xb = batch['image']
        yb = batch['group']

        xb, yb = xb.to(device), yb.to(device)
        predictions = model(xb)
        loss = loss_fn(predictions, yb.long())
        
        train_predictions.append(predictions.detach().cpu().numpy())
        train_targets.append(yb.detach().cpu().numpy())
        
        train_loss_total += loss.item()
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # ---- Gradient Clipping Added Here ----
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    train_predictions = np.concatenate(train_predictions)
    train_targets = np.concatenate(train_targets)
    train_acc = (torch.argmax(torch.tensor(train_predictions), dim=1) == torch.tensor(train_targets)).float().mean()
    
    # Validation phase
    model.eval()
    val_loss_total = 0
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            x_val = batch['image']
            y_val = batch['group']
            x_val, y_val = x_val.to(device), y_val.to(device)
            
            val_preds = model(x_val)
            loss = loss_fn(val_preds, y_val.long())
            val_loss_total += loss.item()
            
            val_predictions.append(val_preds.detach().cpu().numpy())
            val_targets.append(y_val.detach().cpu().numpy())
    
    val_predictions = np.concatenate(val_predictions)
    val_targets = np.concatenate(val_targets)
    val_acc = (torch.argmax(torch.tensor(val_predictions), dim=1) == torch.tensor(val_targets)).float().mean()
    
    avg_train_loss = train_loss_total / len(train_loader)
    avg_val_loss = val_loss_total / len(val_loader)
    
    # Step the scheduler at the end of the epoch.
    scheduler.step()
    
    print(f"Epoch {epoch + 1}: train_loss {avg_train_loss:.4f}, val_loss {avg_val_loss:.4f}, "
          f"train_acc {train_acc:.4f}, val_acc {val_acc:.4f}")
    
    # Early stopping logic based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        break

# =============================================================================
# Testing the Model
# =============================================================================
model.eval()
avg_test_loss = 0
all_predictions = []
all_targets = []
with torch.no_grad():
    for batch in test_loader:
        x_test = batch['image']
        y_test = batch['group']
        x_test, y_test = x_test.to(device), y_test.to(device)
        predictions = model(x_test)
        loss = loss_fn(predictions, y_test.long())
        avg_test_loss += loss.item()
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(y_test.cpu().numpy())

avg_test_loss = avg_test_loss / len(test_loader)
print(f"Test loss: {avg_test_loss:.4f}")

all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

test_acc = (torch.argmax(torch.tensor(all_predictions), dim=1) == torch.tensor(all_targets)).float().mean()
print(f"Test accuracy: {test_acc:.4f}")

