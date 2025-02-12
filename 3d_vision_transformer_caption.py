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
torch.manual_seed(1337)

# =============================================================================
# Updated Hyperparameters for Small, Skewed Data
# =============================================================================
batch_size = 8
epochs = 50
learning_rate = 1e-4           # Increased learning rate from 1e-6
n_heads = 8                  # Number of attention heads (each head has an embedding size of out_channels/n_heads)
n_layers = 4                 # Reduced number of transformer layers from 8 to 4
dropout = 0.1                # Added dropout for regularization (was 0)
img_size = 256
patch_size = 16
in_channels = 1
out_channels = 256           # Reduced model size from 512 to 256

# =============================================================================
# Load the data
# =============================================================================
# Initialize data loader
data_loader = OASISDataLoader(batch_size=batch_size, max_text_length=300)

# Get dataloaders
train_loader, test_loader, val_loader = data_loader.get_dataloaders()

# =============================================================================
# Define the Loss with Variance and Correlation Penalty
# =============================================================================
class MSEWithVarianceLoss(nn.Module):
    def __init__(self, lambda_var=1.0, alpha=1.5, lambda_corr=10, var_threshold=1e-2):
        """
        Args:
            lambda_var (float): Weight for the variance loss term.
            alpha (float): Exponent for the variance loss term.
            lambda_corr (float): Weight for the correlation penalty.
            var_threshold (float): A scaling constant used to gate the correlation penalty when variability is low.
        """
        super().__init__()
        self.mse_loss = nn.MSELoss()  
        self.lambda_var = lambda_var  
        self.alpha = alpha  
        self.lambda_corr = lambda_corr  
        self.var_threshold = var_threshold

    def forward(self, y_pred, y_true):
        # Standard MSE loss
        mse = self.mse_loss(y_pred, y_true)
        
        # Variance of true and predicted values (using biased variance)
        var_true = torch.var(y_true, unbiased=False)
        var_pred = torch.var(y_pred, unbiased=False)
        var_loss = torch.abs(var_true - var_pred) ** self.alpha
        
        # Compute means
        y_pred_mean = torch.mean(y_pred)
        y_true_mean = torch.mean(y_true)
        
        # Compute covariance
        cov = torch.mean((y_pred - y_pred_mean) * (y_true - y_true_mean))
        
        # Compute standard deviations
        std_y_pred = torch.std(y_pred)
        std_y_true = torch.std(y_true)
        
        # Clamp the standard deviations to avoid division by (or near) zero
        std_y_pred = torch.clamp(std_y_pred, min=1e-6)
        std_y_true = torch.clamp(std_y_true, min=1e-6)
        
        # Compute Pearson's correlation (add epsilon to the denominator for extra safety)
        pearson_corr = cov / (std_y_pred * std_y_true + 1e-8)
        
        # Compute variability (as product of standard deviations) and a weight for the correlation loss
        variability = std_y_pred * std_y_true
        corr_weight = variability / (variability + self.var_threshold)
        
        # Compute correlation loss term; note that if the predictions are constant, corr_loss will be small
        corr_loss = corr_weight * (1.0 - pearson_corr)
        
        # Sum up all components of the loss
        loss = mse + self.lambda_var * var_loss + (self.lambda_corr * corr_loss)
        
        return loss

# =============================================================================
# Embedding and Transformer Components
# =============================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, out_channels, image_size, patch_size, in_channels=1, auto_pad=True):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.auto_pad = auto_pad  # Option to auto-pad images if needed

        # Conv3D for patch embedding
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
        # Ensure input size is divisible by patch_size (optional padding)
        if self.auto_pad:
            x = self.pad_image(x)

        x = self.conv(x)  # (B, out_channels, H, W, D)
        x = self.bn(x)
        x = self.act(x)

        # Flatten spatial dimensions while keeping batch and channel dimensions
        x = x.flatten(2)  # (B, out_channels, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, out_channels)

        return x
    
    def pad_image(self, x):
        _, _, h, w, d = x.shape
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        pad_d = (self.patch_size - (d % self.patch_size)) % self.patch_size

        # Padding order: (depth, width, height)
        padding = (0, pad_d, 0, pad_w, 0, pad_h)
        return F.pad(x, padding, mode='constant', value=0)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, kernel_size, stride, padding):
        super().__init__()
        
        # Calculate number of patches per spatial dimension
        num_patches_h = ((img_size - kernel_size + 2 * padding) // stride) + 1
        num_patches_w = ((img_size - kernel_size + 2 * padding) // stride) + 1
        num_patches_d = ((img_size - kernel_size + 2 * padding) // stride) + 1
        
        # Total number of patches + 1 for the CLS token
        num_patches = num_patches_h * num_patches_w * num_patches_d
        total_seq_length = num_patches + 1
        
        self.positional_embeddings = nn.Parameter(torch.randn(1, total_seq_length, d_model))

    def forward(self, x):
        return x + self.positional_embeddings[:, :x.size(1), :]
    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(out_channels, head_size, bias=False)
        self.query = nn.Linear(out_channels, head_size, bias=False)
        self.value = nn.Linear(out_channels, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = self.softmax(wei)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
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
        self.proj = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    def __init__(self, out_channels, n_heads):
        super().__init__()
        head_size = out_channels // n_heads
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln3 = nn.LayerNorm(out_channels)
        self.sa_head = MultiHeadAttention(num_heads=n_heads, head_size=head_size)
        self.ff = FeedForward(out_channels)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ff(self.ln3(x))
        return x

# =============================================================================
# The Transformer Model for Regression
# =============================================================================
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(out_channels, img_size, patch_size, in_channels)
        self.pos_encoding = PositionalEncoding(out_channels, img_size, patch_size, patch_size, patch_size, 0)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Learnable classification token for regression
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))
        
        # Transformer blocks (reduced number)
        self.blocks = nn.Sequential(
            *[Block(out_channels, n_heads=n_heads) for _ in range(n_layers)]
        )
        
        self.ln_f = nn.LayerNorm(out_channels)
        self.regression_head = nn.Linear(out_channels, 1)

    def forward(self, x, targets=None):
        B = x.shape[0]
        x = self.patch_embedding(x)  # (B, num_patches, out_channels)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        
        # Use only the CLS token for regression
        x = x[:, 0, :]
        x = self.ln_f(x)
        x = self.regression_head(x)
        return x

# =============================================================================
# Training Setup
# =============================================================================
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device: ", device)

# Initialize and move model
model = Transformer().to(device)

# Print summary on CPU and then move back to device
summary(model.to('cpu'), (1, 256, 256, 128))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
# Use the custom loss that encourages variance in the predictions
loss_fn = MSEWithVarianceLoss(lambda_var=1.0, alpha=1.5, lambda_corr=10, var_threshold=1e-2)

# Learning rate scheduler (reduces LR when val loss plateaus)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# =============================================================================
# Training Loop
# =============================================================================
for epoch in range(epochs):
    model.train()
    train_loss_total = 0
    train_total = 0
    train_predictions = []
    train_targets = []
    for batch in train_loader:
        xb = batch['image']
        yb = batch['mmse']

        xb, yb = xb.to(device), yb.to(device)
        predictions = model(xb).squeeze(1)
        loss = loss_fn(predictions, yb)
        
        train_predictions.append(predictions.detach().cpu().numpy())
        train_targets.append(yb.detach().cpu().numpy())
        
        train_loss_total += loss.item()
        train_total += yb.size(0)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    train_predictions = np.concatenate(train_predictions)
    train_targets = np.concatenate(train_targets)
    train_pearson_corr = np.corrcoef(train_predictions, train_targets)[0, 1]
    
    # Validation
    model.eval()
    val_loss_total = 0
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            x_val = batch['image']
            y_val = batch['mmse']
            x_val, y_val = x_val.to(device), y_val.to(device)
            
            val_predictions_batch = model(x_val).squeeze(1)
            val_loss = loss_fn(val_predictions_batch, y_val)
            val_loss_total += val_loss.item()
            
            val_predictions.append(val_predictions_batch.detach().cpu().numpy())
            val_targets.append(y_val.detach().cpu().numpy())
    
    val_predictions = np.concatenate(val_predictions)
    val_targets = np.concatenate(val_targets)
    val_pearson_corr = np.corrcoef(val_predictions, val_targets)[0, 1]
    
    avg_train_loss = train_loss_total / len(train_loader)
    avg_val_loss = val_loss_total / len(val_loader)
    
    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch + 1}: train_loss {avg_train_loss:.4f}, val_loss {avg_val_loss:.4f}, "
          f"train_corr {train_pearson_corr:.4f}, val_corr {val_pearson_corr:.4f}")

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
        y_test = batch['mmse']
        x_test, y_test = x_test.to(device), y_test.to(device)
        predictions = model(x_test).squeeze(1)
        loss = loss_fn(predictions, y_test)
        avg_test_loss += loss.item()
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(y_test.cpu().numpy())

avg_test_loss = avg_test_loss / len(test_loader)
print(f"Test loss: {avg_test_loss:.4f}")

all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

pearson_corr = np.corrcoef(all_predictions, all_targets)[0, 1]
print(f"Pearson's correlation: {pearson_corr:.4f}")

predicted_variance = np.var(all_predictions)
true_variance = np.var(all_targets)
print(f"Predicted variance: {predicted_variance:.4f}")
print(f"True variance: {true_variance:.4f}")

