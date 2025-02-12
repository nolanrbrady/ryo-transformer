import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchsummary import summary

torch.manual_seed(1337)

# hyperparameters
batch_size = 16
max_iters = 35
learning_rate = 0.0005
n_heads = 4 # Number of attention heads (each head has an embedding size of 64)
n_layers = 8 # Number of layers
dropout = 0.15
img_size = 224
patch_size = 16
in_channels = 3
out_channels = 128
max_seq_length = (img_size // patch_size) ** 2 + 1  # 197 for 224x224 images with 16x16 patches

#==============================================
# Load dataset and embedding class
#==============================================
# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load the entire dataset
full_dataset = datasets.ImageFolder(root='./ADNI-toy', transform=transform)

# Define the split ratio
train_size = int(0.7 * len(full_dataset))  # 70% for training
val_size = int(0.15 * len(full_dataset))   # 15% for validation
test_size = len(full_dataset) - train_size - val_size  # 15% for testing

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check the number of classes
num_classes = len(full_dataset.classes)
print(f"Number of classes: {num_classes}")
print(train_loader)


class PatchEmbedding(nn.Module):
    def __init__(self, out_channels, image_size, patch_size, in_channels=3):
        super().__init__()
        assert out_channels % patch_size == 0, "out_channels must be divisible by patch_size"
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.patch_size = patch_size

        # Single convolutional layer with increased output channels
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,  # Overlapping patches
            padding=0  # To maintain dimensions
        )
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, img_size, patch_size):
        super().__init__()
        
        # Calculate the number of patches with overlapping
        num_patches = (img_size // patch_size) * (img_size // patch_size)  # 14 * 14 = 196
        total_seq_length = num_patches + 1  # +1 for CLS = 197
        
        # Learnable positional embeddings (including position for CLS token)
        self.positional_embeddings = nn.Parameter(torch.randn(1, total_seq_length, d_model))

    def forward(self, x):
        # Add positional encoding to embeddings
        x = x + self.positional_embeddings[:, :x.size(1), :]
        return x


class Head(nn.Module):
    """
    Implements a single head of self-attention

    Query: What you're interested in (e.g., books about space).
    Key: The catalog entries for each book (e.g., subject tags).
    Value: The actual content of the books.

    Overall, the attention mechanism allows the model to focus on relevant parts of the input and combine information from different parts of the input.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(out_channels, head_size, bias=False)
        self.query = nn.Linear(out_channels, head_size, bias=False)
        self.value = nn.Linear(out_channels, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimension
        k = self.key(x) # K is the address for each peice of information (encodes relevance)
        q = self.query(x) # Q is the question or focus of the attention mechanism
        wei = q @ k.transpose(-2, -1) * C**-0.5 # Weights for the attention mechanism
        wei = self.softmax(wei)
        wei = self.dropout(wei)
        v = self.value(x) # V is the value of the information (encodes content)
        out = wei @ v
        return out
    
class FeedForward(nn.Module):
    """
    Implements a feed-forward layer
    """
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
    

class ChannelAttention(nn.Module):
    """Channel-wise attention module using squeeze-and-excitation style attention"""
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C = x.shape
        # Aggregate spatial information
        y = self.avg_pool(x.transpose(1, 2))  # (B, C, 1)
        y = y.view(B, C)
        # Learn channel-wise attention weights
        y = self.fc(y).view(B, 1, C)
        return x * y.expand_as(x)

class Block(nn.Module):
    """Modified block with both spatial and channel attention"""
    def __init__(self, out_channels, n_heads):
        super().__init__()
        head_size = out_channels // n_heads
        
        # Layer normalizations
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.ln3 = nn.LayerNorm(out_channels)
        
        # Attention and feedforward layers
        self.sa_head = MultiHeadAttention(num_heads=n_heads, head_size=head_size)
        self.ca_head = ChannelAttention(out_channels)  # New channel attention
        self.ff = FeedForward(out_channels)

    def forward(self, x):
        # Apply layer norm before each operation
        x = x + self.sa_head(self.ln1(x))  # Spatial attention
        x = x + self.ca_head(self.ln2(x))  # Channel attention
        x = x + self.ff(self.ln3(x))       # Feed-forward
        return x

class MultiHeadAttention(nn.Module):
    """
    Implements multiple heads of self-attention
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Patch embedding and positional encoding
        self.patch_embedding = PatchEmbedding(out_channels, img_size, patch_size, in_channels)
        self.pos_encoding = PositionalEncoding(out_channels, img_size, patch_size)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))
        
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(out_channels, n_heads=n_heads) for _ in range(n_layers)]
        )
        
        # Final layer norm and classification head
        self.ln_f = nn.LayerNorm(out_channels)
        self.head = nn.Linear(out_channels, num_classes)

    def forward(self, x, targets=None):
        B = x.shape[0]  # batch size
        
        # Get patch embeddings
        x = self.patch_embedding(x)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout to embeddings
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        x = self.blocks(x)
        
        # Use only the CLS token for classification
        x = self.ln_f(x[:, 0])  # Take only the CLS token
        
        # Classification head
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss


#==============================================
# Training the model
#==============================================
# Check if GPU is available and set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device: ", device)

# Initialize the model and move it to the device
m = Transformer().to(device)

# Move to CPU for summary, then back to MPS
summary(m.to('cpu'), (3, 224, 224))  # Generate summary on CPU
m.to(device)  # Move back to MPS

# Use DataLoader to get batches
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-4)
for steps in range(max_iters):
    # Run the training data
    m.train()
    train_loss_total = 0
    total_train_accuracy = 0
    train_total = 0
    for xb, yb in train_loader:
        # Move data to the device
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward pass
        logits, loss = m(xb, yb)
        
        # Calculate training accuracy
        preds = torch.argmax(logits, dim=1)
        train_accuracy = (preds == yb).sum().item()

        # Calculate training loss
        train_loss_total += loss.item()
        total_train_accuracy += train_accuracy
        train_total += yb.size(0)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Run the validation data
    m.eval()
    val_loss_total = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            # Move validation data to the device
            x_val, y_val = x_val.to(device), y_val.to(device)
            
            # Validation forward pass
            val_logits, val_loss = m(x_val, y_val)
            val_loss_total += val_loss.item()
            
            # Calculate validation accuracy
            val_preds = torch.argmax(val_logits, dim=1)
            correct += (val_preds == y_val).sum().item()
            total += y_val.size(0)
        
        # Calculate average training loss and accuracy
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_accuracy = total_train_accuracy / train_total
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss_total / len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch {steps + 1}: train_loss {avg_train_loss}, train_accuracy {avg_train_accuracy}, val_loss {avg_val_loss}, val_accuracy {val_accuracy}")
