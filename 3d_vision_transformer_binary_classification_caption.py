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
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# =============================================================================
# Hyperparameters
# =============================================================================
batch_size = 8
epochs = 300
# learning_rate = 0.0001           # You may try a slightly higher LR if needed (e.g. 1e-4)
learning_rate = 0.0003           # You may try a slightly higher LR if needed (e.g. 1e-4)
n_heads = 4                  # Number of attention heads
n_layers = 3                 # Number of transformer layers
dropout = 0.2              # Dropout rate
patch_size = 16
in_channels = 1
out_channels = 128           # Model capacity
embedding_text_dim = 128     # Model capacity
vocab_size = tokenizer.vocab_size           # Vocabulary size based on  t5-small tokenizer
max_seq_length = 300          # Maximum sequence length

# Early stopping parameters
patience = 20              # Early stopping patience

# Here we define the image dimensions.
# Note: For Conv3d, PyTorch expects (B, C, D, H, W)
# In your summary call you pass (1, 256, 256, 128) so we assume:
img_depth = 128    # e.g. along D
img_height = 256   # e.g. along H
img_width = 256    # e.g. along W
img_dims = (img_depth, img_height, img_width)

print("Hyperparameters:")
print("batch_size:", batch_size)
print("epochs:", epochs)
print("learning_rate:", learning_rate)
print("n_heads:", n_heads)
print("n_layers:", n_layers)
print("dropout:", dropout)
print("patch_size:", patch_size)
print("in_channels:", in_channels)
print("out_channels:", out_channels)
print("embedding_text_dim:", embedding_text_dim)
print("vocab_size:", vocab_size)
print("max_seq_length:", max_seq_length)
print("patience:", patience)
print("img_depth:", img_depth)
print("img_height:", img_height)
print("img_width:", img_width)
print("img_dims:", img_dims)
print("================================================")
print("Notes: ")
print("Testing the model without teacher forcing in the test set. This also incorporates the Gumbel-Softmax differentiable approximation to sample the next token.")
print("This model is also using the T5 start token for the decoder input and checking for the EOS token to break early.")
print("Also trying out a learnable temperature parameter for the Gumbel-Softmax formula.")
print("================================================")

# =============================================================================
# Load the data
# =============================================================================
print("Initializing data loader")
data_loader = OASISDataLoader(batch_size=batch_size, max_text_length=max_seq_length)

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

# This decoders the written text from the MRI scan.
class TransformerDecoder(nn.Module):  
    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout, max_seq_length=300):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, decoder_tokens, encoder_output):
        """Now takes embedded tokens instead of IDs"""
        B, T, _ = decoder_tokens.shape
        
        positions = torch.arange(0, T, device=decoder_tokens.device).unsqueeze(0).expand(B, T)
        positional_embeddings = self.positional_encoding(positions)
        
        # Combine embeddings
        x = decoder_tokens + positional_embeddings
        x = self.dropout(x)
        
        # Transformer processing
        x = x.transpose(0, 1)
        encoder_output = encoder_output.transpose(0, 1)
        
        mask = torch.tril(torch.ones((T, T), device=x.device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        
        x = self.decoder(x, encoder_output, tgt_mask=mask)
        x = x.transpose(0, 1)
        return self.fc_out(x)


# =============================================================================
# The Transformer Model for Classification
# =============================================================================
class Transformer(nn.Module):
    def __init__(self, img_dims, vocab_size, embedding_text_dim):
        super().__init__()
        self.patch_embedding = PatchEmbedding(out_channels, patch_size, in_channels)
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))
        self.pos_encoding = PositionalEncoding(out_channels, img_dims, patch_size)
        self.pos_drop = nn.Dropout(p=dropout)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        # Transformer blocks (encoder)
        self.blocks = nn.Sequential(
            *[Block(out_channels, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(out_channels)
        # Classifier head (for binary or multi-class classification)
        self.classifier = nn.Linear(out_channels, 2)
        # Decoder head for generating natural language captions.
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=embedding_text_dim,  # Note: adjust if you want to match out_channels
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        # Add start token embedding
        self.start_token = nn.Parameter(torch.randn(1, 1, embedding_text_dim))  # Learnable start token

    def forward(self, x):
        """Forward pass using only MRI features"""
        B = x.shape[0]
        
        # Existing encoder processing
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        enc_out = torch.cat((cls_tokens, x), dim=1)
        enc_out = self.pos_encoding(enc_out)
        enc_out = self.pos_drop(enc_out)
        enc_out = self.blocks(enc_out) # (B, T, out_channels)
        
        # Classification head
        cls_feature = enc_out[:, 0, :]
        cls_logits = self.classifier(self.ln_f(cls_feature))
        
        # Decoder initialization
        start_tokens = self.start_token.expand(B, -1, -1)  # (B, 1, embedding_dim)
        decoder_output = self.decode_autoregressive(enc_out)
        
        return cls_logits, decoder_output

    def decode_autoregressive(self, encoder_output):
        """
        Auto-regressive decoding without teacher forcing:
          - Starts with the T5 start token.
          - Uses Gumbel-Softmax for differentiable token selection.
          - Breaks early if all sequences generate the EOS token.
          
        Returns:
          Tensor of shape (B, max_seq_length, vocab_size) containing the logits.
        """
        B = encoder_output.shape[0]
        # Use T5's pad_token_id as the start token
        start_token_id = tokenizer.pad_token_id
        # Initialize the decoder input with the T5 start token.
        start_tokens = torch.full((B, 1), start_token_id, device=encoder_output.device, dtype=torch.long)
        # Embed the start tokens.
        current_tokens = self.decoder.token_embedding(start_tokens)  # (B, 1, d_model)
        
        # Pre-allocate a tensor to hold the logits at each timestep.
        all_logits = torch.zeros((B, max_seq_length, vocab_size), device=encoder_output.device)
        
        # Create an index tensor to convert one-hot vectors to token IDs.
        index_tensor = torch.arange(vocab_size, device=encoder_output.device).unsqueeze(0)  # (1, vocab_size)
        
        for t in range(max_seq_length):
            # Get decoder output for the current tokens.
            dec_out = self.decoder(current_tokens, encoder_output)  # (B, current_length, vocab_size)
            # Extract the logits for the last time step.
            current_logits = dec_out[:, -1, :]  # (B, vocab_size)
            all_logits[:, t, :] = current_logits
            
            # Use Gumbel-Softmax to sample next token (differentiable sample).
            tau = torch.exp(self.log_tau)  # learnable temperature parameter
            next_token_dist = F.gumbel_softmax(current_logits, tau=tau, hard=True)  # (B, vocab_size)
            
            # Compute the predicted token id in a differentiable manner.
            predicted_token_ids = (next_token_dist * index_tensor).sum(dim=-1)  # (B,)
            
            # Break early if all sequences have generated the EOS token.
            if (predicted_token_ids == tokenizer.eos_token_id).all():
                break
            
            # Compute the "next" embedding as a weighted sum of the embedding matrix.
            next_embed = torch.matmul(next_token_dist, self.decoder.token_embedding.weight)  # (B, d_model)
            next_embed = next_embed.unsqueeze(1)  # (B, 1, d_model)
            
            # Append the next embedding to the current tokens.
            current_tokens = torch.cat([current_tokens, next_embed], dim=1)
        
        return all_logits  # (B, max_seq_length, vocab_size)

# =============================================================================
# Training Setup
# =============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device:", device)

# Initialize and move model
model = Transformer(img_dims=img_dims, vocab_size=vocab_size, embedding_text_dim=embedding_text_dim).to(device)

# (Optional) Apply Xavier initialization for better training stability.
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)

# Print model summary on CPU before moving back to GPU
# model_cpu = model.cpu()      # Create a CPU copy for summary
# summary(model_cpu, [(in_channels, img_depth, img_height, img_width), (max_seq_length,)], device="cpu")
# model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

# Use Cosine Annealing LR scheduler for smooth learning rate adjustments.
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

# =============================================================================
# Training Loop
# =============================================================================
best_val_loss = float('inf')
epochs_no_improve = 0
accumulation_steps = 4  # Adjust this value based on your requirements

for epoch in range(epochs):
    model.train()
    train_loss_total = 0
    train_text_loss_total = 0  # Accumulator for training text loss
    train_predictions = []
    train_targets = []
    
    optimizer.zero_grad(set_to_none=True)  # Clear gradients at the start of each epoch
    
    for i, batch in enumerate(train_loader):
        # Add memory cleanup
        if device == "mps":
            torch.mps.empty_cache()  # Add MPS-specific memory clearing
        
        xb = batch['image'].to(device)
        yb = batch['group'].to(device)  
        labels = batch['labels'].to(device)

        # Forward pass (no decoder input)
        class_pred, text_pred = model(xb)
        
        # Calculate losses
        class_loss = loss_fn(class_pred, yb.long())
        assert text_pred.shape[:2] == labels.shape, f"Text pred shape {text_pred.shape} vs labels {labels.shape}"
        text_loss = loss_fn(text_pred.view(-1, vocab_size), labels[:, :max_seq_length].contiguous().view(-1).long())
        loss = class_loss + text_loss
        
        train_predictions.append(class_pred.detach().cpu().numpy())
        train_targets.append(yb.detach().cpu().numpy())
        
        train_loss_total += loss.item()
        train_text_loss_total += text_loss.item()  # Accumulate text loss
        
        loss = loss / accumulation_steps
        loss.backward()
        
        # After accumulation_steps batches, clip gradients, update the parameters, and reset gradients.
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Add memory cleanup
        del class_pred, text_pred
        if device == "mps":
            torch.mps.synchronize()  # Ensure MPS operations complete before continuing
    
    train_predictions = np.concatenate(train_predictions)
    train_targets = np.concatenate(train_targets)
    train_acc = (torch.argmax(torch.tensor(train_predictions), dim=1) == torch.tensor(train_targets)).float().mean()
    
    avg_train_loss = train_loss_total / len(train_loader)
    avg_train_text_loss = train_text_loss_total / len(train_loader)
    
    # =============================================================================
    # Validation phase
    # =============================================================================
    model.eval()
    val_loss_total = 0
    val_text_loss_total = 0  # Accumulator for validation text loss
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            x_val = batch['image'].to(device)
            y_val = batch['group'].to(device)
            labels = batch['labels'].to(device)

            # Normalize tokens for CPU compatibility
            labels = labels.masked_fill(labels == -100, 0)

            val_class_pred, val_text_pred = model(x_val)
            class_loss = loss_fn(val_class_pred, y_val.long())
            text_loss = loss_fn(val_text_pred.view(-1, vocab_size), labels[:, :max_seq_length].contiguous().view(-1).long())
            loss = class_loss + text_loss
            val_loss_total += loss.item()
            val_text_loss_total += text_loss.item()
            
            val_predictions.append(val_class_pred.detach().cpu().numpy())
            val_targets.append(y_val.detach().cpu().numpy())
    
    val_predictions = np.concatenate(val_predictions)
    val_targets = np.concatenate(val_targets)
    val_acc = (torch.argmax(torch.tensor(val_predictions), dim=1) == torch.tensor(val_targets)).float().mean()
    
    avg_val_loss = val_loss_total / len(val_loader)
    avg_val_text_loss = val_text_loss_total / len(val_loader)
    
    # Step the scheduler at the end of the epoch.
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch + 1}: train_loss {avg_train_loss:.4f}, train_text_loss {avg_train_text_loss:.4f}, "
          f"val_loss {avg_val_loss:.4f}, val_text_loss {avg_val_text_loss:.4f}, "
          f"train_acc {train_acc:.4f}, val_acc {val_acc:.4f}")
    
    # Early stopping logic
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
test_text_loss_total = 0  # Accumulator for test text loss
all_predictions = []
all_targets = []
all_texts_output = []
all_text_predictions = []
with torch.no_grad():
    for batch in test_loader:
        x_test = batch['image'].to(device)  
        y_test = batch['group'].to(device)
        labels = batch['labels'].to(device)

        # Normalize tokens for CPU compatibility
        labels = labels.masked_fill(labels == -100, 0)

        test_class_pred, test_text_pred = model(x_test)
        class_loss = loss_fn(test_class_pred, y_test.long())
        text_loss = loss_fn(test_text_pred.view(-1, vocab_size), labels[:, :max_seq_length].contiguous().view(-1).long())
        loss = class_loss + text_loss

        avg_test_loss += loss.item()
        test_text_loss_total += text_loss.item()  # Accumulate test text loss
        all_predictions.append(test_class_pred.cpu().numpy())
        all_targets.append(y_test.cpu().numpy())
        all_texts_output.append(test_text_pred.cpu().numpy())
        all_text_predictions.append(test_text_pred.cpu().numpy())

avg_test_loss = avg_test_loss / len(test_loader)
avg_test_text_loss = test_text_loss_total / len(test_loader)
print(f"Test loss: {avg_test_loss:.4f}, test_text_loss: {avg_test_text_loss:.4f}")

all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)
all_texts_output = np.concatenate(all_texts_output)
all_text_predictions = np.concatenate(all_text_predictions)

test_acc = (torch.argmax(torch.tensor(all_predictions), dim=1) == torch.tensor(all_targets)).float().mean()
print(f"Test accuracy: {test_acc:.4f}")

# Print the first 10 texts output
for i in range(len(all_targets)):
    text_logits = all_text_predictions[i]
    print("Text logits: ", text_logits.shape)  # e.g., (1, 300, vocab_size)
    
    # Convert logits to probabilities and sample token IDs
    text_probs = torch.softmax(torch.tensor(text_logits), dim=-1)
    # Reshape to 2D: (B*T, vocab_size)
    probs_flat = text_probs.view(-1, vocab_size).cpu()
    # Sample tokens using multinomial sampling
    sampled_ids = torch.multinomial(probs_flat, num_samples=1)  # shape: (B*T, 1)
    # Reshape back to (B, T)
    sampled_ids = sampled_ids.view(1, -1)
    
    # Squeeze the batch dimension to get a 1D list and convert tensor to Python list
    sampled_ids_list = sampled_ids.squeeze(0).tolist()
    
    # Decode the list of token IDs (which must be a flat 1D list)
    text_predictions = tokenizer.decode(sampled_ids_list, skip_special_tokens=True)

    print("Diagnosis: ", all_targets[i], "Prediction: ", torch.argmax(torch.tensor(all_predictions[i])))
    # print(f"Generated Text from Image (using softmax) {i+1}: {text_predictions}")
    print(f"Generated Text from Image (using argmax) {i+1}: {text_predictions}")