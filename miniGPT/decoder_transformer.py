import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 16
block_size = 64 # Context window
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embed = 128 # Embedding dimension
n_heads = 4 # Number of attention heads (each head has an embedding size of 64)
n_layers = 4 # Number of layers
dropout = 0.25


with open('input.txt', 'r') as file:
    text = file.read()

# Get the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(len(chars))


# Creating a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars) }
print(stoi)

# Creating a mapping from integers to characters
itos = { i:ch for i, ch in enumerate(chars) }
print(itos)

# Encode the text into integers
test = "hello"
encode = lambda s: [stoi[c] for c in s]
print(encode(test))

decode = lambda l: ''.join([itos[i] for i in l])
print(decode(encode(test)))

#==============================================
# Encode the whole corpus
#==============================================
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(train_data.shape)
print(val_data.shape)

#==============================================
# Function calls
#==============================================
def get_batch(split):
    """
    Randomly samples a batch of data from the dataset
    """
    data = train_data if split == 'train' else val_data
    random_indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in random_indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_indices])
    return x, y


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
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimension
        k = self.key(x) # K is the address for each peice of information (encodes relevance)
        q = self.query(x) # Q is the question or focus of the attention mechanism
        wei = q @ k.transpose(-2, -1) * C**-0.5 # Weights for the attention mechanism
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Masking the weights for the lower triangular part of the matrix
        wei = F.softmax(wei, dim=-1) # Applying the softmax function to the weights
        wei = self.dropout(wei)
        v = self.value(x) # V is the value of the information (encodes content)
        out = wei @ v
        return out
    
class FeedForward(nn.Module):
    """
    Implements a feed-forward layer
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """
    Implements a single block of the transformer
    """
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_head = MultiHeadAttention(num_heads=n_heads, head_size=head_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    """
    Implements multiple heads of self-attention
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx and targets are both (B, T) tensor of integers
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device) % block_size)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None 
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


#==============================================
# Training the model
#==============================================
m = BigramLanguageModel()
xb, yb = get_batch('train')

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for steps in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if steps % 100 == 0:
        x_val, y_val = get_batch('val')
        with torch.no_grad():
            val_logits, val_loss = m(x_val, y_val)
        print(f"step {steps} loss {loss.item()}, val_loss {val_loss.item()}")

print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[0].tolist()))
