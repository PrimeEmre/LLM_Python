import torch
import torch.nn as nn
import torch.nn.functional as F

# ── reading the file ──────────────────────────────────────────
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total characters in dataset: {len(text)}")

# ── getting the characters ────────────────────────────────────
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique characters in dataset: {vocab_size}")

# ── converting characters to numbers ─────────────────────────
chars_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char  = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [chars_to_idx[c] for c in s]

def decode(l):
    return ''.join([idx_to_char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ── settings ──────────────────────────────────────────────────
block_size    = 64
batch_size    = 32
embed_size    = 64
num_heads     = 4
num_layers    = 3
dropout       = 0.1
learning_rate = 3e-4
epochs        = 3000
device        = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# ── creating batches ──────────────────────────────────────────
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ── building the model ────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query   = nn.Linear(embed_size, head_size, bias=False)
        self.key     = nn.Linear(embed_size, head_size, bias=False)
        self.value   = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C  = x.shape
        q        = self.query(x)
        k        = self.key(x)
        scores   = q @ k.transpose(-2, -1) * (C ** -0.5)
        scores   = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights  = F.softmax(scores, dim=-1)
        weights  = self.dropout(weights)
        v        = self.value(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        head_size      = embed_size // num_heads
        self.attention = MultiHeadAttention(num_heads, head_size)
        self.ff        = FeedForward(embed_size)
        self.norm1     = nn.LayerNorm(embed_size)
        self.norm2     = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.blocks             = nn.Sequential(*[TransformerBlock(embed_size, num_heads) for _ in range(num_layers)])
        self.norm               = nn.LayerNorm(embed_size)
        self.output_head        = nn.Linear(embed_size, vocab_size)

    def forward(self, x, targets=None):
        B, T      = x.shape
        token_emb = self.token_embedding(x)
        pos_emb   = self.position_embedding(torch.arange(T, device=device))
        x         = token_emb + pos_emb
        x         = self.blocks(x)
        x         = self.norm(x)
        logits    = self.output_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            x_cropped = x[:, -block_size:]
            logits, _ = self(x_cropped)
            logits    = logits[:, -1, :]
            probs     = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            x         = torch.cat([x, next_char], dim=1)
        return x

# ── training the model ────────────────────────────────────────
model     = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")
print("\nStarting training...\n")

for step in range(epochs):
    x, y = get_batch("train")

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}/{epochs} — Loss: {loss.item():.4f}")

print("\nTraining completed!")
torch.save(model.state_dict(), "model.pt")
print("Model saved to model.pt")

# ── generating text ───────────────────────────────────────────
print("\nGenerating text...\n")
print("─" * 50)

start     = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(start, max_new_tokens=300)
print(decode(generated[0].tolist()))

print("─" * 50)