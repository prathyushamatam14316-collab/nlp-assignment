
# Q2: Mini Transformer Encoder for Sentences
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------- Tokenization ----------------
sentences = [
    "the cat sat on the mat",
    "a dog runs fast",
    "birds fly in the sky",
    "the sun rises early",
    "students read books daily",
    "children play in parks",
    "the moon glows at night",
    "machines learn from data",
    "nature is beautiful",
    "rain falls from clouds"
]

vocab = {}
idx = 0
tokenized = []

for sent in sentences:
    words = sent.split()
    tokens = []
    for w in words:
        if w not in vocab:
            vocab[w] = idx
            idx += 1
        tokens.append(vocab[w])
    tokenized.append(tokens)

max_len = max(len(t) for t in tokenized)

# pad sequences
for t in tokenized:
    while len(t) < max_len:
        t.append(0)

token_ids = torch.tensor(tokenized)

# ---------------- Embedding ----------------
embed_dim = 32
embedding = nn.Embedding(len(vocab), embed_dim)
x = embedding(token_ids)

# ---------------- Sinusoidal Positional Encoding ----------------
def positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / d_model)
    angles = pos * angle_rates

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe

pos_enc = positional_encoding(max_len, embed_dim)
x = x + pos_enc

# ---------------- Multi-Head Self Attention ----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()

        Q = self.Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = scores.softmax(dim=-1)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out), attn_weights

# ---------------- Feed Forward Layer ----------------
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 2)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# ---------------- Transformer Encoder Block ----------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, weights = self.attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, weights

encoder = TransformerEncoder(embed_dim, num_heads=2)
output, weights = encoder(x)

# ------------ Save attention heatmap ------------
plt.imshow(weights[0][0].detach(), cmap="viridis")
plt.colorbar()
plt.title("Attention Heatmap (Head 1)")
plt.savefig("/mnt/data/attention_heatmap.png")

# Save results
torch.save(output, "/mnt/data/final_embeddings.pt")

with open("/mnt/data/input_tokens.txt", "w") as f:
    f.write(str(token_ids))

