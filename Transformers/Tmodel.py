import torch
import torch.nn as nn
import torch.optim as optim
import math

# Dataset preparation
with open('../weathuringheights.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

encode = lambda s: [char_to_int.get(c, 0) for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

# Model parameters (updated to match sequence length)
max_seq_length = 256  # Now consistent throughout
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
dropout = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, v)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        return self.wo(self.combine_heads(attn_output))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm3(x + self.dropout(ffn_output))


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        return src_mask, tgt_mask & nopeak_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.pos_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.pos_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.fc(dec_output)

    def generate(self, src, start_token, end_token, max_length, temperature=1.0, top_k=None):
        self.eval()
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # Encoder forward
        src_embedded = self.dropout(self.pos_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output, src_mask)

        # Decoder generation
        generated = torch.tensor([[start_token]], dtype=torch.long, device=src.device)
        for _ in range(max_length - 1):
            batch_size, seq_length = generated.size()

            # Create 4D masks
            tgt_padding_mask = (generated != 0).unsqueeze(1).unsqueeze(2)
            nopeak_mask = torch.triu(
                torch.ones((1, 1, seq_length, seq_length), device=src.device),
                diagonal=1
            ).bool()

            tgt_mask = tgt_padding_mask & ~nopeak_mask

            tgt_embedded = self.dropout(self.pos_encoding(self.decoder_embedding(generated)))
            dec_output = tgt_embedded
            for dec_layer in self.decoder:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

            logits = self.fc(dec_output)[:, -1, :] / temperature
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf')).scatter(-1, indices, values)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == end_token:
                break

        return generated


# Initialize model with correct sequence length
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length,  # Now using 256 consistently
    dropout=dropout
)

# Training setup
src_data = torch.randint(1, vocab_size, (64, max_seq_length))
tgt_data = torch.randint(1, vocab_size, (64, max_seq_length))

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Training loop with fixed contiguity
model.train()
for epoch in range(1):
    optimizer.zero_grad()
    output = model(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, vocab_size),  # Fixed contigious->contiguous
                     tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Generation example
input_text = 'descendant'
encoded_input = [char_to_int.get(c, 0) for c in input_text]
context = torch.tensor([encoded_input], dtype=torch.long)

generated = model.generate(
    src=context,
    start_token=1,  # Assuming 1 is start token
    end_token=2,  # Assuming 2 is end token
    max_length=50,
    temperature=0.7,
    top_k=50
)

print("\nGenerated Text:")
print(decode(generated[0].cpu().numpy()))