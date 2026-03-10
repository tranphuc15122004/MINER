import torch
import torch.nn as nn
import torch.nn.functional as F


class FastAttention(nn.Module):
    """
    Multi-head self-attention đơn giản:
      - d_model: chiều embedding (phải khớp với news_dim)
      - num_heads: số head, d_model phải chia hết cho num_heads
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        x: [B, L, D]
        attention_mask: [B, L] với 1 = keep, 0 = pad
        """
        B, L, D = x.size()
        H = self.num_heads
        Hd = self.head_dim

        # project
        q = self.q_proj(x).view(B, L, H, Hd).transpose(1, 2)  # [B, H, L, Hd]
        k = self.k_proj(x).view(B, L, H, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Hd).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        if attention_mask is not None:
            mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # True tại pad
            attn_scores = attn_scores.masked_fill(mask, float("-1e9"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)  # [B, H, L, Hd]
        context = context.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]

        out = self.out_proj(context)
        return out


class FastformerLayer(nn.Module):
    """
    1 layer: Self-attn + FFN + residual + LayerNorm
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, ffn_factor: int = 4):
        super().__init__()
        self.self_attn = FastAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)

        hidden_dim = d_model * ffn_factor
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # Self-attention
        attn_out = self.self_attn(x, attention_mask=attention_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # FFN
        ff = self.fc2(F.gelu(self.fc1(x)))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x


class FastformerEncoder(nn.Module):
    """
    Stack nhiều FastformerLayer.
    """
    def __init__(self, d_model: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            FastformerLayer(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


class Fastformer(nn.Module):
    """
    Encoder dùng trong UserEncoder:
      - inputs_embeds: [B, L, D] (D = news_dim)
      - attention_mask: [B, L]
      - output: [B, D] (mean pooling có mask)
    """
    def __init__(self, d_model: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder = FastformerEncoder(d_model, num_layers, num_heads, dropout)
        self.d_model = d_model

    def forward(self, inputs_embeds, attention_mask=None):
        """
        inputs_embeds: [B, L, D]
        attention_mask: [B, L] (1 = thực, 0 = pad)
        """
        x = self.encoder(inputs_embeds, attention_mask=attention_mask)  # [B, L, D]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            x = x * mask
            denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            pooled = x.sum(dim=1) / denom            # [B, D]
        else:
            pooled = x.mean(dim=1)                   # [B, D]

        return pooled
