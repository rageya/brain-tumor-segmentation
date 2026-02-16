import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "model": {
        "in_channels": 4,
        "out_channels": 4,
        "base_channels": 32,
        "embed_dim": 768,
        "num_heads": 12,
        "depth": 6,
        "patch_size": 16,
    }
}

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=20000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.cached_cos = None
        self.cached_sin = None
    
    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self.cached_cos = None
        
        if self.cached_cos is None or self.cached_cos.size(2) < seq_len:
            t = torch.arange(self.max_seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cached_cos = emb.cos()[None, None, :, :]
            self.cached_sin = emb.sin()[None, None, :, :]
        
        return self.cached_cos[:, :, :seq_len, :], self.cached_sin[:, :, :seq_len, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)

class RoPESelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        cos, sin = self.rope(q, seq_len=N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPESelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class UNETRAdvanced(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config["model"]
        self.patch_size = c["patch_size"]
        self.embed_dim = c["embed_dim"]
        
        self.patch_embed = nn.Conv3d(
            c["in_channels"], 
            self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, c["num_heads"]) 
            for _ in range(c["depth"])
        ])
        
        # Decoder
        self.up1 = nn.ConvTranspose3d(self.embed_dim, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.out_head = nn.Conv3d(16, c["out_channels"], kernel_size=1)
        
        # Deep supervision heads (CORRECTED CHANNEL SIZES)
        # ds_head1 takes input from up3 (32 channels)
        # ds_head2 takes input from up2 (64 channels)
        self.ds_head1 = nn.Conv3d(32, c["out_channels"], kernel_size=1)
        self.ds_head2 = nn.Conv3d(64, c["out_channels"], kernel_size=1)
    
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        for blk in self.blocks:
            x = blk(x)
        
        B, N, C = x.shape
        D = int(round(N ** (1/3)))
        x = x.transpose(1, 2).reshape(B, C, D, D, D)
        
        x1 = self.up1(x)    # 128 channels
        x2 = self.up2(x1)   # 64 channels
        x3 = self.up3(x2)   # 32 channels
        x4 = self.up4(x3)   # 16 channels
        
        # During inference, only return main output
        if self.training:
            return self.out_head(x4), self.ds_head1(x3), self.ds_head2(x2)
        else:
            return self.out_head(x4)
