import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.cond_proj = nn.Linear(cond_dim, out_channels) if cond_dim else None
        
    def forward(self, x, t_emb, cond=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb[:, :, None, None]
        
        if cond is not None and self.cond_proj is not None:
            cond_emb = self.cond_proj(cond)
            h = h + cond_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        context_dim = context_dim or dim
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x, context=None):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H * W).transpose(1, 2)
        
        if context is None:
            context = x_flat
        
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.to_out(out)
        
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels, context_dim=None, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = CrossAttention(channels, context_dim, num_heads)
        
    def forward(self, x, context=None):
        return x + self.attn(self.norm(x), context)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        base_channels=128,
        time_emb_dim=512,
        cond_dim=768,
        num_heads=8,
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.down1 = nn.ModuleList([
            ResBlock(base_channels, base_channels, time_emb_dim, cond_dim),
            AttentionBlock(base_channels, cond_dim, num_heads),
            ResBlock(base_channels, base_channels, time_emb_dim, cond_dim),
        ])
        self.downsample1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        
        self.down2 = nn.ModuleList([
            ResBlock(base_channels, base_channels * 2, time_emb_dim, cond_dim),
            AttentionBlock(base_channels * 2, cond_dim, num_heads),
            ResBlock(base_channels * 2, base_channels * 2, time_emb_dim, cond_dim),
        ])
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        self.mid = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels * 2, time_emb_dim, cond_dim),
            AttentionBlock(base_channels * 2, cond_dim, num_heads),
            ResBlock(base_channels * 2, base_channels * 2, time_emb_dim, cond_dim),
        ])
        
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResBlock(base_channels * 4, base_channels * 2, time_emb_dim, cond_dim),
            AttentionBlock(base_channels * 2, cond_dim, num_heads),
            ResBlock(base_channels * 2, base_channels, time_emb_dim, cond_dim),
        ])
        
        self.upsample1 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels, time_emb_dim, cond_dim),
            AttentionBlock(base_channels, cond_dim, num_heads),
            ResBlock(base_channels, base_channels, time_emb_dim, cond_dim),
        ])
        
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(self, x, t, cond=None):
        t_emb = self.time_mlp(t)
        
        h = self.conv_in(x)
        
        h1 = h
        for layer in self.down1:
            if isinstance(layer, ResBlock):
                h1 = layer(h1, t_emb, cond)
            else:
                h1 = layer(h1, cond)
        h1_skip = h1
        h1 = self.downsample1(h1)
        
        h2 = h1
        for layer in self.down2:
            if isinstance(layer, ResBlock):
                h2 = layer(h2, t_emb, cond)
            else:
                h2 = layer(h2, cond)
        h2_skip = h2
        h2 = self.downsample2(h2)
        
        h = h2
        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb, cond)
            else:
                h = layer(h, cond)
        
        h = self.upsample2(h)
        h = torch.cat([h, h2_skip], dim=1)
        for layer in self.up2:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb, cond)
            else:
                h = layer(h, cond)
        
        h = self.upsample1(h)
        h = torch.cat([h, h1_skip], dim=1)
        for layer in self.up1:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb, cond)
            else:
                h = layer(h, cond)
        
        h = self.conv_out(h)
        return h
