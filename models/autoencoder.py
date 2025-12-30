import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        
        attn = torch.softmax(q @ k.transpose(-2, -1) / (C // self.num_heads) ** 0.5, dim=-1)
        h = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        
        return x + self.proj(h)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, base_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(base_channels, base_channels),
                ResidualBlock(base_channels, base_channels),
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(base_channels * 2, base_channels * 2),
                ResidualBlock(base_channels * 2, base_channels * 2),
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(base_channels * 4, base_channels * 4),
                ResidualBlock(base_channels * 4, base_channels * 4),
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)
            ),
        ])
        
        self.mid_block = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4),
            AttentionBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
        )
        
        self.conv_out = nn.Conv2d(base_channels * 4, latent_dim, 3, padding=1)
        
    def forward(self, x):
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3, base_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, base_channels * 4, 3, padding=1)
        
        self.mid_block = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4),
            AttentionBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
        )
        
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(base_channels * 4, base_channels * 4),
                ResidualBlock(base_channels * 4, base_channels * 4),
                nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(base_channels * 4, base_channels * 2),
                ResidualBlock(base_channels * 2, base_channels * 2),
                nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(base_channels * 2, base_channels),
                ResidualBlock(base_channels, base_channels),
                nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
            ),
        ])
        
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block(h)
        for block in self.up_blocks:
            h = block(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, base_channels=64):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, base_channels)
        self.decoder = Decoder(latent_dim, in_channels, base_channels)
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
