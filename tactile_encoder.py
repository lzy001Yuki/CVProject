"""tactile_encoder.py

Diffusion-based Tactile Encoder (DiT-style) with conditional cross-attention.

This module implements a transformer encoder for tactile images following the
Diffusion Transformer (DiT) architectural pattern, adapted to produce a latent
representation instead of predicting added noise. The encoder:

1. Adds forward diffusion noise to the tactile image given a timestep t.
2. Patchifies the (noisy) tactile image into a sequence of tokens.
3. Injects a timestep embedding into the patch tokens.
4. Performs a stack of transformer blocks with self-attention.
5. Optionally performs cross-attention against concatenated vision+text
   condition tokens (already encoded elsewhere), allowing the tactile latent
   to align with action/scene context.
6. Returns either all tactile tokens or a pooled latent embedding suitable
   for downstream unified multi-modal modeling.

Key deviation from a standard DiT: the final prediction linear layer (which
would output noise estimates) is intentionally omitted, making this network
an encoder that produces contextual tactile embeddings.

Usage Example:
--------------

    import torch
    from tactile_encoder import DiffusionTactileEncoder

    B = 2
    tactile_img = torch.randn(B, 3, 64, 64)  # (B, C_in, H, W)
    vision_tokens = torch.randn(B, 128, 512) # Example vision tokens
    text_tokens = torch.randn(B, 32, 512)    # Example text tokens
    cond_tokens = torch.cat([vision_tokens, text_tokens], dim=1)
    t = torch.randint(0, 1000, (B,))        # diffusion timesteps

    model = DiffusionTactileEncoder(
        image_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        cond_dim=512,
        timesteps=1000,
    )

    latent = model(tactile_img, t, condition_tokens=cond_tokens)  # (B, embed_dim)
    print(latent.shape)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create standard sinusoidal timestep embeddings.

    Args:
        t: (B,) integer or float timesteps.
        dim: embedding dimension.
    Returns:
        (B, dim) tensor of embeddings.
    """
    half = dim // 2
    device = t.device
    # print(device)
    # log space frequencies
    freq = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), half, device=device)
    )
    # If t is integer, cast to float for multiplication
    t_float = t.float().unsqueeze(1)  # (B, 1)
    # Outer product t * freq
    angles = t_float / freq.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


# -----------------------------------------------------------------------------
# Diffusion schedule helper
# -----------------------------------------------------------------------------

class NoiseSchedule:
    """Linear beta noise schedule with precomputed cumulative products."""

    def __init__(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        self.register_buffer = None  # placeholder for type hints
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self._betas = betas
        self._alphas_cumprod = alphas_cumprod

    @property
    def betas(self) -> torch.Tensor:
        return self._betas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return self._alphas_cumprod

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x0 at timestep t.

        Args:
            x0: clean tactile images (B, C, H, W)
            t: integer timesteps (B,) in [0, timesteps-1]
        Returns:
            noisy_x: (B, C, H, W)
            noise: noise that was added (B, C, H, W)
        """
        # Gather alpha_bar_t for each batch element
        alphas_cumprod = self.alphas_cumprod.to(x0.device)
        alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1, 1)  # (B,1,1,1)
        noise = torch.randn_like(x0)
        noisy_x = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise


# -----------------------------------------------------------------------------
# Transformer components
# -----------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Single Diffusion Transformer block with self-attn, optional cross-attn, and MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, use_cross_attn: bool = True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.self_ln = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        if use_cross_attn:
            self.cross_ln_q = nn.LayerNorm(embed_dim)
            self.cross_ln_kv = nn.LayerNorm(embed_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.mlp_ln = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x_ln = self.self_ln(x)
        self_out, _ = self.self_attn(x_ln, x_ln, x_ln)
        x = residual + self_out

        # Cross-attention if condition provided
        if self.use_cross_attn and cond is not None:
            residual = x
            q = self.cross_ln_q(x)
            kv = self.cross_ln_kv(cond)
            cross_out, _ = self.cross_attn(q, kv, kv)
            x = residual + cross_out

        # MLP
        residual = x
        x_ln = self.mlp_ln(x)
        mlp_out = self.mlp(x_ln)
        x = residual + mlp_out
        return x


# -----------------------------------------------------------------------------
# Patch Embedding
# -----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Conv patch embedding turning image into sequence of patch tokens."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H/ps, W/ps)
        x = self.proj(x)
        B, E, H, W = x.shape
        x = x.view(B, E, H * W).transpose(1, 2)  # (B, N_patches, E)
        return x


# -----------------------------------------------------------------------------
# Main Encoder
# -----------------------------------------------------------------------------

@dataclass
class DiffusionTactileEncoderConfig:
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 3
    embed_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    cond_dim: int = 512
    timesteps: int = 1000
    use_cross_attn: bool = True
    pool: str = "mean"  # "mean" | "cls" | "none"
    # Optional: modality/type embeddings for condition tokens
    use_cond_type_embed: bool = False
    num_cond_modalities: int = 4
    # Noise prediction head for Stage 1 diffusion training
    use_noise_pred_head: bool = False  # True for Stage 1, False for Stage 2


class DiffusionTactileEncoder(nn.Module):
    """Diffusion-based tactile transformer encoder producing latent tactile embeddings.

    Forward signature:
        forward(tactile_img, t, condition_tokens=None, return_tokens=False)

    Args:
        image_size: Size (H=W) of tactile input images.
        patch_size: Patch granularity for embedding.
        in_channels: Channels in tactile image.
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer (DiT) blocks.
        num_heads: Multi-head attention heads.
        mlp_ratio: Expansion ratio for feed-forward layers.
        cond_dim: Dimension of incoming condition tokens (vision+text). Will be projected.
        timesteps: Number of diffusion timesteps supported.
        use_cross_attn: Whether to apply cross-attention with condition tokens.
        pool: Pooling strategy for output: "mean", "cls", or "none" (tokens).

    Returns:
        If pool != "none" and return_tokens=False: (B, embed_dim) latent.
        If return_tokens=True or pool=="none": (B, N_patches (+1 if cls), embed_dim) token tensor.
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        cond_dim: int = 512,
        timesteps: int = 1000,
        use_cross_attn: bool = True,
        pool: str = "mean",
        use_cond_type_embed: bool = False,
        num_cond_modalities: int = 4,
        use_noise_pred_head: bool = False,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.config = DiffusionTactileEncoderConfig(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            cond_dim=cond_dim,
            timesteps=timesteps,
            use_cross_attn=use_cross_attn,
            pool=pool,
            use_cond_type_embed=use_cond_type_embed,
            num_cond_modalities=num_cond_modalities,
            use_noise_pred_head=use_noise_pred_head,
        )

        # Patch embedding
        # print(in_channels, embed_dim, patch_size)
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        # Optional CLS token for pooled representation
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        # Positional embedding (learned) for patch tokens (+1 if CLS)
        num_patches = (image_size // patch_size) ** 2
        total_tokens = num_patches + (1 if self.cls_token is not None else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, embed_dim) * 0.02)

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Project condition tokens to model dim (if cross-attn enabled)
        if use_cross_attn:
            self.cond_proj = nn.Linear(cond_dim, embed_dim)
        else:
            self.cond_proj = None

        # Optional learned type embeddings to distinguish modalities (e.g., vision/text)
        if use_cross_attn and use_cond_type_embed:
            self.cond_type_embed = nn.Embedding(num_cond_modalities, embed_dim)
        else:
            self.cond_type_embed = None

        # Transformer (DiT) blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio, use_cross_attn=use_cross_attn)
            for _ in range(depth)
        ])

        self.final_ln = nn.LayerNorm(embed_dim)

        # Noise prediction head for Stage 1 diffusion training
        # Maps patch tokens back to noise prediction in image space
        if use_noise_pred_head:
            num_patches = (image_size // patch_size) ** 2
            # Linear layer to predict noise for each patch
            self.noise_pred_head = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        else:
            self.noise_pred_head = None

        # Diffusion noise schedule
        self.noise_schedule = NoiseSchedule(timesteps)

    def add_noise(self, tactile_img: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.noise_schedule.add_noise(tactile_img, t)

    def forward(
        self,
        tactile_img: torch.Tensor,
        t: torch.Tensor,
        condition_tokens: Optional[torch.Tensor] = None,
        condition_modality_ids: Optional[torch.Tensor] = None,
        return_tokens: bool = False,
        return_noise_target: bool = False,
    ) -> torch.Tensor:
        """Compute tactile latent representation or noise prediction.

        Args:
            tactile_img: (B, C, H, W) tactile image.
            t: (B,) timesteps in [0, timesteps-1].
            condition_tokens: Optional (B, N_cond, cond_dim) from vision+text encoders.
            condition_modality_ids: Optional (B, N_cond) modality type IDs.
            return_tokens: If True returns per-patch token sequence.
            return_noise_target: If True, also returns the target noise (for Stage 1 training).
        Returns:
            If use_noise_pred_head=True:
                - pred_noise: (B, C, H, W) predicted noise
                - target_noise (optional if return_noise_target=True): (B, C, H, W) actual noise added
            If use_noise_pred_head=False:
                - Latent embedding or token sequence as described in class docstring.
        """
        B = tactile_img.shape[0]
        device = tactile_img.device
        H, W = tactile_img.shape[2], tactile_img.shape[3]

        # 1. Add diffusion noise
        noisy_img, noise_target = self.add_noise(tactile_img, t)

        # 2. Patchify
        x = self.patch_embed(noisy_img)  # (B, N_patches, E)

        # 3. Optionally prepend CLS token
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,E)
            x = torch.cat([cls, x], dim=1)

        # 4. Add position embedding
        x = x + self.pos_embed.to(device)

        # 5. Time embedding and inject (broadcast addition)
        time_emb = sinusoidal_time_embedding(t, self.config.embed_dim)  # (B,E)
        time_emb = self.time_mlp(time_emb)  # (B,E)
        x = x + time_emb.unsqueeze(1)  # (B, T, E)

        # 6. Prepare condition tokens (project + optional positional?)
        cond = None
        if self.config.use_cross_attn and condition_tokens is not None:
            cond = self.cond_proj(condition_tokens)  # (B, N_cond, E)
            # Add modality/type embeddings if provided
            if self.cond_type_embed is not None and condition_modality_ids is not None:
                # condition_modality_ids: (B, N_cond) with int ids in [0, num_cond_modalities)
                cond = cond + self.cond_type_embed(condition_modality_ids.to(torch.long))

        # 7. Transformer blocks
        for blk in self.blocks:
            x = blk(x, cond)

        # 8. Final norm
        x = self.final_ln(x)

        # 9. Noise prediction head (Stage 1) or feature encoding (Stage 2)
        if self.noise_pred_head is not None:
            # Stage 1: Predict noise
            # Remove CLS token if present
            if self.cls_token is not None:
                x = x[:, 1:, :]  # (B, N_patches, E)

            # Predict noise for each patch
            noise_pred_patches = self.noise_pred_head(x)  # (B, N_patches, patch_size^2 * C)

            # Reshape to image space
            ps = self.config.patch_size
            C = self.config.in_channels
            num_patches_h = H // ps
            num_patches_w = W // ps

            # (B, N_patches, ps*ps*C) -> (B, num_patches_h, num_patches_w, ps, ps, C)
            noise_pred_patches = noise_pred_patches.view(B, num_patches_h, num_patches_w, ps, ps, C)
            # Rearrange to (B, C, H, W)
            noise_pred = noise_pred_patches.permute(0, 5, 1, 3, 2, 4).contiguous()
            noise_pred = noise_pred.view(B, C, H, W)

            if return_noise_target:
                return noise_pred, noise_target
            else:
                return noise_pred
        else:
            # Stage 2: Feature encoding (original behavior)
            if return_tokens or self.config.pool == "none":
                return x  # (B, T, E)

            if self.config.pool == "cls":
                return x[:, 0]  # (B, E)
            elif self.config.pool == "mean":
                return x.mean(dim=1)  # (B, E)
            else:
                raise ValueError(f"Unknown pool strategy: {self.config.pool}")
