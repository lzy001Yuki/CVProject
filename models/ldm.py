import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        autoencoder,
        unet,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        latent_dim=256,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.unet = unet
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        
    def encode_to_latent(self, x):
        with torch.no_grad():
            return self.autoencoder.encode(x)
    
    def decode_from_latent(self, z):
        with torch.no_grad():
            return self.autoencoder.decode(z)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, cond=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.unet(x_noisy, t, cond)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def forward(self, x, cond=None):
        B = x.shape[0]
        device = x.device
        
        z = self.encode_to_latent(x)
        
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        
        loss = self.p_losses(z, t, cond)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, cond=None):
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.unet(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, shape, cond=None, device='cuda'):
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond)
        
        return img
    
    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, device='cuda', ddim_steps=50, eta=0.0):
        b = shape[0]
        
        step_size = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, step_size))
        timesteps.reverse()
        
        img = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            
            pred_noise = self.unet(img, t_tensor, cond)
            
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
            
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * pred_noise
            
            noise = torch.randn_like(img) if i < len(timesteps) - 1 else torch.zeros_like(img)
            img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        return img


class VisionTactileEncoder(nn.Module):
    def __init__(self, ldm, feature_dim=768, pool='mean'):
        super().__init__()
        self.ldm = ldm
        self.pool = pool
        
        for param in self.ldm.parameters():
            param.requires_grad = False
        
        self.feature_proj = nn.Sequential(
            nn.Linear(ldm.latent_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        
    def forward(self, x, t=None, cond=None, return_latent=False):
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        
        with torch.no_grad():
            z = self.ldm.encode_to_latent(x)
            
            z_noisy = self.ldm.q_sample(z, t)
            
            pred_noise = self.ldm.unet(z_noisy, t, cond)
            
            z_denoised = (z_noisy - self.ldm.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1) * pred_noise) / \
                         self.ldm.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        if return_latent:
            return z_denoised
        
        B, C, H, W = z_denoised.shape
        if self.pool == 'mean':
            feat = z_denoised.mean(dim=[2, 3])
        elif self.pool == 'max':
            feat = z_denoised.flatten(2).max(dim=2)[0]
        else:
            feat = z_denoised.flatten(1)
        
        feat = self.feature_proj(feat)
        return feat
