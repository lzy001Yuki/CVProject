import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features1, features2):
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        
        logits = features1 @ features2.T / self.temperature
        
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss_1 = F.cross_entropy(logits, labels)
        loss_2 = F.cross_entropy(logits.T, labels)
        
        return (loss_1 + loss_2) / 2


class MultiModalContrastiveModel(nn.Module):
    def __init__(
        self,
        autoencoder,
        siglip_model,
        t5_model,
        latent_dim=256,
        embed_dim=768,
        projection_dim=512,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.siglip_model = siglip_model
        self.t5_model = t5_model
        
        for param in self.siglip_model.parameters():
            param.requires_grad = False
        for param in self.t5_model.parameters():
            param.requires_grad = False
        
        self.tactile_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        self.vision_projector = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        self.contrastive_loss = ContrastiveLoss()
        
    def encode_tactile(self, tactile):
        z = self.autoencoder.encode(tactile)
        return self.tactile_projector(z)
    
    def encode_vision(self, vision):
        with torch.no_grad():
            vision_features = self.siglip_model(vision)
        return self.vision_projector(vision_features)
    
    def encode_text(self, text_tokens):
        with torch.no_grad():
            if isinstance(text_tokens, dict):
                text_features = self.t5_model(
                    input_ids=text_tokens.get('input_ids'),
                    attention_mask=text_tokens.get('attention_mask', None),
                )
            else:
                text_features = self.t5_model(text_tokens)
        return self.text_projector(text_features)
    
    def forward(self, tactile, vision, text_tokens):
        tactile_feat = self.encode_tactile(tactile)
        vision_feat = self.encode_vision(vision)
        text_feat = self.encode_text(text_tokens)
        
        loss_tv = self.contrastive_loss(tactile_feat, vision_feat)
        loss_tt = self.contrastive_loss(tactile_feat, text_feat)
        loss_vt = self.contrastive_loss(vision_feat, text_feat)
        
        total_loss = loss_tv + loss_tt + loss_vt
        
        return total_loss, {
            'loss_tactile_vision': loss_tv.item(),
            'loss_tactile_text': loss_tt.item(),
            'loss_vision_text': loss_vt.item(),
        }
