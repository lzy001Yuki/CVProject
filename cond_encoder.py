from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    SiglipVisionModel,
    AutoTokenizer,
    T5EncoderModel,
)


class SigLIPVisionEncoder:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self._feat_dim = self.model.config.hidden_size

    @property
    def feature_dim(self) -> int:
        return self._feat_dim

    @torch.no_grad()
    def encode_pil(self, images: List["Image.Image"]) -> torch.Tensor:  # type: ignore
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (B, T, D)
        # Prefer pooled features if available, else mean pool tokens
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = hidden.mean(dim=1)
        feats = feats.float()
        return F.normalize(feats, dim=-1)


class T5TextEncoder:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self._feat_dim = self.model.config.d_model

    @property
    def feature_dim(self) -> int:
        return self._feat_dim

    @torch.no_grad()
    def encode_text(self, texts: List[str], max_length: int = 64) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**toks)
        hidden = out.last_hidden_state  # (B, T, D)
        mask = toks.attention_mask.float().unsqueeze(-1)  # (B, T, 1)
        feats = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        feats = feats.float()
        return F.normalize(feats, dim=-1)


class ConditionEncoders:
    """
    Container to provide consistent interface for vision (SigLIP) and text (T5) features.
    """

    def __init__(
        self,
        device: torch.device,
        siglip_model: str = "google/siglip-base-patch16-224",
        t5_model: str = "t5-base",
        use_text: bool = False,
    ) -> None:
        self.device = device
        self.vision = SigLIPVisionEncoder(siglip_model, device)
        self.text = T5TextEncoder(t5_model, device) if use_text else None

    @property
    def image_dim(self) -> int:
        return self.vision.feature_dim

    @property
    def text_dim(self) -> Optional[int]:
        return self.text.feature_dim if self.text is not None else None

    @torch.no_grad()
    def encode_images(self, pil_images: List["Image.Image"]) -> torch.Tensor:  # type: ignore
        return self.vision.encode_pil(pil_images)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if self.text is None:
            raise RuntimeError("Text encoder not initialized. Set use_text=True.")
        return self.text.encode_text(texts)
