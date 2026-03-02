"""
ECG Signal Classifier Model

Multi-scale CNN + Transformer architecture for multi-label ECG classification.
Architecture: Multi-scale Conv1D (k=3,7,15,31) → Downsample → Transformer → MLP

Includes downsampling to reduce sequence length for memory-efficient attention.
"""

import torch
import torch.nn as nn
from torch.nn import Parameter


class DropPath(nn.Module):
    """Stochastic depth regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and stochastic depth."""
    
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, 
                 dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale 1D convolution with residual connection.
    Kernels k=3,7,15,31 capture different temporal resolutions.
    Includes optional downsampling to reduce sequence length.
    """
    
    def __init__(self, num_leads: int = 12, d_model: int = 256, 
                 downsample_factor: int = 10):
        super().__init__()
        self.downsample_factor = downsample_factor
        
        self.layer_norm = nn.LayerNorm(num_leads)
        
        # Multi-scale convolutions
        self.conv_k3 = nn.Conv1d(num_leads, 64, kernel_size=3, padding=1)
        self.conv_k7 = nn.Conv1d(num_leads, 64, kernel_size=7, padding=3)
        self.conv_k15 = nn.Conv1d(num_leads, 64, kernel_size=15, padding=7)
        self.conv_k31 = nn.Conv1d(num_leads, 64, kernel_size=31, padding=15)
        
        # Residual projection
        self.projection = nn.Conv1d(num_leads, d_model, kernel_size=1)
        
        # Downsampling layer (strided conv for learnable downsampling)
        if downsample_factor > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=downsample_factor, 
                         stride=downsample_factor, padding=0),
                nn.GELU()
            )
        else:
            self.downsample = nn.Identity()
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 12] -> [B, T//downsample_factor, d_model]
        """
        B, T, C = x.shape
        
        # Normalize and transpose for conv
        x_norm = self.layer_norm(x).transpose(1, 2)  # [B, 12, T]
        x_t = x.transpose(1, 2)  # [B, 12, T]
        
        # Multi-scale convolutions
        conv_out = torch.cat([
            self.conv_k3(x_norm),
            self.conv_k7(x_norm),
            self.conv_k15(x_norm),
            self.conv_k31(x_norm)
        ], dim=1)  # [B, 256, T]
        
        # Residual connection
        residual = self.projection(x_t)  # [B, d_model, T]
        out = self.activation(residual + conv_out)  # [B, d_model, T]
        
        # Downsample
        out = self.downsample(out)  # [B, d_model, T//factor]
        
        return out.transpose(1, 2)  # [B, T//factor, d_model]


class ECGEmbeddings(nn.Module):
    """CLS token + positional embeddings."""
    
    def __init__(self, d_model: int = 256, max_seq_len: int = 501):
        super().__init__()
        self.cls_token = Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = Parameter(torch.zeros(1, max_seq_len, d_model))
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d] -> [B, T+1, d]"""
        B, T, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
        x = x + self.pos_emb[:, :T+1, :]
        return x


class SignalClassifier(nn.Module):
    """
    Complete ECG classifier: MultiScaleConv -> Downsample -> Embeddings -> Transformer -> MLP
    
    Default config reduces 5000 timesteps to 500 for efficient attention.
    Memory usage: ~2-4GB for batch_size=32
    """
    
    def __init__(self, num_classes: int, d_model: int = 256, num_leads: int = 12,
                 max_seq_len: int = 5000, num_layers: int = 6, num_heads: int = 8,
                 mlp_ratio: int = 4, dropout: float = 0.1, 
                 stochastic_depth: float = 0.1, classifier_dropout: float = 0.1,
                 downsample_factor: int = 10):
        super().__init__()
        
        self.downsample_factor = downsample_factor
        reduced_seq_len = max_seq_len // downsample_factor
        
        # Conv block with downsampling
        self.conv_block = MultiScaleConvBlock(num_leads, d_model, downsample_factor)
        
        # Embeddings (for reduced sequence length)
        self.embeddings = ECGEmbeddings(d_model, reduced_seq_len + 1)
        
        # Transformer blocks with stochastic depth
        dpr = [stochastic_depth * i / (num_layers - 1) for i in range(num_layers)] if num_layers > 1 else [0.0]
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model * 2, num_classes)
        )
        
        self._init_weights()
        
        # Print model info
        print(f"  Model: {max_seq_len} -> {reduced_seq_len} timesteps (downsample {downsample_factor}x)")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 12] -> logits: [B, num_classes]
        
        T=5000 by default, reduced to 500 after conv block
        """
        # Multi-scale conv + downsample: [B, 5000, 12] -> [B, 500, 256]
        x = self.conv_block(x)
        
        # Add CLS token + positional embeddings: [B, 500, 256] -> [B, 501, 256]
        x = self.embeddings(x)
        
        # Transformer blocks
        for block in self.transformer:
            x = block(x)
        
        # Final norm
        x = self.final_norm(x)
        
        # Classify using CLS token
        return self.classifier(x[:, 0])


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing SignalClassifier...")
    
    model = SignalClassifier(
        num_classes=5,
        d_model=256,
        num_layers=6,
        num_heads=8,
        downsample_factor=10
    )
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(4, 5000, 12)  # [batch, time, leads]
    with torch.no_grad():
        out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Test passed!")